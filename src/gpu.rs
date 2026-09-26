//! The CUDA forward pass.
//!
//! This mirrors `net::forward_cpu` stage for stage, so the two can be compared
//! op by op on random data (`--cuda-selftest` does exactly that).
//!
//! THREE THINGS MAKE A HOST/DEVICE MISMATCH HERE SILENT RATHER THAN LOUD, and
//! all three are why every launch goes through one small set of helpers:
//!
//!   1. A grid that is too small leaves outputs UNWRITTEN and reports success.
//!      Every geometry comes from `grid_for` or from the tile constants in this
//!      file, never from a literal at a call site.
//!   2. An argument of the wrong WIDTH (`i32` where a kernel reads a `long`) is
//!      read from the wrong slot - a wrong number, not an error. `Args` boxes
//!      values by type, so the mistake has to be typed out.
//!   3. A kernel that reads the wrong AXIS still produces an image. That is what
//!      `lg_channel_layer_norm` versus `lg_layer_norm` is: NAFBlock needs the
//!      channel-axis one, which works ONE THREAD PER SPATIAL POSITION over the
//!      channels at that position (grid `hw / BLOCK`), while `lg_layer_norm`
//!      normalises the contiguous width. A grid sized by `c` - or by `hw`
//!      blocks, under the current kernel - still runs and still writes output;
//!      it just leaves most of the tensor untouched.
//!
//! THE RESIDUALS ARE ONE LAUNCH, AND THAT NEEDED A PROJECT KERNEL. The block's
//! `inp + x * beta` was `lg_channel_scale` into a scratch followed by `lg_add`,
//! because the toolkit's `lg_add_scaled` takes a WHOLE-PLANE scalar and beta is
//! per channel - and a second scratch to hold the scaled plane. `nf_residual`
//! does it in one pass over three buffers instead of two passes over four, and
//! drops 72 launches from the pass. It has to use explicit rounding intrinsics:
//! see its comment in `cuda/nafnet.cu`.
//!
//! THE DOWNSAMPLE IS ITS OWN KERNEL, AND CANNOT BE A PADDED 3x3. The obvious
//! shortcut - zero-pad the 2x2 taps into a 3x3 and reuse `lg_conv3x3s1p1` - does
//! not work here, because that kernel's READ offsets are fixed at stride 1: it
//! would compute `in[y+ky-1][x+kx-1]`, and matching `in[2y+ky][2x+kx]` needs the
//! input nearest-upsampled first. So `nf_down2x2s2` is a real kernel. See its
//! comment in cuda/nafnet.cu, which records the attempt.
use crate::cuda::{grid_for, Cuda};
use crate::net::Geometry;
use crate::weights::Weights;
use lightgpu::vm::{Args, DevBuf, Event, Launch};
use std::cell::{Cell, RefCell};

/// Threads per block for the flat elementwise kernels.
const BLOCK: usize = 256;
/// Output channels per thread in `nf_conv1x1_oc`. MUST match NF_OC_TILE in
/// cuda/nafnet.cu - the kernel's `acc[]` is sized by it, so a mismatch is a
/// wrong answer rather than a failed launch.
const OC_TILE: usize = 16;
/// Positions per thread along x in `nf_conv1x1_oc`. MUST match NF_OC_SP in
/// cuda/nafnet.cu, and the launch grid below is sized by it.
const OC_SP: usize = 4;
/// Output channels per thread in `nf_down2x2s2`. MUST match NF_DW_TILE in
/// cuda/nafnet.cu, which sizes the kernel's `acc[]` by it.
const DW_OC_TILE: usize = 16;
/// The tile cuda/nafnet.cu's kernels use: 32 columns x 8 rows per 256-thread
/// block. MUST match NF_DW_TX / NF_DW_TY in that file.
const TX: usize = 32;
const TY: usize = 8;

/// One device activation: [c][h][w].
///
/// A DA is either the OWNER of its `DevBuf` or a view into another allocation
/// (SimpleGate's two halves, and the pooled vector that the sca 1x1 conv reads).
/// `DevBuf` has no alias constructor and frees on drop, so a view has to be
/// built by hand - and a hand-built DevBuf must never be dropped.
///
/// `holds` IS LOAD-BEARING, VIA THE `Drop` BELOW. It began as an unused flag,
/// which made every view a free in disguise: `clone_meta()` (which the pass
/// hands to `block`) and `view()` (which offsets the pointer) were dropped at
/// the end of their scope, and `DevBuf`'s own `Drop` calls `cuMemFree`
/// unconditionally - on the Plan's buffer, or on an INTERIOR pointer. The
/// driver's allocation table loses exactly the entries that were freed that
/// way, and the symptom is a `cuMemcpyDtoD` refused with
/// `CUDA_ERROR_INVALID_VALUE` for a buffer the plan still holds. A per-op
/// selftest cannot catch it because it builds no DA at all.
struct DA {
    buf: DevBuf,
    c: usize,
    h: usize,
    w: usize,
    /// True when `buf` is this DA's own allocation.
    holds: bool,
    /// Where this DA came from. Plan slots carry their slot name; everything
    /// else carries the site that allocated it. Read by the drop-time
    /// diagnostics below - a free that names only a pointer is not traceable.
    tag: &'static str,
}

impl DA {
    /// A view of the SAME allocation at a different shape.
    ///
    /// Used for the shared `pre`/`up` slots, whose buffer is sized to the largest
    /// decoder level and is reused by every smaller one. The kernels index a
    /// plane linearly by `h * w`, so a smaller shape is simply the front of the
    /// same buffer - and because the levels are strictly ordered, the bytes a
    /// smaller level writes are the bytes the level before it wrote, which is
    /// what makes the sharing safe rather than merely small.
    ///
    /// `holds` stays false, so dropping this cannot free the slot.
    fn resized(&self, c: usize, h: usize, w: usize) -> DA {
        DA {
            holds: false,
            buf: DevBuf { ptr: self.buf.ptr, bytes: c * h * w * 4 },
            c,
            h,
            w,
            tag: "resized slot view",
        }
    }

    /// The same view metadata, without touching the allocation. Used to hand a
    /// slot's buffer to `block`, which reads and writes it in place.
    fn clone_meta(&self) -> DA {
        DA { buf: DevBuf { ptr: self.buf.ptr, bytes: self.buf.bytes }, c: self.c, h: self.h, w: self.w, holds: false, tag: "clone_meta" }
    }
}

/// Free ONLY what this DA owns.
///
/// THE CLEARING HAPPENS ON BOTH PATHS, AND THAT IS THE WHOLE POINT. `buf` is a
/// `DevBuf`, whose own `Drop` calls `cuMemFree`; Rust drops a struct's fields
/// AFTER its `Drop::drop` returns. So an early `return` here for a view left
/// `self.buf` holding the real `(ptr, bytes)` of a buffer the view does not own,
/// and the field drop freed it. That is not a theoretical hazard: a view built
/// with channel offset 0 (`Self::view(&t.t3.buf, half, h, wd, 0)`, SimpleGate's
/// first half) has `(ptr, bytes)` bit-identical to its source, so dropping it
/// freed the slot the next `copy_d2d` was about to read - which the driver then
/// answered with CUDA_ERROR_INVALID_VALUE, three ops away from the cause.
///
/// A view's `DevBuf` is a hand-built (ptr, bytes) pair pointing into someone
/// else's allocation - often at an offset - and freeing it corrupts the
/// driver's view of the heap rather than failing loudly.
impl Drop for DA {
    fn drop(&mut self) {
        if std::env::var("NAFNET_TRACE_DA_DROP").is_ok() {
            eprintln!(
                "DA drop: ptr=0x{:x} bytes={} holds={} tag={}",
                self.buf.ptr, self.buf.bytes, self.holds, self.tag
            );
        }
        // KEEP THE SWITCH: it is the cheapest way to answer "who freed this"
        // when a copy fails against an address the plan still lists.
        // Take the buffer out on BOTH paths: an owned one so `DevBuf::drop`
        // frees exactly it, a view's so the field holds nothing freeable by the
        // time the field drop runs.
        let buf = std::mem::replace(&mut self.buf, DevBuf { ptr: 0, bytes: 0 });
        if self.holds {
            drop(buf);
        } else {
            // A LOCAL BINDING IS DROPPED WHEN THE FUNCTION RETURNS, `if` OR NOT.
            // `buf` here is the VIEW's real `(ptr, bytes)` - the plan's buffer,
            // often at an offset - so leaving it to the implicit drop freed the
            // very allocation the next `Plan::copy` was about to read. That is
            // why the flag and the trace disagreed: `holds == false` correctly
            // skipped `drop(buf)`, and then Rust dropped `buf` anyway.
            std::mem::forget(buf);
        }
    }
}

impl DA {
    /// ONE CONTRACT WORTH STATING, because nothing checks it: every kernel
    /// argument that describes a size or a count is an `i32` at the call sites
    /// below, so a plane with more than 2^31 elements would be passed to CUDA
    /// truncated rather than rejected. nafnet is fully convolutional, so the
    /// limits are the caller's image and the checkpoint's width - the largest
    /// single activation the released configs can reach is ~10^7 elements - and a
    /// 2^31-element plane would need 8 GB before anything else was allocated. The
    /// same `i32` contract is in the kernels' own signatures.
    fn new(c: usize, h: usize, w: usize) -> Result<DA, String> {
        Ok(DA { buf: DevBuf::zeros(c * h * w * 4)?, c, h, w, holds: true, tag: "DA::new" })
    }
    fn hw(&self) -> usize {
        self.h * self.w
    }
    fn n(&self) -> usize {
        self.c * self.hw()
    }
}

/// Per-launch timings, filled when `--profile` is on.
///
/// WHY CUDA EVENTS AND NOT A HOST CLOCK: every launch here is asynchronous, so a
/// `Instant::now()` pair around one measures the cost of the CALL, not of the
/// kernel - and the interesting number is the kernel. An event pair is recorded
/// ON THE STREAM, so it brackets the work itself.
///
/// WHY THE SPANS ARE READ OUT AT THE END: `cuEventElapsedTime` needs both events
/// complete. Synchronising per launch would serialise the very pipeline the
/// measurement is about, and on this graph the pipeline is the whole question.
/// So spans accumulate and are resolved once, after the pass.
#[derive(Default)]
pub struct Profile {
    spans: RefCell<Vec<(String, Event, Event)>>,
    /// Wall time of the bracketed region, in milliseconds.
    pub wall: Cell<f32>,
}

impl Profile {
    /// Record the opening event of a span. Returns `None` if events are
    /// unavailable, in which case the caller's `close` is a no-op and the run
    /// continues unprofiled rather than failing.
    fn open(&self, name: &str) -> Option<(String, Event)> {
        let e = Event::new().ok()?;
        e.record().ok()?;
        Some((name.to_string(), e))
    }

    fn close(&self, open: Option<(String, Event)>) {
        let Some((name, e0)) = open else { return };
        let Ok(e1) = Event::new() else { return };
        if e1.record().is_err() {
            return;
        }
        self.spans.borrow_mut().push((name, e0, e1));
    }

    /// Aggregate by name and print, longest first.
    ///
    /// DEVELOPMENT ONLY: printing a table is the whole point of `--profile`, and
    /// a release build has no flag that turns profiling on.
    #[cfg(feature = "dev")]
    pub fn report(&self) {
        let spans = self.spans.borrow();
        let mut by: std::collections::BTreeMap<&str, (f32, usize)> =
            std::collections::BTreeMap::new();
        let mut total = 0.0f32;
        for (name, a, b) in spans.iter() {
            let ms = a.elapsed_ms(b).unwrap_or(0.0);
            total += ms;
            let e = by.entry(name.as_str()).or_insert((0.0, 0));
            e.0 += ms;
            e.1 += 1;
        }
        let mut rows: Vec<(&str, f32, usize)> =
            by.into_iter().map(|(k, (t, n))| (k, t, n)).collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let wall = self.wall.get();
        println!("{:>10} {:>6} {:>10}", "total ms", "count", "mean ms");
        for (name, t, n) in &rows {
            println!("{:>10.2} {:>6} {:>9.3}  {name}", t, n, t / (*n as f32));
        }
        println!("{:>10.2} {:>6} {:>9.3}  (all spans)", total, spans.len(), total / spans.len().max(1) as f32);
        println!("{:>10.2} {:>6} {:>9}  (pass wall)", wall, 1, "");
        println!(
            "{:>10.2} {:>6} {:>9}  (unattributed)",
            wall - total,
            1,
            ""
        );
    }
}

pub struct Gpu {
    cuda: Cuda,
    geo: Geometry,
    /// Weights, device-resident, keyed by checkpoint name. Uploaded once: the
    /// model is 17M parameters and re-uploading per launch would dominate.
    wdev: std::collections::HashMap<String, DevBuf>,
    pub profile: Option<Profile>,
}

impl Gpu {
    /// THE ONLY CONSTRUCTOR, AND THE THIRD ARGUMENT IS NOT OPTIONAL. There used
    /// to be a `Gpu::new` that hard-coded `profiling = false`; the CLI used it,
    /// `--profile` was therefore inert, and the flag's help text advertised a
    /// per-kernel GPU profile that did not exist. A convenience wrapper that
    /// silently picks the false branch of a diagnostic is a trap, so the choice
    /// is now made once, at the call site, and `--profile` reaches this
    /// function from exactly one place.
    pub fn with_profile(weights: &Weights, geo: Geometry, profiling: bool) -> Result<Gpu, String> {
        let cuda = Cuda::new()?;
        let mut wdev = std::collections::HashMap::new();
        for name in weights.file.order() {
            let v = weights.file.f32(name)?;
            wdev.insert(name.clone(), cuda.upload(v)?);
        }
        Ok(Gpu {
            cuda,
            geo,
            wdev,
            profile: if profiling { Some(Profile::default()) } else { None },
        })
    }

    pub fn device_name(&self) -> String {
        lightgpu::vm::device().map(|d| d.name).unwrap_or_else(|_| "?".into())
    }

    fn w(&self, name: &str) -> Result<u64, String> {
        self.wdev
            .get(name)
            .map(|b| b.ptr)
            .ok_or_else(|| format!("weight `{name}` was never uploaded"))
    }

    fn go(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &mut Args,
    ) -> Result<(), String> {
        // THE SPAN WRAPS THE LAUNCH ONLY, and the closing event is recorded after
        // the driver has enqueued the kernel - so what it measures is the
        // kernel's execution on the stream, not this function's call overhead.
        let t = self.profile.as_ref().and_then(|p| p.open(name));
        let r = self.cuda.run(name, Launch::new(grid, block), args);
        if let Some(p) = &self.profile {
            p.close(t);
        }
        r?;
        // CUDA reports an asynchronous failure at the NEXT API call, so without
        // a sync here a bad launch inside a block is blamed on whatever copy
        // happens to run afterwards. With NAFNET_SYNC_LAUNCH the error names the
        // kernel that caused it.
        if std::env::var("NAFNET_SYNC_LAUNCH").is_ok() {
            lightgpu::vm::sync().map_err(|e| format!("launch {name}: {e}"))?;
        }
        Ok(())
    }

    // ---- the kernels the graph is built from ------------------------------

    /// 1x1 conv: `nf_conv1x1_oc(in, w, bias, out, c_in, c_out, h, wd)`.
    /// A null bias is handled by the kernel (`bias ? bias[oc] : 0`), which is
    /// how the bias-free upsample conv is launched.
    ///
    /// THIS IS THE ENGINE'S OWN KERNEL, NOT THE TOOLKIT'S `lg_conv1x1`, AND THE
    /// REASON IS THE PROFILE. `lg_conv1x1` gives one thread one output element,
    /// so the input is walked once per output channel: on the 1280x736 pass it
    /// measured 4000 ms of a 5100 ms forward - 78% of it, over 184 launches, at
    /// 21.7 ms each - because `c_out` passes over a 120 MB input is 7.7 GB of
    /// DRAM traffic for a 241 MB output, and 7.7 GB at this device's ~320 GB/s
    /// is 24 ms. `nf_conv1x1_oc` blocks over output channels instead, so one
    /// input load feeds `OC_TILE` fused multiply-adds and the input is read
    /// `ceil(c_out / OC_TILE)` times. See its comment in cuda/nafnet.cu, which
    /// also records why the accumulation order - and therefore the result - is
    /// unchanged.
    ///
    /// The toolkit kernel stays listed and resolved: it is still what the
    /// selftest compares this one against.
    fn conv1x1(&self, input: &DA, w: &str, b: Option<&str>, c_in: usize, c_out: usize, out: &DA) -> Result<(), String> {
        let bias = match b {
            Some(n) => self.w(n)?,
            None => 0,
        };
        let mut a = Args::new();
        a.ptr(input.buf.ptr)
            .ptr(self.w(w)?)
            .ptr(bias)
            .ptr(out.buf.ptr)
            .i32(c_in as i32)
            .i32(c_out as i32)
            .i32(input.h as i32)
            .i32(input.w as i32);
        // x covers hw/(OC_TILE x OC_SP) THREADS, not positions - the kernel
        // strides NF_OC_SP of them per thread.
        let grid = (
            grid_for(input.hw().div_ceil(OC_SP), BLOCK).0,
            c_out.div_ceil(OC_TILE) as u32,
            1,
        );
        self.go("nf_conv1x1_oc", grid, (BLOCK as u32, 1, 1), &mut a)
    }

    /// 3x3 pad-1 conv: `lg_conv3x3s1p1(in, w, bias, out, c_in, c_out, h, wd)`.
    fn conv3x3(&self, input: &DA, w: &str, b: Option<&str>, c_in: usize, c_out: usize, out: &DA) -> Result<(), String> {
        let bias = match b {
            Some(n) => self.w(n)?,
            None => 0,
        };
        let mut a = Args::new();
        a.ptr(input.buf.ptr)
            .ptr(self.w(w)?)
            .ptr(bias)
            .ptr(out.buf.ptr)
            .i32(c_in as i32)
            .i32(c_out as i32)
            .i32(input.h as i32)
            .i32(input.w as i32);
        self.go("lg_conv3x3s1p1", grid_for(c_out * input.hw(), BLOCK), (BLOCK as u32, 1, 1), &mut a)
    }

    /// Depthwise 3x3 (project kernel): `nf_conv3x3_dw(in, w, bias, out, c, h, wd)`,
    /// grid (ceil(wd/TX), ceil(h/TY), c), block (TX, TY).
    /// `c_in` is passed explicitly rather than read off `input`, so that a
    /// caller holding a VIEW (SimpleGate's half, the pooled vector) cannot
    /// silently launch a grid sized by the view's channel count while the
    /// weights describe something else.
    fn conv3x3_dw(&self, input: &DA, w: &str, b: &str, c_in: usize, out: &DA) -> Result<(), String> {
        let (h, wd, c) = (input.h, input.w, c_in);
        let mut a = Args::new();
        a.ptr(input.buf.ptr)
            .ptr(self.w(w)?)
            .ptr(self.w(b)?)
            .ptr(out.buf.ptr)
            .i32(c as i32)
            .i32(h as i32)
            .i32(wd as i32);
        let grid = (
            wd.div_ceil(TX) as u32,
            h.div_ceil(TY) as u32,
            c as u32,
        );
        self.go("nf_conv3x3_dw", grid, (TX as u32, TY as u32, 1), &mut a)
    }

    /// The downsample (project kernel):
    /// `nf_down2x2s2(in, w, bias, out, c_in, c_out, h, wd)`.
    ///
    /// ONE THREAD PER OUTPUT POSITION AND A BLOCK ROW PER CHANNEL TILE, which is
    /// what changed: the kernel used to give one thread one output ELEMENT, so
    /// it walked the input once per output channel - and here the arithmetic
    /// predicts the cost to within 2% (4 x 7.7 GB = 31 GB = ~96 ms at this
    /// device's ~320 GB/s against a measured 98 ms). See its comment in
    /// cuda/nafnet.cu.
    fn downsample(&self, input: &DA, l: usize, c_in: usize, out: &DA) -> Result<(), String> {
        let wname = format!("downs.{l}.weight");
        let bname = format!("downs.{l}.bias");
        let mut a = Args::new();
        a.ptr(input.buf.ptr)
            .ptr(self.w(&wname)?)
            .ptr(self.w(&bname)?)
            .ptr(out.buf.ptr)
            .i32(c_in as i32)
            .i32(out.c as i32)
            .i32(input.h as i32)
            .i32(input.w as i32);
        // x covers the output PLANE, y covers the channel tiles.
        let grid = (
            grid_for(out.hw(), BLOCK).0,
            out.c.div_ceil(DW_OC_TILE) as u32,
            1,
        );
        self.go("nf_down2x2s2", grid, (BLOCK as u32, 1, 1), &mut a)
    }

    /// LayerNorm over the channel axis: `lg_channel_layer_norm(x, w, b, y, c, hw, eps)`.
    ///
    /// THE WORK ITEM IS A SPATIAL POSITION, NOT A CHANNEL. This is the one
    /// toolkit reduction whose extent is the per-channel LENGTH: one THREAD
    /// handles one position `p` and walks the `c` values spaced `hw` apart to
    /// normalise them together. So the grid is `hw / BLOCK` - sized by the plane,
    /// never by `c` - and `BLOCK` is free to be any width, since it no longer
    /// relates to how many channels there are.
    ///
    /// A WRONG EXTENT HERE IS SILENT. A grid of `c`, or a grid of `hw` blocks
    /// under the current one-thread-per-position kernel, still runs, still
    /// reports success, and leaves most of the tensor at whatever the buffer
    /// held - there is no assertion that the launch covers the plane. That is how
    /// the first selftest run reported a 2.6 absolute MISMATCH.
    ///
    /// The kernel reduces the variance as `E[x^2] - mean^2` rather than the
    /// two-pass form the CPU twin uses, so the two differ by whatever that
    /// cancellation costs; on NAFNet tensors it is ~1e-6 relative.
    fn channel_layer_norm(&self, input: &DA, w: &str, b: &str, out: &DA) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(input.buf.ptr)
            .ptr(self.w(w)?)
            .ptr(self.w(b)?)
            .ptr(out.buf.ptr)
            .i32(input.c as i32)
            .i32(input.hw() as i32)
            // The reference's `LayerNorm2d` default, and the same constant the
            // CPU twin uses - a different eps here is a slightly different image
            // rather than an error.
            .f32(crate::net::LAYERNORM_EPS);
        // GRID IS `hw / BLOCK`, ONE THREAD PER SPATIAL POSITION. The kernel used
        // to be one BLOCK per position, reducing the channels with a shared-memory
        // halving tree; it is now a flat one-thread-per-position kernel, because
        // at c=32 the block form left 224 of every 256 threads accumulating
        // nothing and then idling through the tree, and its `xp[i * hw]` gather
        // strided a whole plane per thread. Between 3x and 8.9x faster on every
        // shape this family has.
        //
        // GETTING THE GRID WRONG IS SILENT: `(hw, 1, 1)` still launches, still
        // succeeds, and processes only the first `blockDim` positions - leaving
        // every later position at whatever the buffer held. Nothing here asserts
        // the coverage, so the two sites that launch this kernel (this one and the
        // selftest's) have to move together.
        self.go(
            "lg_channel_layer_norm",
            ((input.hw() as u32).div_ceil(BLOCK as u32), 1, 1),
            (BLOCK as u32, 1, 1),
            &mut a,
        )
    }

    /// `lg_channel_mean(x, out, c, hw)`, grid (c, 1, 1). The kernel reduces with
    /// strided partials and a halving tree, so its sum order - and therefore its
    /// rounding - is not the CPU's serial one.
    fn channel_mean(&self, input: &DA, out: &DevBuf) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(input.buf.ptr)
            .ptr(out.ptr)
            .i32(input.c as i32)
            .i32(input.hw() as i32);
        self.go("lg_channel_mean", (input.c as u32, 1, 1), (BLOCK as u32, 1, 1), &mut a)
    }

    /// `lg_mul(a, b, y, n)` - SimpleGate, once the two halves are views.
    fn mul(&self, a_in: &DA, b_in: &DA, out: &DA) -> Result<(), String> {
        let n = out.n();
        // THE COUNT COMES FROM THE OUTPUT AND THE KERNEL WALKS ALL THREE
        // OPERANDS TO IT, so an operand of a different shape is read out of
        // bounds rather than computed wrongly - and the driver reports the
        // illegal address at the NEXT synchronising call, which is one kernel
        // away from the cause. Two of this kernel's three call sites pass
        // VIEWS, whose shape is not visible at the launch, so this is checked
        // rather than assumed.
        if a_in.n() != n || b_in.n() != n {
            return Err(format!(
                "lg_mul: {}x{}x{} and {}x{}x{} into {}x{}x{} - the count is the output's",
                a_in.c, a_in.h, a_in.w, b_in.c, b_in.h, b_in.w, out.c, out.h, out.w
            ));
        }
        let mut a = Args::new();
        a.ptr(a_in.buf.ptr).ptr(b_in.buf.ptr).ptr(out.buf.ptr).i32(n as i32);
        self.go("lg_mul", grid_for(n, BLOCK), (BLOCK as u32, 1, 1), &mut a)
    }

    /// `lg_channel_scale(in, s, out, c, hw)`. `s` is any `[c]` device vector: a
    /// weight uploaded at load, a pooled attention vector, or a `beta`/`gamma`
    /// read straight out of the weight map.
    fn channel_scale(&self, x: &DA, s: &DevBuf, out: &DA) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.buf.ptr).ptr(s.ptr).ptr(out.buf.ptr).i32(x.c as i32).i32(x.hw() as i32);
        self.go("lg_channel_scale", grid_for(x.c * x.hw(), BLOCK), (BLOCK as u32, 1, 1), &mut a)
    }

    /// `nf_residual(a, b, s, out, c, hw)` - `out = a + b * s[ch]`, the per-channel
    /// residual. See the kernel comment for why it is not `lg_add_scaled` (which
    /// takes a whole-plane scalar) and for why it uses explicitly-rounded
    /// intrinsics (nvcc would otherwise contract the multiply-add and round once
    /// instead of twice, changing the result).
    fn residual(&self, a_in: &DA, b_in: &DA, s: &str, out: &DA) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(a_in.buf.ptr)
            .ptr(b_in.buf.ptr)
            .ptr(self.w(s)?)
            .ptr(out.buf.ptr)
            .i32(out.c as i32)
            .i32(out.hw() as i32);
        self.go(
            "nf_residual",
            grid_for(out.c * out.hw(), BLOCK),
            (BLOCK as u32, 1, 1),
            &mut a,
        )
    }

    /// `lg_add(a, b, y, n)`.
    fn add(&self, a_in: &DA, b_in: &DA, out: &DA) -> Result<(), String> {
        let n = out.n();
        let mut a = Args::new();
        a.ptr(a_in.buf.ptr).ptr(b_in.buf.ptr).ptr(out.buf.ptr).i32(n as i32);
        self.go("lg_add", grid_for(n, BLOCK), (BLOCK as u32, 1, 1), &mut a)
    }

    /// Depth-to-space (project kernel): `nf_pixel_shuffle2(in, out, c, h, wd)`.
    fn pixel_shuffle2(&self, input: &DA, c: usize, out: &DA) -> Result<(), String> {
        let (h, wd) = (input.h, input.w);
        let (oh, ow) = (2 * h, 2 * wd);
        let mut a = Args::new();
        a.ptr(input.buf.ptr).ptr(out.buf.ptr).i32(c as i32).i32(h as i32).i32(wd as i32);
        let grid = (ow.div_ceil(TX) as u32, oh.div_ceil(TY) as u32, c as u32);
        self.go("nf_pixel_shuffle2", grid, (TX as u32, TY as u32, 1), &mut a)
    }

    /// A sub-view of a device buffer, as an activation. The buffers the block
    /// works in are one allocation with the halves of the channel axis named,
    /// which is what makes SimpleGate a view rather than a copy.
    fn view(buf: &DevBuf, c: usize, h: usize, w: usize, ch_off: usize) -> DA {
        // `DevBuf` is a (ptr, bytes) pair with no notion of shape, so a channel
        // range is a pointer offset - and because nothing here frees a sub-view,
        // the base buffer's `Drop` is the only one that runs.
        let off = ch_off * h * w * 4;
        DA {
            // A VIEW, not an owner: dropping this DA must not free the buffer
            // it points into. The buffer belongs to the Plan.
            holds: false,
            buf: DevBuf { ptr: buf.ptr + off as u64, bytes: buf.bytes - off },
            c,
            h,
            w,
            tag: "view",
        }
    }

    // ---- one NAFBlock -----------------------------------------------------

    /// The block, mirroring `net::block_forward` op for op.
    ///
    /// THE SCRATCH IS SIZED BY THE LIVE SET, NOT BY THE ROLE LIST. `BlockScratch`
    /// used to hold nine planes - t1 t2 t3 g y t4 t5 pooled att - all of them
    /// resident for the whole pass, at every shape the graph visits. At
    /// 2048x2048 level 0 that is nine 512 MiB planes = 4608 MiB for ONE level,
    /// and 8928 MiB over the four, which was the single largest term in the
    /// 13,856 MiB plan. The op list below needs FOUR: `t2` and `t3` (the 2c
    /// expansion) are live together across the depthwise conv, and `t1` is live
    /// with them; everything after SimpleGate works at half the channels and
    /// reuses those. So the workspace is four planes of `2c` plus two [c]
    /// vectors, and the scratch is scoped to the block that is running rather
    /// than to the whole pass.
    ///
    /// EVERY REUSE BELOW IS A REUSE OF A DEAD BUFFER, and that is checkable by
    /// reading the ops in order rather than by trusting the comments:
    ///
    ///   t1 = norm1(inp)              t1 live
    ///   t2 = conv1(t1)               t1 dead, t2 live
    ///   t3 = dwconv(t2)              t2 dead, t3 live
    ///   g  = gate(t3)                t3 dead at the second read, g live
    ///   pooled = mean(g)             needs a [c] vector
    ///   att    = conv1x1(pooled)     `pooled` dead
    ///   t1 = scale(g, att)           g dead, t1 live  (t1's old contents are dead)
    ///   t2 = conv3(t1)               t1 dead, t2 live  (t2's old contents are dead)
    ///   g  = residual(inp, t2, beta) t2 dead, g live   (g's old contents are dead)
    ///   t1 = norm2(g)                g live for the residual at the end
    ///   t2 = conv4(t1)               t1 dead, t2 live
    ///   t3 = gate(t2)                t2 dead, t3 live
    ///   t2 = conv5(t3)               t3 dead, t2 live
    ///   out = residual(g, t2, gamma) both dead afterwards
    ///
    /// FOUR PLANES IS THE MINIMUM FOR THIS OP ORDER: `t2` and `t3` must coexist
    /// across the depthwise conv and `t1` must coexist with `t2` across the 1x1,
    /// which is three planes; the fourth is the input, which is also the
    /// destination. Only `t2` and `t3` need to be `2c` wide - `t1` holds `c`
    /// channels at every step, which is why it is sized at `c`.
    #[allow(clippy::too_many_arguments)]
    fn block(&self, prefix: &str, c: usize, cur: &mut DA, t: &BlockScratch) -> Result<(), String> {
        // A bisect switch: with NAFNET_SKIP_BLOCKS set, the block's launches are
        // skipped entirely and `cur` is left untouched. It separates "a launch
        // inside the block corrupts the context" from "the surrounding pass is
        // wrong", which is the only way to localise a failure that the per-op
        // selftest passes.
        if std::env::var("NAFNET_SKIP_BLOCKS").is_ok() {
            return Ok(());
        }
        let h = cur.h;
        let wd = cur.w;
        let dw = 2 * c;
        let half = c;
        let p = |s: &str| format!("{prefix}.{s}");

        // x = norm1(inp); x = conv1(x); x = dwconv(x)
        self.channel_layer_norm(cur, &p("norm1.weight"), &p("norm1.bias"), &t.t1)?;
        self.conv1x1(&t.t1, &p("conv1.weight"), Some(&p("conv1.bias")), c, dw, &t.t2)?;
        self.conv3x3_dw(&t.t2, &p("conv2.weight"), &p("conv2.bias"), dw, &t.t3)?;

        // SimpleGate: the two halves of the CHANNEL axis, multiplied. `t.t3`'s
        // first `c` channels are the first half by construction, and both halves
        // are read before `t.t3` is written again below.
        let g0 = Self::view(&t.t3.buf, half, h, wd, 0);
        let g1 = Self::view(&t.t3.buf, half, h, wd, half);
        self.mul(&g0, &g1, &t.g)?;

        // sca: global average pool -> 1x1 conv -> per-channel scale of g.
        self.channel_mean(&t.g, &t.pooled)?;
        {
            // The pooled vector is a [half][1][1] activation for the 1x1 conv.
            let pin = DA { holds: false, buf: DevBuf { ptr: t.pooled.ptr, bytes: half * 4 }, c: half, h: 1, w: 1, tag: "pooled view" };
            self.conv1x1(&pin, &p("sca.1.weight"), Some(&p("sca.1.bias")), half, half, &t.att)?;
        }
        // `t.t1`'s norm1 contents are dead here: the depthwise conv above read
        // them into `t.t2`, and nothing since has read `t.t1`.
        self.channel_scale(&t.g, &t.att.buf, &t.t1)?;

        // conv3 back to c, then y = inp + t * beta. `t.t2`'s conv1 contents died
        // at the depthwise conv, so the conv3 output reuses that plane.
        self.conv1x1(&t.t1, &p("conv3.weight"), Some(&p("conv3.bias")), half, c, &t.t2)?;
        // ONE KERNEL, NOT channel_scale + add: see `residual`. `t.t2` is dead
        // after this, and `t.g` holds `y` for the FFN's second residual.
        self.residual(cur, &t.t2, &p("beta"), &t.g)?;

        // FFN: norm2, conv4, SimpleGate, conv5, scaled by gamma. `t.t1` is free
        // again (its scale result was consumed by conv3).
        self.channel_layer_norm(&t.g, &p("norm2.weight"), &p("norm2.bias"), &t.t1)?;
        self.conv1x1(&t.t1, &p("conv4.weight"), Some(&p("conv4.bias")), c, dw, &t.t2)?;
        let f0 = Self::view(&t.t2.buf, half, h, wd, 0);
        let f1 = Self::view(&t.t2.buf, half, h, wd, half);
        // INTO A `c`-WIDE VIEW OF `t3`, AND THE WIDTH IS THE WHOLE POINT. `mul`
        // takes its element count from the OUTPUT, so the output must be exactly
        // `c*h*w`: a `2c`-wide output reads `2c*h*w` elements out of each half of
        // `t2` and runs `c*h*w` elements past its end (the driver answers with
        // CUDA_ERROR_ILLEGAL_ADDRESS at the copy that follows, which is not the
        // launch that caused it - see the check in `mul`). Every plane in this
        // workspace is `2c` wide except `t.g`, which holds `y`, so the gate
        // product is written into the first half of `t3` - untouched since the
        // depthwise conv, and wide enough either way.
        self.mul(&f0, &f1, &Self::view(&t.t3.buf, half, h, wd, 0))?;
        {
            let prod = Self::view(&t.t3.buf, half, h, wd, 0);
            self.conv1x1(&prod, &p("conv5.weight"), Some(&p("conv5.bias")), half, c, &t.t2)?;
        }
        self.residual(&t.g, &t.t2, &p("gamma"), cur)?;
        Ok(())
    }

    // ---- the graph --------------------------------------------------------

    /// The whole network. `input` is [3][h][w] on the host, already padded to a
    /// multiple of `geo.padder_size()`.
    ///
    /// STRUCTURE: one workspace per distinct block shape, not one per block. A
    /// width-32 GoPro model runs 28 of its 36 blocks at the same shape (256
    /// channels at h/8 x w/8), so allocating the scratch per shape is a handful
    /// of allocations instead of one per block. Every buffer is reused across the
    /// forward pass, and the ORDER of the levels is what makes that safe: a
    /// level's scratch is only live while that level runs. The block count is
    /// `sum(enc_blk_nums) + middle_blk_num + sum(dec_blk_nums)` read from the
    /// checkpoint, not a constant.
    /// THE NO-DUMP ENTRY POINT, AND IT IS NOT A DUMP WITH AN EMPTY CLOSURE -
    /// which is exactly what it used to be. `snap` downloaded every intermediate
    /// unconditionally and THEN called the closure, so the fast path did a
    /// synchronising device-to-host copy after each of ~46 stages for data it
    /// threw away: measured at 0.53 s of a 1.95 s pass (27%), outputs
    /// byte-identical with the copy skipped. The distinction is now carried by
    /// the TYPE - `None` here, `Some` in `forward_dump` - so it cannot be lost by
    /// passing a closure that happens to do nothing.
    pub fn forward(&self, input: &[f32], h: usize, wd: usize) -> Result<Vec<f32>, String> {
        let mut host = Vec::new();
        self.forward_inner(input, h, wd, None, &mut host)
    }

    /// The same pass, reporting every intermediate through `dump`.
    ///
    /// THE GPU NEEDS THIS AS MUCH AS THE CPU DOES: this backend has no way to
    /// inspect a device buffer except by downloading it, so without a dump the
    /// only available signal is the final image - and a final image that is
    /// merely "wrong" says nothing about which block produced it.
    ///
    /// DEVELOPMENT ONLY (the CLI's `--dump`), and it stays in the source rather
    /// than behind a module gate because it is the SAME pass - `forward` is this
    /// with `None` - so a build that dropped it would be a different code path
    /// from the one being verified.
    #[cfg(feature = "dev")]
    pub fn forward_dump(
        &self,
        input: &[f32],
        h: usize,
        wd: usize,
        dump: &mut dyn FnMut(&str, usize, usize, usize, &[f32]),
    ) -> Result<Vec<f32>, String> {
        let mut host = Vec::new();
        self.forward_inner(input, h, wd, Some(dump), &mut host)
    }

    /// One forward pass against a pre-built plan.
    ///
    /// EVERY BUFFER IS ALREADY ALLOCATED (see `Plan`): this function only names
    /// slots, uploads the input, launches, and reads the result back. Nothing
    /// here allocates, so nothing here can hand the allocator an address it has
    /// already given to a buffer that is still live.
    fn forward_inner(
        &self,
        input: &[f32],
        h: usize,
        wd: usize,
        mut dump: Option<&mut dyn FnMut(&str, usize, usize, usize, &[f32])>,
        host: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        // THE PASS WALL IS HOST-SIDE, AND IT IS HONEST ONLY BECAUSE THE PASS ENDS
        // WITH A DOWNLOAD OF THE RESULT: that copy synchronises, so by the time
        // this returns every kernel has retired. What the summed event spans do
        // NOT contain is how long the driver spent ACCEPTING the ~600 launches
        // and how long the host spent between them - on a graph this shallow,
        // that gap is a large share of the wall, which is why both are reported.
        let t0 = std::time::Instant::now();
        let geo = &self.geo;
        let levels = geo.levels();
        // THE PLAN IS SIZED BEFORE ANYTHING IS ALLOCATED, which is the whole
        // point: `PlanShape::of` is arithmetic and touches no CUDA state, so the
        // engine can say what a pass will cost - and refuse it - before it has
        // asked the driver for a single byte. `Plan::new` allocates exactly this
        // list, so the number reported and the number allocated cannot drift.
        let shape = PlanShape::of(geo, h, wd)?;
        // AND THE PLAN IS CHECKED AGAINST FREE VRAM BEFORE THE FIRST
        // `cuMemAlloc`. THIS IS THE WHOLE ANSWER TO A PASS THAT DOES NOT FIT:
        // there is no CPU fallback behind it and no retry, because a fallback
        // hides the reason - and the reason is actionable, while a silent switch
        // to a 5 GiB host allocation on a machine with 8 GiB of RAM is not.
        //
        // THE NUMBERS ARE REPORTED IN MIB, NOT GiB, because the decision is made
        // at a boundary a human has to be able to check against `nvidia-smi`.
        let free = lightgpu::vm::free_vram().unwrap_or(usize::MAX);
        // REPORTED EITHER WAY, AND BEFORE THE ALLOCATION: which plan this pass
        // will run and against how much free memory is a property of the result,
        // like the device line - so `--quiet` does not silence it, and a run that
        // then fails for some other reason still says what it was trying to do.
        eprintln!(
            "nafnet: plan {} MiB ({} activations + {} workspace), {} MiB free",
            shape.bytes / 1048576,
            shape.slot_bytes / 1048576,
            shape.workspace_bytes / 1048576,
            free / 1048576,
        );
        let describe = |s: &PlanShape| {
            format!(
                "{} MiB ({} activations + {} workspace)",
                s.bytes / 1048576,
                s.slot_bytes / 1048576,
                s.workspace_bytes / 1048576
            )
        };
        if shape.bytes > free {
            // ONE LINE PER FACT, AND NO INDENTATION: every line carries the
            // engine's own prefix so a caller can strip it and show the rest
            // verbatim. The first line says what happened and to what, the second
            // the arithmetic, the third why it is a limit rather than a guess,
            // and the fourth what the user can actually do about it.
            return Err(format!(
                "not enough device memory for a {}x{} pass\n\
                 nafnet: the plan needs {}; {} MiB is free\n\
                 nafnet: the plan is exact - it is what the driver would be asked for - so this is a hard limit, not a guess\n\
                 nafnet: a smaller image, a narrower checkpoint, or a freer card is what fits",
                wd,
                h,
                describe(&shape),
                free / 1048576,
            ));
        }
        let plan = Plan::new(&self.cuda, &shape).map_err(|e| {
            // THE CHECK ABOVE CAN PASS AND THE ALLOCATION STILL FAIL: free VRAM
            // is a total, and the driver has to find one CONTIGUOUS run for each
            // buffer. So a failure here is reported with the same numbers plus
            // the one fact that distinguishes it - the plan was supposed to fit -
            // rather than as a bare `cuMemAlloc failed`.
            if e.contains("OUT_OF_MEMORY") {
                format!(
                    "the plan for {}x{} ({}) did not fit after all, with {} MiB free\n\
                     nafnet: {e}\n\
                     nafnet: free VRAM is a total, and each buffer needs one contiguous run, so a fragmented card can refuse a plan the total says fits",
                    wd, h, describe(&shape), free / 1048576
                )
            } else {
                e
            }
        })?;

        // The input goes into its own slot in the plan, so it lives as long as
        // the residual add at the end needs it.
        plan.da("in").buf.upload(input)?;
        // THE INTRO CONV WRITES STRAIGHT INTO `enc.0`. It used to write into a
        // slot of its own and then `copy_d2d` into `enc.0` - a second
        // full-resolution activation, plus a copy of it, for a value the copy
        // reproduced byte for byte. The dump still reports it as `intro`, the
        // name `tools/reference.py` uses, so the stage comparison is unchanged.
        {
            let cur = plan.da("in");
            let out = plan.da("enc.0");
            self.conv3x3(cur, "intro.weight", Some("intro.bias"), 3, geo.width, out)?;
        }
        self.snap("intro", plan.da("enc.0"), host, &mut dump)?;

        // The ENCODER. `cur` names the activation in flight, and `enc.{l}` is
        // ALSO the skip the decoder reads back and ALSO the next level's
        // destination - see `PlanShape::of`. Nothing is ever re-pointed and
        // nothing is copied: every transition is a kernel writing one slot
        // while reading another.
        let mut cur = "enc.0".to_string();
        for l in 0..levels {
            let c = geo.width_at(l);
            for b in 0..geo.enc_blk_nums[l] {
                let prefix = format!("encoders.{l}.{b}");
                let hh = plan.da(&cur).h;
                let ww = plan.da(&cur).w;
                let t = plan.scratch(c, hh, ww)?;
                let mut cur_da = plan.da(&cur).clone_meta();
                self.block(&prefix, c, &mut cur_da, t)?;
                self.snap(&prefix, &cur_da, host, &mut dump)?;
            }
            // `cur` STAYS THE SKIP. The decoder reads this slot again and
            // nothing between here and then writes it, so the copy that used to
            // preserve it - and the second full-size allocation it preserved it
            // into - are both gone.
            let next = if l + 1 == levels {
                "bottleneck".to_string()
            } else {
                format!("enc.{}", l + 1)
            };
            {
                let src = plan.da(&cur);
                let dst = plan.da(&next);
                self.downsample(src, l, c, dst)?;
            }
            self.snap(&format!("downs.{l}"), plan.da(&next), host, &mut dump)?;
            cur = next;
        }

        let mid_c = geo.middle_width();
        for b in 0..geo.middle_blk_num {
            let prefix = format!("middle_blks.{b}");
            let hh = plan.da(&cur).h;
            let ww = plan.da(&cur).w;
            let t = plan.scratch(mid_c, hh, ww)?;
            let mut cur_da = plan.da(&cur).clone_meta();
            self.block(&prefix, mid_c, &mut cur_da, t)?;
            self.snap(&prefix, &cur_da, host, &mut dump)?;
        }

        // The DECODER, ON ONE `pre` AND ONE `up` FOR EVERY LEVEL. The shapes
        // come from `pre_shape`/`up_shape` - the same two functions
        // `PlanShape::of` sizes the slots with - so a view here is by
        // construction inside the buffer the plan allocated.
        //
        // `cur_da` NAMES THE ACTIVATION IN FLIGHT AND IS REBOUND, NOT RE-NAMED:
        // the decoder's input is the bottleneck at l = 0 and the previous
        // level's `up` after that, and both are already DAs.
        let mut cur_da = plan.da(&cur).clone_meta();
        for l in 0..levels {
            let c_after = geo.width_at(levels - 1 - l);
            let (pc, ph, pw) = pre_shape(geo, h, wd, l);
            let pre = plan.da("pre").resized(pc, ph, pw);
            let (uc, uh, uw) = up_shape(geo, h, wd, l);
            let up = plan.da("up").resized(uc, uh, uw);
            // 1x1 conv to 4x the channels (bias-free), sized 2*c_after, at the
            // resolution the shuffle will double.
            {
                let c_before = cur_da.c;
                self.conv1x1(&cur_da, &format!("ups.{l}.0.weight"), None, c_before, c_before * 2, &pre)?;
            }
            self.pixel_shuffle2(&pre, c_after, &up)?;
            // `ups.{l}` IS THE SHUFFLE OUTPUT, and the dump has to happen HERE:
            // before the skip add mutates this slot in place, and before the
            // decoder blocks rewrite it again. That is the stage
            // `tools/reference.py` dumps as `ups.{i}` (`x = up(x); dump(...)`
            // then `x = x + skips[-i-1]`), and snapping later compares a
            // post-decoder tensor against a pre-skip-add one - which reads as a
            // difference larger than either tensor's own peak.
            self.snap(&format!("ups.{l}"), &up, host, &mut dump)?;
            // The skip IS the encoder's slot for the mirrored level - see
            // `PlanShape::of` for why there is no separate `skip.{l}` slot.
            {
                let skip = plan.da(&format!("enc.{}", levels - 1 - l));
                self.add(&up, skip, &up)?;
            }
            // The decoder blocks run in place in the `up` slot at THIS level's
            // shape, so they get their own scratch keyed by that shape. A
            // lookup, to fail here rather than inside the block loop if the plan
            // never enumerated it.
            plan.scratch(c_after, uh, uw)?;
            for b in 0..geo.dec_blk_nums[l] {
                let prefix = format!("decoders.{l}.{b}");
                let t = plan.scratch(c_after, uh, uw)?;
                let mut blk_da = up.clone_meta();
                self.block(&prefix, c_after, &mut blk_da, t)?;
                self.snap(&prefix, &blk_da, host, &mut dump)?;
            }
            cur_da = up;
        }

        {
            let out = plan.da("ending");
            self.conv3x3(&cur_da, "ending.weight", Some("ending.bias"), geo.width, 3, out)?;
            let din = plan.da("in");
            self.add(out, din, out)?;
        }
        let out = plan.da("ending");
        // THE FINAL DOWNLOAD IS NOT OPTIONAL - it is how the caller gets a result,
        // and it is the sync that makes the pass wall meaningful.
        host.resize(out.n(), 0.0);
        out.buf.download(host)?;
        if let Some(d) = dump.as_mut() {
            d("out", out.c, out.h, out.w, host);
        }
        if let Some(p) = &self.profile {
            p.wall.set(t0.elapsed().as_secs_f32() * 1000.0);
        }
        Ok(host.clone())
    }

    /// Download one activation and hand it to the dump closure.
    ///
    /// The stage name is in EVERY error: a device-side failure during a dump
    /// says which buffer was bad, and `DevBuf::download` sizes the copy from the
    /// SLICE, so a host vector that is too short would silently copy too little
    /// rather than fail. Checking the length against the buffer here is what
    /// keeps a dump honest.
    fn snap(
        &self,
        name: &str,
        a: &DA,
        host: &mut Vec<f32>,
        dump: &mut Option<&mut dyn FnMut(&str, usize, usize, usize, &[f32])>,
    ) -> Result<(), String> {
        // NO DUMP, NO DOWNLOAD. The copy below is a SYNCHRONISING device-to-host
        // transfer, and on a pass that is not dumping there is nothing to hand it
        // to. `forward()` used to reach here with a no-op closure, so the fast
        // path paid one DtoH per stage - ~46 of them, up to 108 MB each - for
        // data it discarded, and the cost was not only the copy: once the host
        // blocks on a transfer, the driver can no longer overlap the next
        // kernel's launch with the previous one's tail.
        let Some(dump) = dump.as_mut() else { return Ok(()); };
        host.resize(a.n(), 0.0);
        if host.len() * 4 != a.buf.bytes {
            return Err(format!(
                "{name}: host {} floats but the device buffer is {} bytes",
                host.len(),
                a.buf.bytes
            ));
        }
        a.buf.download(host).map_err(|e| format!("{name}: {e}"))?;
        dump(name, a.c, a.h, a.w, host);
        Ok(())
    }

    // ---- the selftest -----------------------------------------------------

    /// Compare every kernel this engine launches against its CPU twin on random
    /// data, op by op, and report the worst deviation per op.
    ///
    /// WHY THIS EXISTS: a wrong grid, a wrong argument width or a kernel reading
    /// the wrong axis all produce numbers rather than errors, and the only way
    /// to catch a host/device disagreement is to compute the same thing twice by
    /// different means. This is the same contract `realesrgan --cuda-selftest`
    /// and `rmbg --cuda-selftest` implement.
    /// DEVELOPMENT ONLY: `--cuda-selftest`, run by a developer rather than by a
    /// user restoring an image.
    #[cfg(feature = "dev")]
    pub fn selftest(&self) -> Result<(usize, Vec<String>), String> {
        let mut rows = Vec::new();
        let mut fails = 0usize;
        let rng = |n: usize, seed: u32| -> Vec<f32> {
            let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((s >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let cmp = |name: &str, gpu: &[f32], cpu: &[f32], rows: &mut Vec<String>, fails: &mut usize| {
            let mut worst = 0.0f32;
            for (a, b) in gpu.iter().zip(cpu) {
                worst = worst.max((a - b).abs());
            }
            let ok = worst < 1e-4;
            if !ok {
                *fails += 1;
            }
            rows.push(format!(
                "{:24} {:>10} {:>12.3e}",
                name,
                if ok { "ok" } else { "MISMATCH" },
                worst
            ));
        };
        // A RELATIVE check, because `cmp`'s absolute threshold only means
        // something on the O(1) data above. The real graph runs these
        // reductions at `hw` in the tens of thousands with activations at
        // magnitude ~1e4, where a good reduction still lands at 1e-3 absolute -
        // and where the ORDER of the sum is the thing under test. Judging those
        // with `worst < 1e-4` marks every correct implementation a failure, and
        // judging the small cases above by relative error marks none of them
        // anything, which is why both are here.
        let cmp_rel =
            |name: &str, gpu: &[f32], cpu: &[f32], rows: &mut Vec<String>, fails: &mut usize| {
                let mut worst = 0.0f32;
                for (a, b) in gpu.iter().zip(cpu) {
                    worst = worst.max((a - b).abs());
                }
                let peak = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
                let rel = worst / peak;
                let ok = rel < 1e-5;
                if !ok {
                    *fails += 1;
                }
                rows.push(format!(
                    "{:24} {:>10} {:>12.3e}",
                    name,
                    if ok { "ok" } else { "MISMATCH" },
                    rel
                ));
            };

        // 1x1 conv
        {
            let (c_in, c_out, h, w) = (5usize, 7usize, 9usize, 11usize);
            let x = rng(c_in * h * w, 1);
            let wt = rng(c_out * c_in, 2);
            let bs = rng(c_out, 3);
            let din = DA { holds: true, buf: self.cuda.upload(&x)?, c: c_in, h, w, tag: "selftest din c_in" };
            let dout = DA::new(c_out, h, w)?;
            let mut a = Args::new();
            a.ptr(din.buf.ptr)
                .ptr(self.cuda.upload(&wt)?.ptr)
                .ptr(self.cuda.upload(&bs)?.ptr)
                .ptr(dout.buf.ptr)
                .i32(c_in as i32)
                .i32(c_out as i32)
                .i32(h as i32)
                .i32(w as i32);
            self.go("lg_conv1x1", grid_for(c_out * h * w, BLOCK), (BLOCK as u32, 1, 1), &mut a)?;
            let mut got = vec![0.0f32; c_out * h * w];
            dout.buf.download(&mut got)?;
            let mut want = vec![0.0f32; c_out * h * w];
            crate::net::conv1x1(&x, &wt, &bs, c_in, c_out, h * w, &mut want);
            cmp("lg_conv1x1", &got, &want, &mut rows, &mut fails);
        }

        // THE ENGINE'S OWN 1x1 CONV, `nf_conv1x1_oc`, against BOTH references:
        // the CPU twin (does it compute the right thing?) and the toolkit kernel
        // it replaces (is it the SAME? - the forward pass now launches this one,
        // so any disagreement here is a changed image). `c_out = 7` is not a
        // multiple of `OC_TILE`, which is the case the kernel's clamped READ path
        // and guarded STORE exist for; `c_out = 8` exercises whole tiles; the
        // 64x64x64 case runs it at the shape the graph uses, where the
        // accumulation error is visible at all.
        for (c_in, c_out, h, w, tag, big) in [
            (5usize, 7usize, 9usize, 11usize, "c_out%4!=0", false),
            (5, 8, 9, 11, "c_out%4==0", false),
            (64, 64, 64, 64, "64x64x64", true),
        ] {
            let x = rng(c_in * h * w, 30);
            let wt = rng(c_out * c_in, 31);
            let bs = rng(c_out, 32);
            let din = DA { holds: true, buf: self.cuda.upload(&x)?, c: c_in, h, w, tag: "selftest tiled din" };
            let dwt = self.cuda.upload(&wt)?;
            let dbs = self.cuda.upload(&bs)?;
            let dout = DA::new(c_out, h, w)?;
            let mut a = Args::new();
            a.ptr(din.buf.ptr)
                .ptr(dwt.ptr)
                .ptr(dbs.ptr)
                .ptr(dout.buf.ptr)
                .i32(c_in as i32)
                .i32(c_out as i32)
                .i32(h as i32)
                .i32(w as i32);
            let grid = (grid_for(h * w, BLOCK).0, c_out.div_ceil(OC_TILE) as u32, 1);
            self.go("nf_conv1x1_oc", grid, (BLOCK as u32, 1, 1), &mut a)?;
            let mut got = vec![0.0f32; c_out * h * w];
            dout.buf.download(&mut got)?;

            let mut want = vec![0.0f32; c_out * h * w];
            crate::net::conv1x1(&x, &wt, &bs, c_in, c_out, h * w, &mut want);
            let name = format!("nf_conv1x1_oc {tag}");
            if big {
                cmp_rel(&name, &got, &want, &mut rows, &mut fails);
            } else {
                cmp(&name, &got, &want, &mut rows, &mut fails);
            }

            // And against the toolkit kernel over the identical inputs.
            let dref = DA::new(c_out, h, w)?;
            let mut a = Args::new();
            a.ptr(din.buf.ptr)
                .ptr(dwt.ptr)
                .ptr(dbs.ptr)
                .ptr(dref.buf.ptr)
                .i32(c_in as i32)
                .i32(c_out as i32)
                .i32(h as i32)
                .i32(w as i32);
            self.go("lg_conv1x1", grid_for(c_out * h * w, BLOCK), (BLOCK as u32, 1, 1), &mut a)?;
            let mut tk = vec![0.0f32; c_out * h * w];
            dref.buf.download(&mut tk)?;
            cmp(&format!("   vs lg_conv1x1 {tag}"), &got, &tk, &mut rows, &mut fails);
        }

        // 3x3 conv, and 3x3 depthwise
        {
            let (c_in, c_out, h, w) = (4usize, 6usize, 9usize, 10usize);
            let x = rng(c_in * h * w, 4);
            let wt = rng(c_out * c_in * 9, 5);
            let bs = rng(c_out, 6);
            // Pad the input so both implementations see a border.
            let din = DA { holds: true, buf: self.cuda.upload(&x)?, c: c_in, h, w, tag: "selftest din c_in" };
            let dout = DA::new(c_out, h, w)?;
            let mut a = Args::new();
            a.ptr(din.buf.ptr)
                .ptr(self.cuda.upload(&wt)?.ptr)
                .ptr(self.cuda.upload(&bs)?.ptr)
                .ptr(dout.buf.ptr)
                .i32(c_in as i32)
                .i32(c_out as i32)
                .i32(h as i32)
                .i32(w as i32);
            self.go("lg_conv3x3s1p1", grid_for(c_out * h * w, BLOCK), (BLOCK as u32, 1, 1), &mut a)?;
            let mut got = vec![0.0f32; c_out * h * w];
            dout.buf.download(&mut got)?;
            let mut want = vec![0.0f32; c_out * h * w];
            crate::net::conv3x3(&x, &wt, &bs, c_in, c_out, h, w, &mut want);
            cmp("lg_conv3x3s1p1", &got, &want, &mut rows, &mut fails);

            // depthwise: the weight is [c_out][1][3][3] and c_in == c_out.
            let dw = c_out;
            let x = rng(dw * h * w, 7);
            let wt = rng(dw * 9, 8);
            let bs = rng(dw, 9);
            let din = DA { holds: true, buf: self.cuda.upload(&x)?, c: dw, h, w, tag: "selftest din dw" };
            let dout = DA::new(dw, h, w)?;
            let dwt = self.cuda.upload(&wt)?;
            let dbs = self.cuda.upload(&bs)?;
            let mut a = Args::new();
            a.ptr(din.buf.ptr)
                .ptr(dwt.ptr)
                .ptr(dbs.ptr)
                .ptr(dout.buf.ptr)
                .i32(dw as i32)
                .i32(h as i32)
                .i32(w as i32);
            let grid = (
                w.div_ceil(TX) as u32,
                h.div_ceil(TY) as u32,
                dw as u32,
            );
            self.go("nf_conv3x3_dw", grid, (TX as u32, TY as u32, 1), &mut a)?;
            let mut got = vec![0.0f32; dw * h * w];
            dout.buf.download(&mut got)?;
            let mut want = vec![0.0f32; dw * h * w];
            crate::net::conv3x3_dw(&x, &wt, &bs, dw, h, w, &mut want);
            cmp("nf_conv3x3_dw", &got, &want, &mut rows, &mut fails);
        }

        // LayerNorm over the channel axis, channel mean, mul, channel scale, add
        {
            let (c, h, w) = (6usize, 4usize, 5usize);
            let hw = h * w;
            let x = rng(c * hw, 10);
            let wt = rng(c, 11);
            let bs = rng(c, 12);
            let din = DA { holds: true, buf: self.cuda.upload(&x)?, c, h, w, tag: "selftest din c" };
            let dout = DA::new(c, h, w)?;
            let mut a = Args::new();
            a.ptr(din.buf.ptr)
                .ptr(self.cuda.upload(&wt)?.ptr)
                .ptr(self.cuda.upload(&bs)?.ptr)
                .ptr(dout.buf.ptr)
                .i32(c as i32)
                .i32(hw as i32)
                .f32(crate::net::LAYERNORM_EPS);
            // `hw / BLOCK`, matching `Gpu::channel_layer_norm` - see the note
            // there for why a grid of `hw` would silently under-cover.
            self.go(
                "lg_channel_layer_norm",
                ((hw as u32).div_ceil(BLOCK as u32), 1, 1),
                (BLOCK as u32, 1, 1),
                &mut a,
            )?;
            let mut got = vec![0.0f32; c * hw];
            dout.buf.download(&mut got)?;
            let mut want = vec![0.0f32; c * hw];
            crate::net::channel_layer_norm(&x, &wt, &bs, c, hw, &mut want);
            cmp("lg_channel_layer_norm", &got, &want, &mut rows, &mut fails);

            let dmean = self.cuda.buf(c)?;
            self.channel_mean(&din, &dmean)?;
            let mut got = vec![0.0f32; c];
            dmean.download(&mut got)?;
            let mut want = vec![0.0f32; c];
            crate::net::channel_mean(&x, c, hw, &mut want);
            cmp("lg_channel_mean", &got, &want, &mut rows, &mut fails);

            let b2 = rng(c * hw, 13);
            let db = DA { holds: true, buf: self.cuda.upload(&b2)?, c, h, w, tag: "selftest db" };
            let dmul = DA::new(c, h, w)?;
            self.mul(&din, &db, &dmul)?;
            let mut got = vec![0.0f32; c * hw];
            dmul.buf.download(&mut got)?;
            let mut want = vec![0.0f32; c * hw];
            crate::net::mul(&x, &b2, &mut want);
            cmp("lg_mul", &got, &want, &mut rows, &mut fails);

            let ds = self.cuda.upload(&bs)?;
            let dsc = DA::new(c, h, w)?;
            self.channel_scale(&din, &ds, &dsc)?;
            let mut got = vec![0.0f32; c * hw];
            dsc.buf.download(&mut got)?;
            let mut want = vec![0.0f32; c * hw];
            crate::net::channel_scale(&x, &bs, c, hw, &mut want);
            cmp("lg_channel_scale", &got, &want, &mut rows, &mut fails);

            let dadd = DA::new(c, h, w)?;
            self.add(&din, &db, &dadd)?;
            let mut got = vec![0.0f32; c * hw];
            dadd.buf.download(&mut got)?;
            let mut want = vec![0.0f32; c * hw];
            crate::net::add(&x, &b2, &mut want);
            cmp("lg_add", &got, &want, &mut rows, &mut fails);
        }

        // The same two reductions AT THE SIZE AND SCALE THE GRAPH ACTUALLY
        // RUNS THEM. The `hw = 20` case above cannot tell two summation orders
        // apart - at 20 elements they agree bit for bit, which is why
        // `lg_channel_mean` prints an exact zero up there and still drifts from
        // the CPU twin by 1e-2 through a 36-block forward pass. This is the
        // case that sees it: 16384 elements per channel, values around 500 with
        // a spread of 1000, so `E[x^2] - mean^2` cancels the way it does inside
        // a real block.
        {
            let (c, h, w) = (64usize, 128usize, 128usize);
            let hw = h * w;
            let base = rng(c * hw, 21);
            let x: Vec<f32> = base.iter().map(|v| v * 1000.0 + 500.0).collect();
            let din = DA { holds: true, buf: self.cuda.upload(&x)?, c, h, w, tag: "selftest din big" };

            let dmean = self.cuda.buf(c)?;
            self.channel_mean(&din, &dmean)?;
            let mut got = vec![0.0f32; c];
            dmean.download(&mut got)?;
            let mut want = vec![0.0f32; c];
            crate::net::channel_mean(&x, c, hw, &mut want);
            cmp_rel("lg_channel_mean@16384", &got, &want, &mut rows, &mut fails);

            let wt = rng(c, 22);
            let bs = rng(c, 23);
            let dout = DA::new(c, h, w)?;
            let mut a = Args::new();
            a.ptr(din.buf.ptr)
                .ptr(self.cuda.upload(&wt)?.ptr)
                .ptr(self.cuda.upload(&bs)?.ptr)
                .ptr(dout.buf.ptr)
                .i32(c as i32)
                .i32(hw as i32)
                .f32(crate::net::LAYERNORM_EPS);
            self.go(
                "lg_channel_layer_norm",
                ((hw as u32).div_ceil(BLOCK as u32), 1, 1),
                (BLOCK as u32, 1, 1),
                &mut a,
            )?;
            let mut got = vec![0.0f32; c * hw];
            dout.buf.download(&mut got)?;
            let mut want = vec![0.0f32; c * hw];
            crate::net::channel_layer_norm(&x, &wt, &bs, c, hw, &mut want);
            cmp_rel("lg_channel_layer_norm@16384", &got, &want, &mut rows, &mut fails);
        }

        // AND THE CONVOLUTIONS AT THEIR REAL SIZE. These are the deepest inner
        // loops in the graph - every block runs `lg_conv3x3s1p1` over `2c`
        // channels - so their accumulation order is the remaining candidate for
        // the CPU/GPU drift that the reductions above turned out not to explain.
        {
            let (ci, co, h, w) = (64usize, 64usize, 64usize, 64usize);
            let xb = rng(ci * h * w, 24);
            let x: Vec<f32> = xb.iter().map(|v| v * 1000.0 + 500.0).collect();
            // Small weights, so the 576-term sums produce O(10-100) outputs
            // rather than 1e6 - the regime the real block runs in.
            let wtb = rng(co * ci * 9, 25);
            let wt: Vec<f32> = wtb.iter().map(|v| v * 1e-2).collect();
            let bs: Vec<f32> = rng(co, 26).iter().map(|v| v * 1e-2).collect();
            let din = DA { holds: true, buf: self.cuda.upload(&x)?, c: ci, h, w, tag: "selftest din big" };
            let dout = DA::new(co, h, w)?;
            let mut a = Args::new();
            a.ptr(din.buf.ptr)
                .ptr(self.cuda.upload(&wt)?.ptr)
                .ptr(self.cuda.upload(&bs)?.ptr)
                .ptr(dout.buf.ptr)
                .i32(ci as i32)
                .i32(co as i32)
                .i32(h as i32)
                .i32(w as i32);
            self.go("lg_conv3x3s1p1", grid_for(co * h * w, BLOCK), (BLOCK as u32, 1, 1), &mut a)?;
            let mut got = vec![0.0f32; co * h * w];
            dout.buf.download(&mut got)?;
            let mut want = vec![0.0f32; co * h * w];
            crate::net::conv3x3(&x, &wt, &bs, ci, co, h, w, &mut want);
            cmp_rel("lg_conv3x3s1p1@64x64x64", &got, &want, &mut rows, &mut fails);

            // the depthwise 3x3, one channel per group, 9 terms per output
            let dwb = rng(co * h * w, 27);
            let dwx: Vec<f32> = dwb.iter().map(|v| v * 1000.0 + 500.0).collect();
            let dwtb = rng(co * 9, 28);
            let dwt: Vec<f32> = dwtb.iter().map(|v| v * 1e-2).collect();
            let ddw = DA { holds: true, buf: self.cuda.upload(&dwx)?, c: co, h, w, tag: "selftest ddw big" };
            let ddwout = DA::new(co, h, w)?;
            let mut a = Args::new();
            a.ptr(ddw.buf.ptr)
                .ptr(self.cuda.upload(&dwt)?.ptr)
                .ptr(self.cuda.upload(&bs)?.ptr)
                .ptr(ddwout.buf.ptr)
                .i32(co as i32)
                .i32(h as i32)
                .i32(w as i32);
            let grid = (w.div_ceil(TX) as u32, h.div_ceil(TY) as u32, co as u32);
            self.go("nf_conv3x3_dw", grid, (TX as u32, TY as u32, 1), &mut a)?;
            let mut got = vec![0.0f32; co * h * w];
            ddwout.buf.download(&mut got)?;
            let mut want = vec![0.0f32; co * h * w];
            crate::net::conv3x3_dw(&dwx, &dwt, &bs, co, h, w, &mut want);
            cmp_rel("nf_conv3x3_dw@64x64x64", &got, &want, &mut rows, &mut fails);
        }

        // pixel_shuffle2 and the stride-2 2x2 downsample
        {
            let (c, h, w) = (3usize, 4usize, 5usize);
            let x = rng(c * 4 * h * w, 14);
            let din = DA { holds: true, buf: self.cuda.upload(&x)?, c: c * 4, h, w, tag: "selftest din c*4" };
            let dout = DA::new(c, h * 2, w * 2)?;
            self.pixel_shuffle2(&din, c, &dout)?;
            let mut got = vec![0.0f32; c * 4 * h * w];
            dout.buf.download(&mut got)?;
            let mut want = vec![0.0f32; c * 4 * h * w];
            crate::net::pixel_shuffle2(&x, c, h, w, &mut want);
            cmp("nf_pixel_shuffle2", &got, &want, &mut rows, &mut fails);

            let (ci, co, h, w) = (3usize, 5usize, 8usize, 6usize);
            let x = rng(ci * h * w, 15);
            let wt = rng(co * ci * 4, 16);
            let bs = rng(co, 17);
            let din = DA { holds: true, buf: self.cuda.upload(&x)?, c: ci, h, w, tag: "selftest din ci" };
            let dout = DA::new(co, h / 2, w / 2)?;
            let mut a = Args::new();
            a.ptr(din.buf.ptr)
                .ptr(self.cuda.upload(&wt)?.ptr)
                .ptr(self.cuda.upload(&bs)?.ptr)
                .ptr(dout.buf.ptr)
                .i32(ci as i32)
                .i32(co as i32)
                .i32(h as i32)
                .i32(w as i32);
            // The real geometry: positions on x, channel tiles on y.
            let grid = (
                grid_for((h / 2) * (w / 2), BLOCK).0,
                co.div_ceil(DW_OC_TILE) as u32,
                1,
            );
            self.go("nf_down2x2s2", grid, (BLOCK as u32, 1, 1), &mut a)?;
            let mut got = vec![0.0f32; co * (h / 2) * (w / 2)];
            dout.buf.download(&mut got)?;
            let mut want = vec![0.0f32; co * (h / 2) * (w / 2)];
            crate::net::conv2x2s2(&x, &wt, &bs, ci, co, h, w, &mut want);
            cmp("nf_down2x2s2", &got, &want, &mut rows, &mut fails);
        }

        Ok((fails, rows))
    }
}

/// Every device buffer one forward pass will use, as SHAPES rather than
/// allocations.
///
/// THIS IS THE ARITHMETIC AND NOTHING ELSE - no CUDA call, no `cuMemAlloc` - so
/// it can be run before there is anything to allocate. That is what lets the
/// engine answer "will this fit on this card?" and say so, instead of dying
/// inside a launch with `CUDA_ERROR_OUT_OF_MEMORY` half a second in. `Plan::new`
/// allocates from exactly this list, so the number the engine reports and the
/// bytes it then asks the driver for cannot drift apart.
///
/// It is also where the plan's two invariants are checked, because both are
/// properties of the LIST and not of the driver: names must be unique (a
/// duplicate would mean two slots for one buffer) and a shape may not be
/// enumerated twice (two workspaces for one shape).
pub struct PlanShape {
    /// `(name, c, h, w)`, in the order they are enumerated.
    slots: Vec<(String, usize, usize, usize)>,
    /// One workspace per distinct block shape, as `(c, h, w)`. THE POOL IS THE
    /// LARGEST OF THESE, not their sum - see `Plan::new`.
    scratch: Vec<(usize, usize, usize)>,
    /// Bytes the pass holds at once: every slot, plus ONE workspace pool.
    pub bytes: usize,
    /// The activation slots alone. Kept separate because the two halves fail
    /// differently - activations scale with the image and the level count, the
    /// workspace with one block - so a message that reports only the total
    /// cannot say which of them to attack.
    pub slot_bytes: usize,
    /// The workspace pool alone.
    pub workspace_bytes: usize,
}

impl PlanShape {
    pub fn of(geo: &Geometry, h: usize, wd: usize) -> Result<PlanShape, String> {
        // A DUPLICATE NAME WOULD FREE THE FIRST ALLOCATION, and that is why this
        // is an error rather than an overwrite: two slots for one buffer means
        // the executor can read a name it has already written. Names must be
        // unique.
        fn put(
            slots: &mut Vec<(String, usize, usize, usize)>,
            n: &str,
            c: usize,
            hh: usize,
            ww: usize,
        ) -> Result<(), String> {
            if slots.iter().any(|(m, ..)| m == n) {
                return Err(format!("plan slot `{n}` was enumerated twice"));
            }
            slots.push((n.to_string(), c, hh, ww));
            Ok(())
        }
        // ONE workspace per distinct block shape. A second entry for a shape
        // would be a second allocation for the same work, which is the whole
        // memory cost this plan exists to bound.
        fn workspace_for(
            scratch: &mut Vec<(usize, usize, usize)>,
            c: usize,
            hh: usize,
            ww: usize,
        ) {
            if !scratch.iter().any(|k| *k == (c, hh, ww)) {
                scratch.push((c, hh, ww));
            }
        }

        let levels = geo.levels();
        let mut slots: Vec<(String, usize, usize, usize)> = Vec::new();
        let mut scratch: Vec<(usize, usize, usize)> = Vec::new();

        // ONE SLOT PER LEVEL, CARRYING THREE ROLES: the encoder's activation, the
        // skip the decoder reads back, and the downsample's destination. The
        // chain is `enc.{l}` written by level l's blocks (in place), read by
        // level l's downsample, and read again by the decoder - and NOTHING
        // writes it in between, so one buffer holds all three. Naming them
        // separately cost a `copy_d2d` per level plus a second full-size
        // allocation: at 2048x2048 those copies were 1920 MiB of a 13.5 GiB plan,
        // and 2432 MiB counting the input and the intro slot the same way.
        let (mut ch, mut cw) = (h, wd);
        for l in 0..levels {
            let c = geo.width_at(l);
            workspace_for(&mut scratch, c, ch, cw);
            put(&mut slots, &format!("enc.{l}"), c, ch, cw)?;
            let (mc, mh, mw) = up_shape(geo, h, wd, l);
            workspace_for(&mut scratch, mc, mh, mw);
            ch /= 2;
            cw /= 2;
        }
        // THE BOTTLENECK IS ITS OWN SLOT: the middle blocks run in place on the
        // last downsample's output, at `middle_width()` channels on the smallest
        // plane, which is a shape no encoder level has.
        put(&mut slots, "bottleneck", geo.middle_width(), ch, cw)?;
        workspace_for(&mut scratch, geo.middle_width(), ch, cw);
        put(&mut slots, "in", 3, h, wd)?;
        put(&mut slots, "ending", 3, h, wd)?;
        // ONE `pre` AND ONE `up`, BOTH SIZED TO THE LARGEST DECODER LEVEL.
        //
        // They were per level, which was 2432 MiB of the plan at 2048x2048 for
        // the two of them. What makes one pair enough is that the decoder's
        // stages are STRICTLY ORDERED: level l's shuffle finishes reading `pre`
        // before level l+1's 1x1 writes it, and level l's blocks finish with
        // `up` before level l+1's shuffle writes it. So the largest shape's
        // buffer serves every level, and a smaller level simply uses the front
        // of it.
        //
        // THE VIEWS ARE BUILT FROM THE PLAN'S OWN SHAPES, and not from the
        // running counters, which is what makes a shared slot safe: every level
        // writes the same bytes of the same allocation that the level before it
        // wrote, so there is no address for the driver to recycle and no
        // lifetime for the borrow checker to get wrong.
        let (pc, ph, pw) = (0..levels)
            .map(|l| pre_shape(geo, h, wd, l))
            .max_by_key(|(c, hh, ww)| c * hh * ww)
            .expect("levels is never zero");
        put(&mut slots, "pre", pc, ph, pw)?;
        let (uc, uh, uw) = (0..levels)
            .map(|l| up_shape(geo, h, wd, l))
            .max_by_key(|(c, hh, ww)| c * hh * ww)
            .expect("levels is never zero");
        put(&mut slots, "up", uc, uh, uw)?;
        // The executor's slot set is exactly: in, enc.{l}, bottleneck, pre, up,
        // ending - nothing else is allocated during the pass, and `in` is the
        // only slot the whole pass holds for its own sake.

        let mut slot_bytes = 0usize;
        for (_, c, hh, ww) in &slots {
            slot_bytes += c * hh * ww * 4;
        }
        // The WORKSPACE IS COUNTED ONCE, because one pool serves every shape -
        // see `Plan::new`. Counting one per shape is what the engine did before
        // the pool existed, and it is what made 2048x2048/w32 need 9952 MiB when
        // the same plan in one pool needs 6600.
        let workspace_bytes = scratch
            .iter()
            .map(|(c, hh, ww)| workspace_floats(*c, *hh, *ww) * 4)
            .max()
            .unwrap_or(0);
        Ok(PlanShape {
            slots,
            scratch,
            bytes: slot_bytes + workspace_bytes,
            slot_bytes,
            workspace_bytes,
        })
    }
}

/// Every device buffer one forward pass will use, allocated up front.
///
/// THE BUG THIS EXISTS TO KILL: with activations allocated per stage and block
/// scratch kept in a map keyed by shape, a scratch buffer allocated during the
/// encoder can be handed, by `cuMemAlloc`, the address of a decoder activation
/// that is still live - because `DevBuf` frees on drop and the driver recycles
/// addresses. The two then overlap silently: this engine wrote a depthwise
/// result over its own input and produced a wrong image, and later the same
/// overlap made a `cuMemcpyDtoH` fail outright. A pointer trace at the failing
/// block is what identified it (`cur` and `t3` at the same address).
///
/// The geometry is known before the first launch, so the fix is a static plan:
/// name every buffer the pass needs, allocate them together, and let nothing be
/// dropped until the pass ends. `slots` holds the named activations, `scratch`
/// one workspace per distinct block shape. Both are keyed by name/shape, and a
/// shape is never allocated twice.
struct Plan {
    /// THE WORKSPACE POOL, AND IT IS WHY EVERY `BlockScratch` IS A SET OF VIEWS:
    /// one allocation, sized to the largest block shape, handed to each shape at
    /// an offset. It is held only so that it outlives the views into it - the
    /// `Drop` order inside `Plan` would otherwise be able to free it first.
    _pool: DevBuf,
    slots: std::collections::HashMap<String, DA>,
    scratch: std::collections::HashMap<(usize, usize, usize), BlockScratch>,
}

impl Plan {
    /// Allocate every buffer `shape` names. The shapes - and therefore the byte
    /// total the engine reports before it launches anything - come from
    /// `PlanShape::of`, which is pure; this function only turns them into
    /// allocations.
    fn new(cuda: &Cuda, shape: &PlanShape) -> Result<Plan, String> {
        // ONE POOL FOR EVERY WORKSPACE, SIZED TO THE LARGEST. The workspaces do
        // not overlap in time - level l's blocks run before level l+1's, and the
        // decoder's blocks reuse the encoder's shapes - so a single allocation
        // serves all of them and the plan pays for one, not for the sum. That is
        // the difference between 6944 MiB of workspace and 3584 MiB at
        // 2048x2048/w32, and it is why `PlanShape::bytes` counts the maximum
        // rather than the total.
        let pool_floats = shape
            .scratch
            .iter()
            .map(|(c, hh, ww)| workspace_floats(*c, *hh, *ww))
            .max()
            .unwrap_or(0);
        let pool = cuda.buf(pool_floats)?;
        let mut scratch = std::collections::HashMap::new();
        for (c, hh, ww) in &shape.scratch {
            scratch.insert((*c, *hh, *ww), BlockScratch::new(pool.ptr, *c, *hh, *ww));
        }
        let mut slots = std::collections::HashMap::new();
        for (n, c, hh, ww) in &shape.slots {
            slots.insert(n.clone(), DA::new(*c, *hh, *ww)?);
        }
        // A slot whose `bytes` does not match its (c,h,w) is the failure mode
        // that made this plan necessary: a launch would then read or write past
        // the allocation, or a `copy_d2d` would name a range that is not there
        // and fail with CUDA_ERROR_INVALID_VALUE. Check it here, once, before
        // any kernel runs.
        for (name, d) in &slots {
            if d.buf.bytes != d.c * d.h * d.w * 4 {
                return Err(format!(
                    "plan slot `{name}`: ({}x{}x{}) needs {} bytes but has {}",
                    d.c, d.h, d.w, d.c * d.h * d.w * 4, d.buf.bytes
                ));
            }
        }
        if std::env::var("NAFNET_DEBUG_PLAN").is_ok() {
            let mut names: Vec<&String> = slots.keys().collect();
            names.sort();
            for n in names {
                let d = &slots[n];
                eprintln!("{n:10} c={} h={} w={} bytes={}", d.c, d.h, d.w, d.buf.bytes);
            }
        }
        Ok(Plan { _pool: pool, slots, scratch })
    }

    fn da(&self, name: &str) -> &DA {
        self.slots
            .get(name)
            .unwrap_or_else(|| panic!("plan has no slot `{name}`"))
    }

    fn scratch(&self, c: usize, h: usize, w: usize) -> Result<&BlockScratch, String> {
        self.scratch.get(&(c, h, w)).ok_or_else(|| {
            format!("the plan has no scratch for c={c} h={h} w={w} - the shapes it enumerated are wrong")
        })
    }

}

/// The workspace ONE NAFBlock needs, at one shape. One set per distinct
/// (c, h, w) in the plan, shared by every block that runs at that shape.
///
/// THREE PLANES AND TWO VECTORS, WHICH IS THE LIVE SET AND NOT THE ROLE LIST -
/// see `Gpu::block`, whose op order is the proof that each reuse is a reuse of a
/// dead buffer. `t2` and `t3` are `2c`-wide so that one plane serves the
/// width-`2c` roles and the width-`c` roles that follow them; `g` holds the first
/// residual while the FFN runs; and `t1` is the FIRST HALF OF `t3`.
struct BlockScratch {
    /// norm1 output, then the sca-scaled gate, then norm2 output. A VIEW of
    /// `t3`'s first half, never a plane of its own - see `Gpu::block` for the
    /// three places `t3` is dead while `t1` is live.
    t1: DA,
    /// conv1 output, then conv3 output, then conv4 output, then conv5 output.
    t2: DA,
    /// dwconv output, then the FFN's gate product.
    t3: DA,
    /// the SimpleGate product, which becomes `y` at the first residual.
    g: DA,
    /// the global average pool, a `[c]` vector.
    pooled: DevBuf,
    /// the sca 1x1 conv's output, a `[c][1][1]` activation.
    att: DA,
}

/// Floats one block's workspace needs at `(c, h, w)`: three `2c`-wide planes, one
/// `c`-wide, and the two `[c]` vectors. The single source of the workspace
/// arithmetic - `PlanShape::of` uses it for the reported total and `Plan::new`
/// uses it to size the pool, so the number the engine reports and the bytes it
/// allocates cannot disagree.
fn workspace_floats(c: usize, h: usize, w: usize) -> usize {
    // TWO `2c`-wide planes (`t2`, `t3`), ONE `c`-wide (`g`), and the two `[c]`
    // vectors - 5c*hw + 2c, which is exactly the total `PlanShape::of` reports.
    //
    // `t1` IS NOT A PLANE OF ITS OWN: it is a view of `t3`'s first half. Every
    // value `t1` holds - norm1's output, the sca-scaled gate, norm2's output - is
    // `c` channels, and the op order in `Gpu::block` is the proof that `t3` is
    // dead exactly when `t1` is first written and that `t1` is consumed before
    // `t3` is written again. That is one plane at the block's shape, which is 512
    // MiB at 2048x2048 level 0.
    5 * c * h * w + 2 * c
}

/// The pre-shuffle staging shape at decoder level `l`.
///
/// `2 * c_before` CHANNELS AT HALF THE `up` SLOT'S RESOLUTION, where `c_before`
/// is the width of the level ABOVE the mirrored one - `width_at(levels - l)`,
/// except at `l = 0` where the input is the middle, `width_at(levels)`. Sizing it
/// by `width_at(levels - 1 - l)` coincides at `l = 0` and is off by a factor of
/// two everywhere else, which is why width 8 looked perfect and every larger
/// width did not.
///
/// `PlanShape::of` and the pass both call this, so the slot the plan sizes and
/// the view the decoder takes cannot drift apart.
fn pre_shape(geo: &Geometry, h: usize, wd: usize, l: usize) -> (usize, usize, usize) {
    let levels = geo.levels();
    let ml = levels - 1 - l;
    (
        2 * geo.width_at(levels - l),
        (h >> (ml + 1)).max(1),
        (wd >> (ml + 1)).max(1),
    )
}

/// The decoder's staging shape at level `l`: `width_at(levels-1-l)` channels at
/// the encoder's level-(levels-1-l) resolution. The CPU reference's own dump
/// fixes the rule - `ups.0 64 4 4`, `ups.1 32 8 8`, `ups.2 16 16 16`,
/// `ups.3 8 32 32` - which is `(width_at(levels-1-l), h >> (levels-1-l), ...)`.
/// Getting this wrong is not an error: the decoder runs, at the wrong
/// resolution, and the output is a different image.
fn up_shape(geo: &Geometry, h: usize, wd: usize, l: usize) -> (usize, usize, usize) {
    let levels = geo.levels();
    let ml = levels - 1 - l;
    (geo.width_at(ml), (h >> ml).max(1), (wd >> ml).max(1))
}

impl BlockScratch {
    /// The workspace for one block shape, as VIEWS into a pool.
    ///
    /// `base` is the pool's base pointer; nothing here owns its memory, so
    /// nothing here may be dropped as an owner (see `DA`'s `Drop`). `Plan` owns
    /// the pool for the whole pass.
    ///
    /// THE LAYOUT IS `t2 t3 g pooled att`, with `t2`/`t3` at `2c` channels, `g`
    /// at `c`, and the two `[c]` vectors last. `t1` is a view of `t3`'s first
    /// half and has no space of its own.
    /// `workspace_floats` is the same arithmetic, and the two are checked against
    /// each other by `Plan::new`'s size check.
    fn new(base: u64, c: usize, h: usize, w: usize) -> BlockScratch {
        let dw = 2 * c;
        let hw = h * w;
        let plane = |ch: usize, off: usize| DA {
            holds: false,
            buf: DevBuf { ptr: base + (off * 4) as u64, bytes: ch * hw * 4 },
            c: ch,
            h,
            w,
            tag: "workspace",
        };
        BlockScratch {
            // `t1` IS `t3`'s FIRST HALF. `t3` is `2c` wide and `t1` is `c`, and
            // the two are never live at once - `Gpu::block` writes `t1` only
            // after the gate product has been read out of `t3`, and consumes
            // `t1` before the FFN writes `t3` again. So the two planes are one
            // allocation and the block costs 5c*hw instead of 6c*hw.
            t1: plane(c, 2 * c * hw),
            t2: plane(dw, 0),
            t3: plane(dw, 2 * c * hw),
            g: plane(c, 4 * c * hw),
            pooled: DevBuf { ptr: base + ((5 * c * hw) * 4) as u64, bytes: c * 4 },
            att: DA {
                holds: false,
                buf: DevBuf { ptr: base + ((5 * c * hw + c) * 4) as u64, bytes: c * 4 },
                c,
                h: 1,
                w: 1,
                tag: "workspace att",
            },
        }
    }
}
