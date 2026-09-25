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
use lightgpu::vm::{copy_d2d, Args, DevBuf, Event, Launch};
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
    /// Buffers: `t1` the normalised input, `t2` the 2c expansion, `t3` the
    /// depthwise result, `g` the SimpleGate product, `t4` the sca result, `t5`
    /// the branch output, `y` the first residual, plus `pooled`/`att` for the
    /// attention. They are allocated once per forward pass and reused by every
    /// block, and a block that no longer needs one - `res`, before the residual
    /// became a single kernel - is not in `BlockScratch` at all.
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
        // first `c` channels are the first half by construction.
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
        self.channel_scale(&t.g, &t.att.buf, &t.t4)?;

        // conv3 back to c, then y = inp + t * beta.
        self.conv1x1(&t.t4, &p("conv3.weight"), Some(&p("conv3.bias")), half, c, &t.t5)?;
        // ONE KERNEL, NOT channel_scale + add: see `residual`.
        self.residual(cur, &t.t5, &p("beta"), &t.y)?;

        // FFN: norm2, conv4, SimpleGate, conv5, scaled by gamma.
        self.channel_layer_norm(&t.y, &p("norm2.weight"), &p("norm2.bias"), &t.t1)?;
        self.conv1x1(&t.t1, &p("conv4.weight"), Some(&p("conv4.bias")), c, dw, &t.t2)?;
        let f0 = Self::view(&t.t2.buf, half, h, wd, 0);
        let f1 = Self::view(&t.t2.buf, half, h, wd, half);
        self.mul(&f0, &f1, &t.g)?;
        self.conv1x1(&t.g, &p("conv5.weight"), Some(&p("conv5.bias")), half, c, &t.t5)?;
        self.residual(&t.y, &t.t5, &p("gamma"), cur)?;
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
        let mut plan = Plan::new(&self.cuda, geo, h, wd)?;

        // The input goes into its own slot in the plan, so it lives as long as
        // the residual add at the end needs it.
        plan.da("in").buf.upload(input)?;
        {
            let cur = plan.da("in");
            let out = plan.da("intro");
            self.conv3x3(cur, "intro.weight", Some("intro.bias"), 3, geo.width, out)?;
        }
        self.snap("intro", plan.da("intro"), host, &mut dump)?;

        // The ENCODER. `cur` names the activation in flight; `enc.{l}` is the
        // level's own slot and `down.{l}` the folded one. Nothing is ever
        // re-pointed: every transition is a copy between two distinct slots, so
        // a slot's (c,h,w) is a fixed property of the plan.
        plan.copy("intro", "enc.0")?;
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
            // The skip is its own slot: the encoder output is read again by the
            // decoder, long after the downsample has run.
            let skip_name = format!("skip.{l}");
            plan.copy(&cur, &skip_name)?;
            let down_name = format!("down.{l}");
            {
                let src = plan.da(&cur);
                let dst = plan.da(&down_name);
                self.downsample(src, l, c, dst)?;
            }
            self.snap(&format!("downs.{l}"), plan.da(&down_name), host, &mut dump)?;
            cur = down_name;
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

        // The DECODER. The first level takes the bottleneck, so its staging
        // slot is the middle activation itself; the rest read `pre.{l-1}`,
        // which is why `pre` and `up` are separate slots rather than one
        // reuse-by-name trick.
        for l in 0..levels {
            let c_after = geo.width_at(levels - 1 - l);
            // 1x1 conv to 4x the channels (bias-free), sized 2*c_after, at the
            // resolution the shuffle will double.
            let pre_name = format!("pre.{l}");
            {
                let src = plan.da(&cur);
                let dst = plan.da(&pre_name);
                let c_before = src.c;
                self.conv1x1(src, &format!("ups.{l}.0.weight"), None, c_before, c_before * 2, dst)?;
            }
            let up_name = format!("up.{l}");
            {
                let src = plan.da(&pre_name);
                let dst = plan.da(&up_name);
                self.pixel_shuffle2(src, c_after, dst)?;
            }
            // `ups.{l}` IS THE SHUFFLE OUTPUT, and the dump has to happen HERE:
            // before the skip add mutates this slot in place, and before the
            // decoder blocks rewrite it again. That is the stage
            // `tools/reference.py` dumps as `ups.{i}` (`x = up(x); dump(...)`
            // then `x = x + skips[-i-1]`), and snapping later compares a
            // post-decoder tensor against a pre-skip-add one - which reads as a
            // difference larger than either tensor's own peak.
            {
                let up = plan.da(&up_name);
                self.snap(&format!("ups.{l}"), up, host, &mut dump)?;
            }
            let skip_level = levels - 1 - l;
            let skip_name = format!("skip.{skip_level}");
            {
                let up = plan.da(&up_name);
                let skip = plan.da(&skip_name);
                self.add(up, skip, up)?;
            }
            // The decoder blocks run in place in the `up` slot at the PREVIOUS
            // shape, so they get their own scratch keyed by the SHAPE the
            // shuffle produced.
            // A lookup, to fail here rather than inside the block loop if the
            // plan never enumerated this shape.
            let (hh, ww) = (plan.da(&up_name).h, plan.da(&up_name).w);
            plan.scratch(c_after, hh, ww)?;
            for b in 0..geo.dec_blk_nums[l] {
                let prefix = format!("decoders.{l}.{b}");
                let hh = plan.da(&up_name).h;
                let ww = plan.da(&up_name).w;
                let t = plan.scratch(c_after, hh, ww)?;
                let mut cur_da = plan.da(&up_name).clone_meta();
                self.block(&prefix, c_after, &mut cur_da, t)?;
                self.snap(&prefix, &cur_da, host, &mut dump)?;
            }
            cur = up_name;
        }

        {
            let cur_da = plan.da(&cur);
            let out = plan.da("ending");
            self.conv3x3(cur_da, "ending.weight", Some("ending.bias"), geo.width, 3, out)?;
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
    slots: std::collections::HashMap<String, DA>,
    scratch: std::collections::HashMap<(usize, usize, usize), BlockScratch>,
}

impl Plan {
    fn new(cuda: &Cuda, geo: &Geometry, h: usize, wd: usize) -> Result<Plan, String> {
        let levels = geo.levels();
        let mut slots: std::collections::HashMap<String, DA> = std::collections::HashMap::new();
        let mut scratch = std::collections::HashMap::new();
        let mut put = |n: &str, c: usize, hh: usize, ww: usize| -> Result<(), String> {
            // A DUPLICATE NAME WOULD FREE THE FIRST ALLOCATION: `insert` returns
            // the old value and dropping it calls cuMemFree, so a name the pass
            // still reads would point at memory the driver may hand to someone
            // else. Names must be unique, so make a collision an error rather
            // than a silent free.
            if let Some(old) = slots.insert(n.to_string(), DA::new(c, hh, ww)?) {
                return Err(format!(
                    "plan slot `{n}` was allocated twice ({}x{}x{} then {}x{}x{})",
                    old.c, old.h, old.w, c, hh, ww
                ));
            }
            Ok(())
        };
        // One block workspace per shape the graph will actually use, and one
        // activation slot per name the executor asks for. `h`/`wd` are the
        // PADDED geometry, so every level is an exact halving.
        let (mut ch, mut cw) = (h, wd);
        for l in 0..levels {
            let c = geo.width_at(l);
            let key = (c, ch, cw);
            if !scratch.contains_key(&key) {
                scratch.insert(key, BlockScratch::new(cuda, c, ch, cw)?);
            }
            put(&format!("enc.{l}"), c, ch, cw)?;
            put(&format!("skip.{l}"), c, ch, cw)?;
            // `pre.{l}` IS THE PRE-SHUFFLE BUFFER: the decoder's 1x1 conv writes
            // `2*width_at(ml)` channels into it and `nf_pixel_shuffle2` reads
            // `4*c_after` = `4*width_at(ml)` planes out of it, so it must be
            // exactly that many channels AT HALF THE `up` RESOLUTION. Sizing it
            // by the encoder's `c` at the encoder's resolution wrote the
            // expansion past the allocation and read the shuffle from the wrong
            // planes - which is a wrong image, not a crash.
            let ml = levels - 1 - l;
            // THE PRE-SHUFFLE SLOT IS `2 * c_before` CHANNELS WIDE, at the
            // resolution one halving below the `up` slot it feeds. The decoder's
            // 1x1 conv at level l reads `cur` - which at l is the level ABOVE the
            // mirrored one, `width_at(levels-l)`, except at l=0 where it is the
            // middle, `width_at(levels)` - and writes `c_before * 2` channels;
            // `nf_pixel_shuffle2` then reads `4 * c_after` planes back out.
            // Sizing it by `width_at(levels-1-l)` happens to coincide at l=0
            // (where the middle is twice that level) and is off by a factor of
            // two everywhere else, which is why width 8 looked perfect and every
            // larger width did not.
            let (ph, pw) = ((h >> (ml + 1)).max(1), (wd >> (ml + 1)).max(1));
            put(&format!("pre.{l}"), 2 * geo.width_at(levels - l), ph, pw)?;
            // `up.{l}` IS THE DECODER'S SLOT FOR THE SAME LEVEL INDEX, AND BOTH
            // ITS AXES COME FROM THE MIRRORED LEVEL. The decoder walks the
            // levels in reverse: at l its channel count is
            // `width_at(levels-1-l)` and its resolution is the encoder's
            // level-(levels-1-l) resolution, NOT `2*ch` from the running loop
            // counter. The CPU reference's own dump fixes the rule -
            // `ups.0 64 4 4`, `ups.1 32 8 8`, `ups.2 16 16 16`, `ups.3 8 32 32`
            // - which is `(width_at(levels-1-l), h >> (levels-1-l), wd >> ...)`.
            // Getting this wrong is not an error: the decoder runs, at the wrong
            // resolution, and the output is a different image.
            let (uh, uw) = ((h >> ml).max(1), (wd >> ml).max(1));
            put(&format!("up.{l}"), geo.width_at(ml), uh, uw)?;
            // THE DOWNSAMPLE OUTPUT IS HALF THE RESOLUTION, TWICE THE CHANNELS.
            // Allocating it at `(2c, ch, cw)` - the source resolution with the
            // doubled channel count - made every `down.{l}` four times its real
            // size, so `cur` carried a shape the network does not have and the
            // next level's `copy down.{l} -> skip.{l+1}` was refused by
            // `Plan::copy`'s own consistency guard. `ch`/`cw` halve after this,
            // so the next iteration's `key` already covers this shape and the
            // separate `key2` scratch set was both wrong and redundant.
            let (dh, dwd) = ((ch / 2).max(1), (cw / 2).max(1));
            put(&format!("down.{l}"), 2 * c, dh, dwd)?;
            // THE DECODER'S BLOCK SHAPE IS ITS OWN KEY, at the mirrored level's
            // resolution - the shape the decoder stage actually runs its blocks
            // at, which the encoder never visits at that channel count.
            let key3 = (geo.width_at(ml), uh, uw);
            if !scratch.contains_key(&key3) {
                scratch.insert(key3, BlockScratch::new(cuda, geo.width_at(ml), uh, uw)?);
            }
            ch /= 2;
            cw /= 2;
        }
        // THE BARE `insert` HERE IS A FREE WAITING TO HAPPEN: `insert` returns the
        // old value, and dropping a BlockScratch calls cuMemFree on every buffer
        // in it - while the plan's slots still name the addresses the pass will
        // read. The loops above guard with `contains_key`, this one did not, and
        // the middle shape repeats a key whenever `middle_width() == 2 * width_at(0)`
        // at the smallest spatial size.
        let mid_key = (geo.middle_width(), ch, cw);
        if !scratch.contains_key(&mid_key) {
            scratch.insert(mid_key, BlockScratch::new(cuda, geo.middle_width(), ch, cw)?);
        }
        put("in", 3, h, wd)?;
        put("intro", geo.width, h, wd)?;
        put("ending", 3, h, wd)?;
        // The decoder's activations reuse the encoder's slots by name, so the
        // executor's slot set is exactly: in, intro, enc, skip, down, pre, up,
        // ending - nothing else is allocated during the pass.
        //
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
        if std::env::var("NAFNET_DEBUG_VRAM").is_ok() {
            let own: usize = slots.values().map(|d| d.buf.bytes).sum::<usize>()
                + scratch
                    .values()
                    .map(|s| {
                        s.t1.buf.bytes
                            + s.t2.buf.bytes
                            + s.t3.buf.bytes
                            + s.t4.buf.bytes
                            + s.t5.buf.bytes
                            + s.g.buf.bytes
                            + s.y.buf.bytes
                            + s.pooled.bytes
                            + s.att.buf.bytes
                    })
                    .sum::<usize>();
            let free = lightgpu::vm::free_vram().unwrap_or(0);
            eprintln!(
                "plan owns {:.1} MiB in {} slots + {} scratch sets; {} MiB free",
                own as f64 / 1048576.0,
                slots.len(),
                scratch.len(),
                free / 1048576
            );
        }
        Ok(Plan { slots, scratch })
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

    /// Copy the CONTENTS of one slot into another.
    ///
    /// NOT a pointer handoff: both slots keep their own allocation, so a slot's
    /// (c,h,w) never changes identity. An earlier version re-pointed the name
    /// instead, which is what made a `copy_d2d` name a range that did not exist
    /// (the slot's metadata and its buffer disagreed) - and it also silently
    /// destroyed the input slot the final residual add reads.
    fn copy(&mut self, src: &str, dst: &str) -> Result<(), String> {
        let (sp, sn, sb) = {
            let s = self.slots.get(src).expect("copy source");
            (s.buf.ptr, s.n(), s.buf.bytes)
        };
        let (dp, dn, db) = {
            let d = self.slots.get(dst).expect("copy destination");
            (d.buf.ptr, d.n(), d.buf.bytes)
        };
        if sn != dn || sn * 4 > sb || sn * 4 > db {
            return Err(format!(
                "copy {src} -> {dst}: {sn} floats into {dn} (buffers {sb} and {db} bytes)"
            ));
        }
        copy_d2d(dp, sp, sn * 4).map_err(|e| format!("copy {src} -> {dst} ({sn} floats): {e}"))
    }
}

/// The workspace ONE NAFBlock needs, at one shape. One set per distinct
/// (c, h, w) in the plan, shared by every block that runs at that shape.
///
/// FIELD NAMES ARE THE SAME NAMES `net::block_forward` USES, so the two
/// implementations of the block can be read side by side; `t5` and `t4` are
/// separate planes rather than one reused buffer because the residual kernel
/// reads `t5` while writing the activation `t4` is no longer needed for.
/// Sizing is by ROLE, not by the widest thing the shape could hold - see
/// `BlockScratch::new`.
struct BlockScratch {
    t1: DA,
    t2: DA,
    t3: DA,
    t4: DA,
    t5: DA,
    g: DA,
    y: DA,
    pooled: DevBuf,
    att: DA,
}

impl BlockScratch {
    /// The scratch for one block shape.
    fn new(cuda: &Cuda, c: usize, h: usize, w: usize) -> Result<BlockScratch, String> {
        let dw = 2 * c;
        let half = c;
        Ok(BlockScratch {
            t1: DA::new(c, h, w)?,
            // `t2`/`t3` hold the 2c expansion; `t4` is the sca-scaled GATE
            // (half the channels) and `t5` the branch output back at c.
            t2: DA::new(dw, h, w)?,
            t3: DA::new(dw, h, w)?,
            t4: DA::new(half, h, w)?,
            t5: DA::new(c, h, w)?,
            g: DA::new(half, h, w)?,
            y: DA::new(c, h, w)?,
            pooled: cuda.buf(half)?,
            att: DA::new(half, 1, 1)?,
        })
    }
}
