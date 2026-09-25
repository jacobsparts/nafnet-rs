//! The NAFNet graph: geometry, weights, and the CPU kernels that mirror the GPU
//! ones.
//!
//! The architecture is a single-stage U-Net: a 3x3 `intro` conv, then for each
//! level a stack of NAFBlocks followed by a stride-2 2x2 conv that doubles the
//! channels, then the middle block stack, then for each level a 1x1 conv +
//! PixelShuffle(2) whose output is added to the matching encoder skip and fed
//! through another NAFBlock stack, then a 3x3 `ending` conv and a global
//! residual. Everything between `intro` and `ending` is NAFBlocks.
//!
//! A NAFBLOCK HAS NO ACTIVATION FUNCTION. That is the paper's point, and it is
//! also what makes the block cheap to port: the only non-linearity is
//! SimpleGate's multiply.
//!
//! ```text
//!   x = norm1(inp)
//!   x = conv2(conv1(x))         1x1 c->2c then depthwise 3x3 (groups=2c)
//!   g = x.chunk(2, dim=1)       SimpleGate: split the CHANNEL axis in half
//!   x = g.0 * g.1
//!   x = x * sca(x)              sca = 1x1 conv over the SAME c channels,
//!                               applied to the map globally averaged first
//!   x = conv3(x)                1x1 c->c
//!   y = inp + x * beta          beta is per-channel, trained from zero
//!   x = conv5(conv4(norm2(y)))  1x1 c->2c, SimpleGate, 1x1 c->c
//!   return y + x * gamma
//! ```
//!
//! `norm1` and `norm2` are `LayerNorm2d`: a LayerNorm over the CHANNEL axis of
//! an NCHW tensor, with per-channel weight and bias and eps 1e-6. That is the
//! toolkit's `lg_channel_layer_norm`, and it is NOT the toolkit's
//! `lg_layer_norm`, whose reduction is over the contiguous rows of the other
//! layout - reading the wrong axis here produces an image rather than an error.
//!
//! The checkpoint's conv weights are already `[c_out][c_in][kh][kw]` for the
//! dense convs and `[c_out][1][3][3]` for the depthwise one, which is the layout
//! both backends want, so nothing is transposed at load.
use crate::weights::Weights;
use rayon::prelude::*;

/// The reference's `LayerNorm2d` default, and therefore the eps BOTH backends
/// must normalise with. It lives here rather than in two places because a
/// mismatch is not an error - it is a slightly different image.
pub const LAYERNORM_EPS: f32 = 1e-6;

/// Fused multiply-add: `a * b + c` with ONE rounding, where the target can.
///
/// THE CUDA KERNELS FUSE AND THE PORT MUST TOO. `lightgpu/build-support` passes
/// nvcc `--fmad=true` (see `conv1x1`), so every `acc += w * x` in a kernel is a
/// single-rounded FFMA. `f32::mul_add` is the fused form - but on a target
/// without the `fma` feature rustc lowers it to a CALL into
/// `compiler_builtins`' soft-float `fmaf`, which also blocks vectorisation:
/// using `mul_add` directly in this file took the CPU path from 96 s to 550 s of
/// user time for one 1280x725 image, with the same results. Hence the cfg: fused
/// on an FMA target, plain otherwise. The plain form is fast and correct but
/// rounds twice, so it will not match the kernels to the last bit.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_feature = "fma")]
    let r = a.mul_add(b, c);
    #[cfg(not(target_feature = "fma"))]
    let r = a * b + c;
    r
}

/// PER-OP TIMING FOR THE CPU BACKEND, which is the same idea as the GPU path's
/// CUDA-event profile and exists for the same reason: without it, tuning is
/// guesswork. This box has no `perf` and no `gdb`, so there is no sampler to fall
/// back on either.
///
/// WHEN OFF IT COSTS ONE `OnceLock::get` PER OP CALL, and an op here is a whole
/// tensor pass over up to a hundred megabytes - so the instrumentation cannot
/// perturb what it measures. When on, each op records its own WALL time, and
/// since every op is internally parallel those times sum to about the pass wall
/// rather than to a thread-seconds figure, which is what makes them directly
/// comparable with the numbers the GPU profile prints.
///
/// Enabled by `NAFNET_CPU_PROFILE=1`, read once at the top of the pass.
pub static CPU_PROF: std::sync::OnceLock<
    std::sync::Mutex<std::collections::BTreeMap<&'static str, (f64, usize)>>,
> = std::sync::OnceLock::new();

/// Turn the CPU profile on. Without this `prof_start` returns `None` and no op
/// does any recording work at all.
pub fn cpu_prof_enable() {
    let _ = CPU_PROF.set(std::sync::Mutex::new(std::collections::BTreeMap::new()));
}

/// Records into `CPU_PROF` when dropped, so an early return cannot lose a
/// sample and every op needs exactly one line to be measured.
pub struct CpuTimer {
    name: &'static str,
    t: std::time::Instant,
}

impl Drop for CpuTimer {
    fn drop(&mut self) {
        if let Some(m) = CPU_PROF.get() {
            if let Ok(mut g) = m.lock() {
                let e = g.entry(self.name).or_insert((0.0, 0));
                e.0 += self.t.elapsed().as_secs_f64() * 1000.0;
                e.1 += 1;
            }
        }
    }
}

/// Start timing `name`, or do nothing at all when profiling is off.
#[inline]
pub fn prof_start(name: &'static str) -> Option<CpuTimer> {
    CPU_PROF.get().map(|_| CpuTimer { name, t: std::time::Instant::now() })
}

/// Print the totals, longest first.
pub fn cpu_prof_report() {
    let Some(m) = CPU_PROF.get() else { return };
    let g = match m.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let mut rows: Vec<(&str, f64, usize)> = g.iter().map(|(k, v)| (*k, v.0, v.1)).collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total: f64 = rows.iter().map(|r| r.1).sum();
    println!("{:>10} {:>6} {:>10}  op", "total ms", "count", "mean ms");
    for (n, ms, c) in &rows {
        let mean = if *c == 0 { 0.0 } else { ms / (*c as f64) };
        println!("{:>10.1} {:>6} {:>10.3}  {n}", ms, c, mean);
    }
    println!("{:>10.1} {:>6} {:>10}  (summed over all ops)", total, "", "");
}

/// The block size the toolkit's `lg_channel_mean` reduces with. The CPU twin
/// must use the same one to accumulate in the same order, since that op sums
/// strided partials into a per-thread slot and then halves a tree - not a serial
/// sum over the plane.
pub const CHANNEL_MEAN_BLOCK: usize = 256;

/// Model geometry, read from the checkpoint metadata.
///
/// WHICH HALF OF THIS IS `cuda`-GATED: `levels` and `padder_size` are what the
/// CPU path needs (the level count and the pad). The rest - the feature width,
/// the middle block count and the per-level block counts - is read by the GPU
/// executor and by `gpu::Plan` alone, so a CPU-only build has no reader for them
/// and the compiler says so. Gating them here rather than silencing the warning
/// keeps both configurations checked.
#[derive(Clone, Debug)]
pub struct Geometry {
    #[cfg(feature = "cuda")]
    pub width: usize,
    pub enc_blk_nums: Vec<usize>,
    #[cfg(feature = "cuda")]
    pub middle_blk_num: usize,
    #[cfg(feature = "cuda")]
    pub dec_blk_nums: Vec<usize>,
}

impl Geometry {
    pub fn levels(&self) -> usize {
        self.enc_blk_nums.len()
    }

    /// The padder size: the input is padded to a multiple of this.
    pub fn padder_size(&self) -> usize {
        1 << self.levels()
    }

    /// Feature width at level `l` into the encoder. Level 0 is `width`; the
    /// middle sits at `width << levels`.
    #[cfg(feature = "cuda")]
    pub fn width_at(&self, level: usize) -> usize {
        self.width << level
    }

    #[cfg(feature = "cuda")]
    pub fn middle_width(&self) -> usize {
        self.width << self.levels()
    }
}

/// Borrowed views of the tensors one NAFBlock needs.
struct Block<'a> {
    conv1_w: &'a [f32],
    conv1_b: &'a [f32],
    conv2_w: &'a [f32],
    conv2_b: &'a [f32],
    conv3_w: &'a [f32],
    conv3_b: &'a [f32],
    sca_w: &'a [f32],
    sca_b: &'a [f32],
    conv4_w: &'a [f32],
    conv4_b: &'a [f32],
    conv5_w: &'a [f32],
    conv5_b: &'a [f32],
    n1w: &'a [f32],
    n1b: &'a [f32],
    n2w: &'a [f32],
    n2b: &'a [f32],
    beta: &'a [f32],
    gamma: &'a [f32],
}

impl<'a> Block<'a> {
    /// `c` is not a parameter: a field holding it was stored and never read, and
    /// the block's channel count is already implied by the weight shapes that
    /// `weights.rs` asserts at load.
    fn load(w: &'a Weights, prefix: &str) -> Result<Block<'a>, String> {
        let g = |s: &str| w.get(&format!("{prefix}.{s}"));
        Ok(Block {
            conv1_w: g("conv1.weight")?,
            conv1_b: g("conv1.bias")?,
            conv2_w: g("conv2.weight")?,
            conv2_b: g("conv2.bias")?,
            conv3_w: g("conv3.weight")?,
            conv3_b: g("conv3.bias")?,
            sca_w: g("sca.1.weight")?,
            sca_b: g("sca.1.bias")?,
            conv4_w: g("conv4.weight")?,
            conv4_b: g("conv4.bias")?,
            conv5_w: g("conv5.weight")?,
            conv5_b: g("conv5.bias")?,
            n1w: g("norm1.weight")?,
            n1b: g("norm1.bias")?,
            n2w: g("norm2.weight")?,
            n2b: g("norm2.bias")?,
            beta: g("beta")?,
            gamma: g("gamma")?,
        })
    }
}

/// A planar NCHW activation: `data` is [c][h][w].
pub struct Act {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Act {
    pub fn new(c: usize, h: usize, w: usize) -> Act {
        Act { c, h, w, data: vec![0.0; c * h * w] }
    }

    #[inline]
    pub fn hw(&self) -> usize {
        self.h * self.w
    }

}

/// Output channels per task in [`conv1x1`], and positions per register tile.
///
/// MEASURED, AND THE RESULT IS THE OPPOSITE OF WHAT THE GPU KERNEL WANTED. On
/// the GPU, widening the OUTPUT-CHANNEL tile is what cut the cost (`nf_conv1x1_oc`
/// runs OC_TILE=16); here, widening it HURTS at every position tile, while
/// widening the POSITION tile helps a lot. Min-of-2 `conv1x1` times in ms at
/// 1280x725, the same figure the function's comment quotes:
///
///   OC_TILE \ P_TILE   8     16     32     64    128    256
///          1                   --     --   1791   1580   1716
///          2                   --    --   1651   1462   **1351**
///          4                   --   2447  1586   1564     --
///          8                 4571  2576  1781   1627     --
///         16                 5390  3002  2167   1776     --
///         32                 8106  4185  2992     --     --
///
/// The two axes are not independent because `OC_TILE * P_TILE` is the number of
/// live accumulators, and that total is what the machine cares about - 512
/// flonums is 32 AVX vectors, which is already past what can be kept in flight.
/// Given a fixed budget, the position axis is worth more here: a longer
/// contiguous run vectorises and lets each loaded weight be reused across more
/// lanes, whereas the extra output channels buy nothing that the L1 was not
/// already giving. On the GPU the trade goes the other way.
const OC_TILE: usize = 2;
const P_TILE: usize = 256;

/// SEVERAL TWINS BELOW TAKE A PARAMETER THE KERNEL NEEDS AND THE HOST DOES NOT,
/// AND IT IS UNDERSCORED RATHER THAN REMOVED. `c_out` and `c` are implied by the
/// length of the output slice the caller passes, but keeping the name in place
/// keeps each signature a line-by-line match for the kernel's argument list -
/// which is the whole reason these functions exist as twins. Removing the
/// parameter would make the two lists harder to compare, so the unused ones are
/// `_c_out` / `_c` and the kernels they mirror are named in each doc comment.
///
/// 1x1 conv, NCHW: `out[oc] = bias[oc] + sum_ic w[oc][ic] * in[ic]`.
///
/// Accumulated in `ic` order, which is what the toolkit's `lg_conv1x1` does, so
/// the two backends agree bit for bit on a given build.
pub fn conv1x1(
    input: &[f32],
    w: &[f32],
    b: &[f32],
    c_in: usize,
    _c_out: usize,
    hw: usize,
    out: &mut [f32],
) {
    let _p = prof_start("conv1x1");
    // TILED OVER OUTPUT CHANNELS, AND THIS IS THE SINGLE BIGGEST COST IN THE CPU
    // BACKEND. The loop used to be `par_chunks_mut(hw)`, i.e. one parallel task
    // per OUTPUT CHANNEL, each walking all `c_in` input planes - so the input was
    // read `c_out` times for one pass over the output. Measured with
    // `NAFNET_CPU_PROFILE=1` at 1280x725, this op was 3492 ms of a 4566 ms pass
    // (76.5%), and the arithmetic says why: one level-3 conv reads
    // 512 x 256 x 10120 x 4 = 5.3 GB of input for a 20 MB output, ~150 GB over
    // the 28 level-3 blocks, and 150 GB at this box's 43.9 GB/s is 3.4 s against
    // the measured 3.49 s. The FLOPs for the same work are ~0.33 s of the
    // machine's peak, so it was 10x off compute and all of the gap was re-reading.
    //
    // Processing OC_TILE channels per task divides that by OC_TILE: one load of
    // `input[ic]` now feeds OC_TILE fused multiply-adds. The GPU kernel
    // `nf_conv1x1_oc` does the same thing, which is why the two backends keep
    // agreeing.
    //
    // THE ACCUMULATION ORDER PER OUTPUT ELEMENT IS UNCHANGED - bias first, then
    // one fused multiply-add per `ic` ascending - because only the loop
    // STRUCTURE around the sum moved, never the sum. The 32x32 stage dump is a
    // bit-identity test of that, not a tolerance.
    out.par_chunks_mut(OC_TILE * hw).enumerate().for_each(|(ob, o)| {
        let oc0 = ob * OC_TILE;
        // The last task may be short: `c_out` need not be a multiple of OC_TILE.
        let mut planes: Vec<&mut [f32]> = o.chunks_mut(hw).collect();
        let nt = planes.len();
        let empty_bias = b.is_empty();
        // An EMPTY bias slice means a bias-free conv - the decoder's upsample
        // 1x1. The GPU twin passes a null pointer for the same case, and the
        // toolkit kernel's `bias ? bias[oc] : 0.f` is the same contract.
        let mut p0 = 0usize;
        while p0 < hw {
            let n = P_TILE.min(hw - p0);
            // THE ACCUMULATORS LIVE IN REGISTERS, and that is the whole point:
            // with `OC_TILE x P_TILE` of them, each weight load feeds P_TILE
            // fused multiply-adds, each input load feeds OC_TILE of them, and the
            // output is written EXACTLY ONCE. The first attempt at this tiling
            // kept the accumulators in the output buffer and read-modify-wrote
            // `OC_TILE` rows per `ic`, which measured WORSE than the untiled loop
            // (5157 ms against 3492) because the original kept its single output
            // row in L1 for the whole sweep and the tiled one spilled to L2/L3.
            let mut acc = [[0.0f32; P_TILE]; OC_TILE];
            for t in 0..nt {
                let bias = if empty_bias { 0.0 } else { b[oc0 + t] };
                for j in 0..n {
                    acc[t][j] = bias;
                }
            }
            for ic in 0..c_in {
                let x = &input[ic * hw + p0..ic * hw + p0 + n];
                for t in 0..nt {
                    // FUSED, unlike `*o += k * x`: the toolkit is compiled with
                    // nvcc's `--fmad=true` (see lightgpu's build script), which
                    // contracts this multiply-add into a single-rounded FFMA, and
                    // Rust disables FP contraction by default. Fusing is both the
                    // faithful twin and the more accurate sum - see `fma` above
                    // for why it is not a bare `mul_add`.
                    let k = w[(oc0 + t) * c_in + ic];
                    for j in 0..n {
                        acc[t][j] = fma(k, x[j], acc[t][j]);
                    }
                }
            }
            for t in 0..nt {
                planes[t][p0..p0 + n].copy_from_slice(&acc[t][..n]);
            }
            p0 += n;
        }
    });
}

/// 3x3 pad-1 dense conv. Accumulation order: ky, kx, ci - matching the toolkit's
/// `lg_conv3x3s1p1`.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3(
    input: &[f32],
    w: &[f32],
    b: &[f32],
    c_in: usize,
    _c_out: usize,
    h: usize,
    wd: usize,
    out: &mut [f32],
) {
    let _p = prof_start("conv3x3");
    let hw = h * wd;
    // ONE TASK PER (OUTPUT CHANNEL, OUTPUT ROW), NOT PER OUTPUT CHANNEL, and both
    // halves of that matter.
    //
    // Why the ROW: `ending` is a 32 -> 3 conv, so a per-channel split gives it
    // THREE parallel tasks on a 24-core machine while `intro` (3 -> 32) gets 32.
    // The two launches cost 135 ms each at 1280x725, which is what a barrier
    // against a mostly-idle machine looks like. Chunking by output row gives
    // `c_out * h` tasks - 2208 for `ending`, 23552 for `intro` - and the row is
    // the natural unit anyway because the output is written exactly once.
    //
    // Why the CHANNEL is still in the task: it is what makes a row's weight set
    // and bias scalar for the whole inner loop.
    //
    // THE INNER LOOP IS NOW A CONTIGUOUS RANGE with no branch in it. The old one
    // tested `sx < 0 || sx >= wd` PER OUTPUT ELEMENT, inside the innermost loop
    // of a 27-tap convolution, which blocks vectorisation outright; here each
    // (ky, kx) tap contributes to output columns `lo..hi` derived ONCE from the
    // tap's horizontal offset, and every remaining access is in bounds by
    // construction.
    //
    // Accumulating into a row buffer and storing at the end performs the same
    // operations in the same ORDER per element - `ky`, `kx`, then `ic` - so this
    // is a re-scheduling, not a different sum, and the 32x32 dump stays
    // bit-identical.
    out.par_chunks_mut(wd).enumerate().for_each_init(
        Vec::new,
        |acc, (idx, o)| {
            let oc = idx / h;
            let y = idx % h;
            if acc.len() < wd {
                acc.resize(wd, 0.0);
            }
            let bias = b[oc];
            for a in acc[..wd].iter_mut() {
                *a = bias;
            }
            let wp = &w[oc * c_in * 9..(oc + 1) * c_in * 9];
            for ky in 0..3usize {
                let sy = y as isize + ky as isize - 1;
                if sy < 0 || sy >= h as isize {
                    continue;
                }
                let srow = sy as usize * wd;
                for kx in 0..3usize {
                    // Columns of the OUTPUT row that this tap can reach.
                    let lo = if kx == 0 { 1 } else { 0 };
                    let hi = if kx == 2 { wd - 1 } else { wd };
                    if lo >= hi {
                        continue;
                    }
                    for ic in 0..c_in {
                        let k = wp[ic * 9 + ky * 3 + kx];
                        if k == 0.0 {
                            continue;
                        }
                        let xrow = &input[ic * hw + srow..ic * hw + srow + wd];
                        // Contiguous in `x` for a fixed tap: this is the loop that
                        // vectorises.
                        for x in lo..hi {
                            acc[x] = fma(k, xrow[x + kx - 1], acc[x]);
                        }
                    }
                }
            }
            o.copy_from_slice(&acc[..wd]);
        },
    );
}

/// 3x3 depthwise conv: one 3x3 kernel per channel, `groups == c`. Accumulation
/// order ky, kx, matching `nf_conv3x3_dw`.
pub fn conv3x3_dw(
    input: &[f32],
    w: &[f32],
    b: &[f32],
    _c: usize,
    h: usize,
    wd: usize,
    out: &mut [f32],
) {
    let _p = prof_start("conv3x3_dw");
    let hw = h * wd;
    // THE SAME RESTRUCTURING AS `conv3x3`, and for the same two reasons: one task
    // per CHANNEL is only 32 tasks at level 0 and 512 at level 3 with wildly
    // uneven rows, and the bounds test sat in the innermost loop of a 9-tap
    // convolution. Splitting by output row gives `c * h` tasks (23552 at level 0,
    // 47104 at level 3) and leaves a contiguous branch-free inner loop.
    out.par_chunks_mut(wd).enumerate().for_each_init(
        Vec::new,
        |acc, (idx, o)| {
            let ch = idx / h;
            let y = idx % h;
            if acc.len() < wd {
                acc.resize(wd, 0.0);
            }
            let x = &input[ch * hw..(ch + 1) * hw];
            let wk = &w[ch * 9..(ch + 1) * 9];
            let bias = b[ch];
            for a in acc[..wd].iter_mut() {
                *a = bias;
            }
            // Order stays `ky` then `kx`, matching `nf_conv3x3_dw`.
            for ky in 0..3usize {
                let sy = y as isize + ky as isize - 1;
                if sy < 0 || sy >= h as isize {
                    continue;
                }
                let srow = sy as usize * wd;
                for kx in 0..3usize {
                    let lo = if kx == 0 { 1 } else { 0 };
                    let hi = if kx == 2 { wd - 1 } else { wd };
                    if lo >= hi {
                        continue;
                    }
                    let k = wk[ky * 3 + kx];
                    let xrow = &x[srow..srow + wd];
                    for x0 in lo..hi {
                        acc[x0] = fma(k, xrow[x0 + kx - 1], acc[x0]);
                    }
                }
            }
            o.copy_from_slice(&acc[..wd]);
        },
    );
}

/// Stride-2 2x2 conv (the downsample): `out[oc][y][x] = b + sum_ic sum_ky,kx
/// w[oc][ic][ky][kx] * in[ic][2y+ky][2x+kx]`.
pub fn conv2x2s2(
    input: &[f32],
    w: &[f32],
    b: &[f32],
    c_in: usize,
    _c_out: usize,
    h: usize,
    wd: usize,
    out: &mut [f32],
) {
    let _p = prof_start("down2x2s2");
    let (oh, ow) = (h / 2, wd / 2);  // input is h x wd, the output is half of each
    let hw = h * wd;
    // ONE TASK PER (OUTPUT CHANNEL, OUTPUT ROW). The old version took one task per
    // output channel and walked the whole input inside it, so the input plane was
    // read `c_out` times - 1.9 G floats, 7.7 GB, for the level-0 launch alone.
    // Row splits cost nothing here because each output row needs exactly two
    // input rows, so a row task reads `c_in * 2 * wd` and neighbouring rows share
    // almost all of their input rows in cache.
    //
    // THE ACCUMULATION CHAIN IS UNCHANGED. `acc` starts at the bias and `ic`
    // folds its four taps in (ky, kx) order, then the row is stored - the same
    // operations in the same order as before. Summing each `ic`'s four taps from
    // zero first and adding the total afterwards is a different summation tree
    // over the same terms, and it showed up as 2 ulp.
    out.par_chunks_mut(ow).enumerate().for_each_init(
        Vec::new,
        |acc, (idx, o)| {
            let oc = idx / oh;
            let oy = idx % oh;
            if acc.len() < ow {
                acc.resize(ow, 0.0);
            }
            let bias = b[oc];
            for a in acc[..ow].iter_mut() {
                *a = bias;
            }
            let wp = &w[oc * c_in * 4..(oc + 1) * c_in * 4];
            let r0 = (2 * oy) * wd;
            let r1 = (2 * oy + 1) * wd;
            for ic in 0..c_in {
                let x = &input[ic * hw..(ic + 1) * hw];
                let (t0, t1) = (&x[r0..r0 + wd], &x[r1..r1 + wd]);
                let k = &wp[ic * 4..ic * 4 + 4];
                for ox in 0..ow {
                    let mut a = acc[ox];
                    a = fma(k[0], t0[2 * ox], a);
                    a = fma(k[1], t0[2 * ox + 1], a);
                    a = fma(k[2], t1[2 * ox], a);
                    a = fma(k[3], t1[2 * ox + 1], a);
                    acc[ox] = a;
                }
            }
            o.copy_from_slice(&acc[..ow]);
        },
    );
}

/// Depth-to-space, the inverse of the toolkit's `lg_pixel_unshuffle2`:
/// `out[c][2y+dy][2x+dx] = in[c*4 + dy*2 + dx][y][x]`.
pub fn pixel_shuffle2(input: &[f32], _c: usize, h: usize, wd: usize, out: &mut [f32]) {
    // `c` is the OUTPUT channels; the input carries 4 * c of them, grouped as
    // (dy, dx) for each output channel. The kernel takes it because its grid is
    // sized by it; here the task list comes from the output slice.
    let _p = prof_start("pixel_shuffle2");
    let (oh, ow) = (2 * h, 2 * wd);
    let hw = h * wd;
    let ohw = oh * ow;
    out.par_chunks_mut(ohw).enumerate().for_each(|(oc, o)| {
        for y in 0..h {
            for x in 0..wd {
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        let v = input[(oc * 4 + dy * 2 + dx) * hw + y * wd + x];
                        o[(2 * y + dy) * ow + 2 * x + dx] = v;
                    }
                }
            }
        }
    });
}

/// LayerNorm over the CHANNEL axis of an NCHW tensor, eps 1e-6.
///
/// THE REDUCTION IS OVER THE CHANNELS AT EACH SPATIAL POSITION, not over the
/// spatial positions of each channel. For every `p` in `0..hw`, the `c` values
/// `x[ch * hw + p]` are normalised together, using per-channel `w` and `b` after
/// the reduction. That is what `LayerNormFunction` in the reference does (its
/// mean/var are over C at each (n,h,w)) and what the toolkit's
/// `lg_channel_layer_norm` does - one work item per position, walking the `c`
/// values spaced `hw` apart.
///
/// THE SUMMATION ORDER IS PART OF THE CONTRACT, and this twin takes the
/// kernel's. The op used to reduce with a shared-memory halving tree (one BLOCK
/// per position); it is now one THREAD per position summing `c` SERIALLY
/// ASCENDING - which is what the loop below does, in the same order. The
/// multiplications follow `conv1x1` too, where the device's arithmetic is fused
/// (see that function for the `--fmad=true` story); where measurement says the
/// kernel is NOT fused, the plain form is kept and the code says so.
///
/// THE VARIANCE IS STILL THE ONE-PASS `E[x^2] - mean^2` FORM, mirroring the
/// kernel exactly, for the same reason. The two-pass form is numerically better,
/// but it would be a different op from the one being checked.
///
/// ONE TERM CANNOT BE MATCHED: the kernel's scale is
/// `rsqrtf(fmaxf(var, 0) + eps)`, CUDA's approximate reciprocal square root,
/// while this computes `1.0 / (..).sqrt()`. `--prec-sqrt=true` governs how a
/// division and a `sqrt` are generated, not what an explicit `rsqrtf()` returns,
/// and Rust exposes no equivalent intrinsic - so `--cuda-selftest` reports this
/// op at 2.384e-7 (2 ulp of f32) and not at zero. Every other op in the engine
/// compares bit-exact.
pub fn channel_layer_norm(
    input: &[f32],
    w: &[f32],
    b: &[f32],
    c: usize,
    hw: usize,
    out: &mut [f32],
) {
    let _p = prof_start("channel_layer_norm");
    let n = c as f32;
    // Statistics parallel over POSITIONS, each position walking its own
    // stride-`hw` column - a position-indexed map, not a chunked iterator,
    // because the input is read at stride `hw`.
    let stats: Vec<(f32, f32)> = (0..hw)
        .into_par_iter()
        .map(|p| {
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            for ch in 0..c {
                let v = input[ch * hw + p];
                s1 += v;
                // FUSED, and measured rather than assumed: contracting this one
                // improves BOTH the op-level and the graph-level agreement with
                // the GPU (`--cuda-selftest` cannot see it - that row is dominated
                // by the `rsqrtf` note below), while leaving `var` below plain is
                // better than fusing it. ptxas decides per expression, so each is
                // decided here by measurement.
                s2 = fma(v, v, s2);
            }
            let mean = s1 / n;
            // NOT fused, and measured rather than assumed: writing
            // `(-mean).mul_add(mean, s2 / n)` here moved the op's selftest row not
            // at all (it is `rsqrtf`, below) while making the graph-level
            // cpu-vs-gpu agreement WORSE (7.32e-03 -> 1.05e-02 worst stage), so
            // ptxas is not contracting this expression the way `--fmad=true` lets
            // it contract the convs. The plain form is the better match.
            let var = s2 / n - mean * mean;
            (mean, 1.0 / (var.max(0.0) + LAYERNORM_EPS).sqrt())
        })
        .collect();
    // The OUTPUT is parallel over CHANNELS: this pass is `c * hw` elements, the
    // largest single-threaded stretch in the CPU forward path, and it ran serially
    // until now. Each element is written by the same expression as before, so the
    // values are unchanged; only the thread that computes them is.
    out.par_chunks_mut(hw).enumerate().for_each(|(ch, row)| {
        let (ww, bb) = (w[ch], b[ch]);
        let inp = &input[ch * hw..(ch + 1) * hw];
        for (p, o) in row.iter_mut().enumerate() {
            let (mean, scale) = stats[p];
            *o = (inp[p] - mean) * scale * ww + bb;
        }
    });
}

/// Per-channel spatial mean: `out[c] = mean_p in[c][p]`.
///
/// Delegates to `channel_mean_block` at `CHANNEL_MEAN_BLOCK`, which is the
/// toolkit kernel's own block size - the two must reduce in the same order, and
/// that order is a function of the block.
pub fn channel_mean(input: &[f32], c: usize, hw: usize, out: &mut [f32]) {
    channel_mean_block(input, c, hw, out, CHANNEL_MEAN_BLOCK)
}

/// The same reduction, at the block size a launch used. `lg_channel_mean`
/// strides `hw` into `block` per-thread partials and then halves a tree, so its
/// sum order is a function of the block size, and the twin takes the block as an
/// argument so that the `--cuda-selftest` row measures the PORT rather than the
/// summation order. It is the same choice maxim's CPU twin makes.
pub fn channel_mean_block(input: &[f32], _c: usize, hw: usize, out: &mut [f32], block: usize) {
    let _p = prof_start("channel_mean");
    let n = hw as f32;
    // THE PARTIALS ARE FILLED IN CHUNKS OF `block`, which is the same assignment
    // as `slots[i % block] += p[i]` - element `i` lands in slot `i % block`
    // either way - but without the integer division per element. The toolkit's
    // kernel strides `p[threadIdx.x + k * blockDim.x]` into `slot[threadIdx.x]`,
    // so this is also literally the same access pattern.
    let partial = |p: &[f32], slots: &mut Vec<f32>| {
        if slots.len() < block {
            slots.resize(block, 0.0);
        }
        slots[..block].fill(0.0);
        for chunk in p.chunks(block) {
            for (slot, v) in slots.iter_mut().zip(chunk) {
                *slot += v;
            }
        }
        let mut s = block;
        while s > 1 {
            s >>= 1;
            for t in 0..s {
                slots[t] += slots[t + s];
            }
        }
        slots[0] / n
    };
    // `for_each_init` gives each rayon worker one reusable slot vector instead of
    // one per channel - 512 allocations of 1 KB per invocation, twice per block.
    out.par_iter_mut().enumerate().for_each_init(
        Vec::new,
        |slots, (ch, o)| {
            *o = partial(&input[ch * hw..(ch + 1) * hw], slots);
        },
    );
}

/// `y = a * b`, elementwise over the whole plane.
pub fn mul(a: &[f32], b: &[f32], y: &mut [f32]) {
    y.par_iter_mut()
        .zip(a.par_iter().zip(b.par_iter()))
        .for_each(|(y, (a, b))| *y = a * b);
}

/// `y = x * s[c]`, per channel of an NCHW tensor.
pub fn channel_scale(x: &[f32], s: &[f32], _c: usize, hw: usize, y: &mut [f32]) {
    y.par_chunks_mut(hw).enumerate().for_each(|(ch, o)| {
        let k = s[ch];
        for (o, x) in o.iter_mut().zip(&x[ch * hw..(ch + 1) * hw]) {
            *o = x * k;
        }
    });
}

/// `y = a + b`, elementwise.
/// DEVELOPMENT ONLY, and GPU-SELFTEST ONLY: the twin for `lg_add`. The forward
/// pass does not use it - the residual is `nf_residual` and the decoder's skip
/// add is inlined into the block loops - so the only caller is `--cuda-selftest`.
#[cfg(all(feature = "dev", feature = "cuda"))]
pub fn add(a: &[f32], b: &[f32], y: &mut [f32]) {
    y.par_iter_mut()
        .zip(a.par_iter().zip(b.par_iter()))
        .for_each(|(y, (a, b))| *y = a + b);
}

/// A reusable set of activation-sized buffers for the CPU forward pass.
///
/// WHY THIS EXISTS: the CPU backend is a FALLBACK ENGINE, not a reference - the
/// reference is `tools/reference.py` - so it is held to the same standard as the
/// GPU path, and that includes not reallocating half a gigabyte per block. Done
/// naively each block allocates thirteen fresh planes (`t1 t2 t3 g gs t4 y f1 f2
/// fg f3 pooled att`), several of them `2c x hw`; at the widest block that is
/// ~600 MB per invocation and ~2 GB of peak RSS over a 1280x725 image, all of it
/// written before it is read. Those shapes repeat across the blocks of a level,
/// so the buffers are allocated once and handed out by index.
///
/// Slots are also SHARED BY ROLE where the live ranges cannot overlap; see the
/// `S_*` constants for which roles share a slot and why that is safe. Sharing is
/// what makes the peak proportional to the live set rather than to the number of
/// mathematical intermediates.
pub struct Scratch {
    /// Capacity only. The LENGTH an op sees is an argument to each accessor,
    /// never a property of the slot: one slot serves roles of different widths
    /// (a buffer holding a `2c` expansion is later reused for a `c`-channel
    /// result), so a stored per-slot length would hand the wider role's extent to
    /// the narrower one. That is not a hypothetical - a `conv1x1` iterates
    /// `hw`-sized chunks of its output, so a double-length slot made it walk
    /// twice as many output channels and run off the end of its weight matrix.
    bufs: Vec<Vec<f32>>,
}

impl Default for Scratch {
    fn default() -> Scratch {
        Scratch::new()
    }
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch { bufs: Vec::new() }
    }

    /// Size every slot for this call, growing only what is too small.
    ///
    /// Called once at the top of `block_forward`, deliberately BEFORE any slot is
    /// borrowed: with growth done up front, the accessors below need no `&mut self`
    /// for growth and an input and an output slot can be borrowed in the same
    /// statement instead of through a chain of temporaries. Growth only ever adds
    /// CAPACITY; the length an op sees comes from its own arguments, because one
    /// slot serves roles of different widths (a slot holding a `2c` expansion is
    /// later reused for a `c`-channel result).
    fn reserve(&mut self, sizes: &[(usize, usize)]) {
        for &(i, n) in sizes {
            while self.bufs.len() <= i {
                self.bufs.push(Vec::new());
            }
            let b = &mut self.bufs[i];
            if b.len() < n {
                b.resize(n, 0.0);
            }
        }
    }

    /// The first `n` floats of slot `i`, read-only.
    fn get(&self, i: usize, n: usize) -> &[f32] {
        debug_assert!(n <= self.bufs[i].len(), "slot {i} reserved too small");
        &self.bufs[i][..n]
    }

    /// The first `n` floats of slot `i`, written. Every producer writes all `n`
    /// elements it is asked for, so no clearing is needed here and none is done -
    /// that is a full pass of memory traffic per slot saved.
    fn get_mut(&mut self, i: usize, n: usize) -> &mut [f32] {
        debug_assert!(n <= self.bufs[i].len(), "slot {i} reserved too small");
        &mut self.bufs[i][..n]
    }

    /// One input slot and one output slot, borrowed together, each cut to the
    /// length its op wants. Splitting them here is what keeps the slots
    /// independent without an intermediate copy.
    fn pair(&mut self, r: usize, nr: usize, w: usize, nw: usize) -> (&[f32], &mut [f32]) {
        assert_ne!(r, w, "an op cannot read and write the same slot");
        debug_assert!(nr <= self.bufs[r].len() && nw <= self.bufs[w].len());
        let (lo, hi) = if r < w { (r, w) } else { (w, r) };
        let (lo_s, hi_s) = self.bufs.split_at_mut(hi);
        if r < w {
            (&lo_s[lo][..nr], &mut hi_s[0][..nw])
        } else {
            (&hi_s[0][..nr], &mut lo_s[lo][..nw])
        }
    }
}

// SLOT INDICES, SHARED BY ROLE. Each role's live range is written before the next
// role's write and after the previous role's last read, so one buffer serves
// several intermediates; the numbers in brackets are the steps in
// `block_forward` that make the hand-off safe.
const S_A: usize = 0; // `t1` norm+expand (1-2), then `f1` the FFN norm (10-11)
const S_B: usize = 1; // `t2` the 1x1 expansion (2-3), then `y` (9-10)
const S_C: usize = 2; // `t3` depthwise output (3-4), then `f2` FFN expansion (11-12)
const S_D: usize = 3; // `g` gate product (4-7), then `t4` (8-9), then `f3` (13-14)
const S_E: usize = 4; // `gs` the sca-scaled gate (7-8), then `fg` (12-13)
const S_POOL: usize = 5;
const S_ATT: usize = 6;
/// The decoder's transient pre-shuffle plane, used outside `block_forward`.
const S_UP_TMP: usize = 7;
/// One NAFBlock on the CPU; `a` is the input and the result is written back into
/// `a`. `scratch` holds the planes, reused across every block - see `Scratch`.
fn block_forward(blk: &Block, a: &mut Act, scratch: &mut Scratch) {
    let (c, h, wd) = (a.c, a.h, a.w);
    let hw = h * wd;
    let dw = c * 2;
    let half = dw / 2;

    // One reservation for the whole block, sized to the WIDEST role each slot
    // serves, so every borrow below is of an already-sized buffer.
    scratch.reserve(&[
        (S_A, c * hw),
        (S_B, dw * hw),
        (S_C, dw * hw),
        (S_D, c * hw),
        (S_E, half * hw),
        (S_POOL, half),
        (S_ATT, half),
    ]);

    // 1-2: norm, then the 1x1 expansion into `2c`.
    {
        let t1 = scratch.get_mut(S_A, c * hw);
        channel_layer_norm(&a.data, blk.n1w, blk.n1b, c, hw, t1);
    }
    {
        let (t1, t2) = scratch.pair(S_A, c * hw, S_B, dw * hw);
        conv1x1(t1, blk.conv1_w, blk.conv1_b, c, dw, hw, t2);
    }
    // 3-4: depthwise 3x3, then SimpleGate (split the channel axis, multiply).
    {
        let (t2, t3) = scratch.pair(S_B, dw * hw, S_C, dw * hw);
        conv3x3_dw(t2, blk.conv2_w, blk.conv2_b, dw, h, wd, t3);
    }
    {
        let (t3, g) = scratch.pair(S_C, dw * hw, S_D, half * hw);
        let (g0, g1) = t3.split_at(half * hw);
        mul(g0, g1, g);
    }
    // 5-7: sca - global average pool, a 1x1 conv over the SAME channels, then a
    // per-channel scale of the gate product.
    {
        let (g, pooled) = scratch.pair(S_D, half * hw, S_POOL, half);
        channel_mean(g, half, hw, pooled);
    }
    {
        let (pooled, att) = scratch.pair(S_POOL, half, S_ATT, half);
        conv1x1(pooled, blk.sca_w, blk.sca_b, half, half, 1, att);
    }
    {
        // `att` is one value per channel: copying it costs `half` floats, which
        // is cheaper than the three-way borrow that reading it alongside `g` and
        // `gs` would otherwise need.
        let att = scratch.get(S_ATT, half).to_vec();
        let (g, gs) = scratch.pair(S_D, half * hw, S_E, half * hw);
        channel_scale(g, &att, half, hw, gs);
    }
    // 8-9: back to `c` channels, then `y = inp + t4 * beta`. `S_D` is free again
    // once the scale above has read `g`, and `S_B` once the depthwise conv has
    // read `t2`.
    {
        let (gs, t4) = scratch.pair(S_E, half * hw, S_D, c * hw);
        conv1x1(gs, blk.conv3_w, blk.conv3_b, half, c, hw, t4);
    }
    {
        let (t4, y) = scratch.pair(S_D, c * hw, S_B, c * hw);
        let src = &a.data;
        y.par_chunks_mut(hw).enumerate().for_each(|(ch, o)| {
            let s = src.chunk(ch, hw);
            for (o, (s, t)) in o
                .iter_mut()
                .zip(s.iter().zip(&t4[ch * hw..(ch + 1) * hw]))
            {
                *o = s + t * blk.beta[ch];
            }
        });
    }

    // 10-13: the FFN - norm, 1x1 to 2c, SimpleGate, 1x1 back to c.
    {
        let (y, f1) = scratch.pair(S_B, c * hw, S_A, c * hw);
        channel_layer_norm(y, blk.n2w, blk.n2b, c, hw, f1);
    }
    {
        let (f1, f2) = scratch.pair(S_A, c * hw, S_C, dw * hw);
        conv1x1(f1, blk.conv4_w, blk.conv4_b, c, dw, hw, f2);
    }
    {
        let (f2, fg) = scratch.pair(S_C, dw * hw, S_E, half * hw);
        let (h0, h1) = f2.split_at(half * hw);
        mul(h0, h1, fg);
    }
    {
        let (fg, f3) = scratch.pair(S_E, half * hw, S_D, c * hw);
        conv1x1(fg, blk.conv5_w, blk.conv5_b, half, c, hw, f3);
    }

    // 14: two read-only slots feeding a write into `a.data` - two shared borrows
    // of `scratch` cannot conflict, and `a.data` is not in `scratch` at all.
    {
        let yd = scratch.get(S_B, c * hw);
        let f3 = scratch.get(S_D, c * hw);
        let dst = &mut a.data;
        dst.par_chunks_mut(hw).enumerate().for_each(|(ch, o)| {
            let s = yd.chunk(ch, hw);
            for (o, (s, t)) in o.iter_mut().zip(s.iter().zip(&f3[ch * hw..(ch + 1) * hw])) {
                *o = s + t * blk.gamma[ch];
            }
        });
    }
}

/// The whole network, minus the padding and the crop: `input` is [3][h][w].
///
/// The returned activation is [3][h][w]. This is the CPU backend - a FALLBACK
/// ENGINE for a machine with no usable GPU, not the correctness record. The
/// reference is `tools/reference.py`, and it is the only thing that can say
/// whether the network itself is right; this backend and the GPU agree on the
/// paper's reading of NAFNet, so agreeing with each other proves nothing about
/// either. It is also the twin `--cuda-selftest` compares each kernel against,
/// which is a statement about the two implementations of one op, not about
/// correctness - the same distinction, one level down.
pub fn forward_cpu(w: &Weights, input: &[f32], h: usize, wd: usize) -> Result<Act, String> {
    forward_cpu_dump(w, input, h, wd, &mut |_, _| {})
}

/// The CPU pass, writing every intermediate to `path` when one is given.
///
/// DEVELOPMENT ONLY: the CLI's `--dump` is the only caller.
#[cfg(feature = "dev")]
pub fn forward_cpu_maybe_dump(
    w: &Weights,
    input: &[f32],
    h: usize,
    wd: usize,
    path: Option<&str>,
) -> Result<Act, String> {
    match path {
        None => forward_cpu(w, input, h, wd),
        Some(p) => {
            let mut index: Vec<String> = Vec::new();
            let mut blob: Vec<f32> = Vec::new();
            let out = forward_cpu_dump(w, input, h, wd, &mut |name: &str, a: &Act| {
                let off = blob.len();
                blob.extend_from_slice(&a.data);
                index.push(format!("{name} {} {} {} {}", a.c, a.h, a.w, off));
            })?;
            let mut bytes = Vec::with_capacity(blob.len() * 4);
            for v in &blob {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let text = index.join("\n") + "\n";
            std::fs::write(format!("{p}.txt"), text).map_err(|e| e.to_string())?;
            std::fs::write(p, bytes).map_err(|e| e.to_string())?;
            Ok(out)
        }
    }
}

/// The same pass, reporting every intermediate through `dump`.
///
/// This exists so a divergence against tools/reference.py can be located by
/// stage rather than by bisection on the output: the two implementations are
/// handed the same input and their 47 tensors are compared name by name, which
/// turns "the image is wrong" into "encoders.0.0 is already wrong".
pub fn forward_cpu_dump(
    w: &Weights,
    input: &[f32],
    h: usize,
    wd: usize,
    dump: &mut dyn FnMut(&str, &Act),
) -> Result<Act, String> {
    // THE PROFILE IS ARMED AND REPORTED HERE, once per pass, so no caller has to
    // know it exists: every CPU forward goes through this function (the no-dump
    // entry point is the same function with a no-op closure), and the closure
    // that decides what to do with each stage cannot be the thing that decides
    // whether the ops are timed.
    if std::env::var_os("NAFNET_CPU_PROFILE").is_some() {
        cpu_prof_enable();
    }
    let cfg = &w.config;
    let levels = cfg.levels();
    let mut a = Act::new(cfg.width, h, wd);
    // The input plane itself is reported as "in", so a divergence at stage one
    // can be told apart from a divergence caused by the two sides being handed
    // DIFFERENT inputs (the reference dumps its `in` the same way).
    dump("in", &Act { c: 3, h, w: wd, data: input.to_vec() });
    conv3x3(input, w.get("intro.weight")?, w.get("intro.bias")?, 3, cfg.width, h, wd, &mut a.data);
    dump("intro", &a);

    let mut skips: Vec<Act> = Vec::with_capacity(levels);
    let mut cur = a;
    let mut scratch = Scratch::new();

    for l in 0..levels {
        let c = cfg.width_at(l);
        for b in 0..cfg.enc_blk_nums[l] {
            let name = format!("encoders.{l}.{b}");
            let blk = Block::load(w, &name)?;
            block_forward(&blk, &mut cur, &mut scratch);
            dump(&name, &cur);
        }
        // Downsample: stride-2 2x2 conv, channels doubled and the plane HALVED.
        // `h` and `wd` are the INPUT size here - the output activation is
        // `cur.h / 2 x cur.w / 2`, and taking the shape from `cur` rather than
        // from the caller is what keeps the two agree-ing through the decoder's
        // pixel_shuffle2, which doubles whichever size it is handed.
        let mut down = Act::new(2 * c, cur.h / 2, cur.w / 2);
        conv2x2s2(
            &cur.data,
            w.get(&format!("downs.{l}.weight"))?,
            w.get(&format!("downs.{l}.bias"))?,
            c,
            2 * c,
            cur.h,
            cur.w,
            &mut down.data,
        );
        dump(&format!("downs.{l}"), &down);
        // The skip TAKES `cur`'s buffer. Cloning it first - which the order of
        // this loop used to force, because the downsample conv above still needs
        // `cur.data` - doubled the widest encoder plane (120 MB at 1280x725
        // level 0) for the length of one conv, and it was pure waste.
        skips.push(Act {
            c: cur.c,
            h: cur.h,
            w: cur.w,
            data: std::mem::take(&mut cur.data),
        });
        cur = down;
    }

    for b in 0..cfg.middle_blk_num {
        let name = format!("middle_blks.{b}");
        let blk = Block::load(w, &name)?;
        block_forward(&blk, &mut cur, &mut scratch);
        dump(&name, &cur);
    }

    for l in 0..levels {
        let c_before = cur.c;
        let c_after = c_before / 2;
        // 1x1 conv to 4x the channels (bias-free), then depth-to-space.
        // `up_tmp` is TRANSIENT - only `pixel_shuffle2` reads it - so it lives in
        // a scratch slot rather than in a fresh plane of its own. `up` becomes
        // `cur`, so it keeps its own buffer.
        let up_tmp_n = c_before * 2 * cur.hw();
        scratch.reserve(&[(S_UP_TMP, up_tmp_n)]);
        {
            let up_tmp = scratch.get_mut(S_UP_TMP, up_tmp_n);
            conv1x1(
                &cur.data,
                w.get(&format!("ups.{l}.0.weight"))?,
                &[],
                c_before,
                c_before * 2,
                cur.hw(),
                up_tmp,
            );
        }
        let mut up = Act::new(c_after, cur.h * 2, cur.w * 2);
        {
            let up_tmp = scratch.get(S_UP_TMP, up_tmp_n);
            pixel_shuffle2(up_tmp, c_after, cur.h, cur.w, &mut up.data);
        }
        // `ups.{l}` IS THE SHUFFLE OUTPUT: dumped here, before the skip add, to
        // be the same stage `tools/reference.py` dumps as `ups.{i}` and the same
        // one `Gpu::forward_dump` snaps. Dumping after the add instead compares a
        // post-skip tensor against a pre-skip one.
        dump(&format!("ups.{l}"), &up);
        // Add the encoder skip IN PLACE. `add` computed `a + b` elementwise into
        // a third buffer and then dropped the first; this is the same arithmetic
        // per element with no third buffer (134 MB at the widest level).
        let skip = skips.pop().expect("one skip per level");
        up.data
            .par_iter_mut()
            .zip(&skip.data)
            .for_each(|(o, s)| *o += s);
        cur = up;
        for b in 0..cfg.dec_blk_nums[l] {
            let name = format!("decoders.{l}.{b}");
            let blk = Block::load(w, &name)?;
            block_forward(&blk, &mut cur, &mut scratch);
            dump(&name, &cur);
        }
    }

    let mut out = Act::new(3, h, wd);
    // (the ending conv is not a separate dump name: it is reported as "out")
    conv3x3(
        &cur.data,
        w.get("ending.weight")?,
        w.get("ending.bias")?,
        cfg.width,
        3,
        h,
        wd,
        &mut out.data,
    );
    out.data.par_iter_mut().zip(input).for_each(|(o, i)| *o += i);
    dump("out", &out);
    cpu_prof_report();
    Ok(out)
}

/// Tiny helper so the channel-wise loops above read like the reference.
trait ChunkAt {
    fn chunk(&self, c: usize, hw: usize) -> &[f32];
}

impl ChunkAt for [f32] {
    fn chunk(&self, c: usize, hw: usize) -> &[f32] {
        &self[c * hw..(c + 1) * hw]
    }
}
