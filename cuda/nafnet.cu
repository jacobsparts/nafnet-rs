//! NAFNet's own kernels: the two ops the lightgpu toolkit does not have.
//!
//! Everything else this engine runs is a toolkit kernel (`lg_conv3x3s1p1`,
//! `lg_conv1x1`, `lg_channel_layer_norm`, `lg_channel_mean`, `lg_mul`,
//! `lg_channel_scale`, `lg_add`, `lg_add_scaled`). These two are here because
//! the family had no grouped/depthwise convolution anywhere and no
//! depth-to-space PixelShuffle - the toolkit's `lg_pixel_unshuffle2` is the
//! OPPOSITE direction - so neither is a duplicate of anything shared. They move
//! into the toolkit the moment a second family needs them (docs/MAINTAINING.md).
//!
//! Both follow cuda/CONVENTIONS.md: `extern "C" __global__`, the engine's prefix,
//! raw pointers and scalars only, `const T *__restrict__` in / `T *__restrict__`
//! out, no caller-supplied dynamic shared memory, and `long` for a count a
//! batched caller could overflow.

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// nf_conv3x3_dw - 3x3 depthwise convolution, stride 1, pad 1.
//
//   out[c][y][x] = bias[c] + sum_{ky,kx} w[c][ky][kx] * in[c][y+ky-1][x+kx-1]
//
// ONE CHANNEL PER BLOCK, one output pixel per thread. `groups == c` in the
// reference (`nn.Conv2d(dw_channel, dw_channel, 3, groups=dw_channel)`), so a
// channel owns exactly its own 9 weights and there is no cross-channel sum: the
// whole computation is nine multiply-adds per output and the only thing that
// matters is memory traffic. A row of the output needs three rows of the input,
// and the middle one is read three times by the three ky taps unless it is
// staged, so the kernel stages the (BLOCK_H + 2) x (wd + 2) input patch for the
// channel's row band in shared memory.
//
// `w` is [c][9] in (ky, kx) order - the reference stores [c][1][3][3], which is
// the same nine values with the same order, so the engine uploads it unchanged.
//
// The tile is 32 output columns x 8 rows per 256-thread block, and the halo
// makes the staged patch 34 x 10 floats - small enough that the shared-memory
// cost is a compile-time constant and no caller passes a dynamic amount.
//
// wd is the plane width; the last tile's columns past wd are simply not written,
// and the staged halo is padded with zeroes, so no width precondition beyond
// wd >= 1 is needed (contrast the tile-staged 3x3 kernels, which require
// wd % 64 == 0).
// ---------------------------------------------------------------------------

#define NF_DW_TX 32
#define NF_DW_TY 8

extern "C" __global__ void nf_conv3x3_dw(
    const float *__restrict__ in,
    const float *__restrict__ w,
    const float *__restrict__ bias,
    float *__restrict__ out,
    int c,
    int h,
    int wd)
{
    // The staged patch holds the halo columns too, so an unaligned width needs a
    // row stride that is a runtime value... except that it is not: the patch is
    // indexed in this thread's own registers and the shared array is sized for
    // the FULL tile plus halo, exactly 34 x 10.
    __shared__ float s[3][NF_DW_TY + 2][NF_DW_TX + 2];

    const int ch = blockIdx.z;
    if (ch >= c) return;

    const int y0 = blockIdx.y * NF_DW_TY;
    const int x0 = blockIdx.x * NF_DW_TX;

    const float *ip = in + ((long)ch * h) * wd;
    float *op = out + ((long)ch * h) * wd;

    // Stage the (TY + 2) x (TX + 2) patch, three rows of it per pass, so the
    // whole tile costs (10 * 34) / 256 = 1.3 loads per thread instead of the
    // nine a naive gather would issue.
    for (int i = threadIdx.y * NF_DW_TX + threadIdx.x; i < 3 * (NF_DW_TY + 2) * (NF_DW_TX + 2);
         i += NF_DW_TY * NF_DW_TX) {
        const int r = i / ((NF_DW_TY + 2) * (NF_DW_TX + 2));
        const int rem = i % ((NF_DW_TY + 2) * (NF_DW_TX + 2));
        const int yy = rem / (NF_DW_TX + 2);
        const int xx = rem % (NF_DW_TX + 2);
        const int gy = y0 + yy - 1;
        const int gx = x0 + xx - 1;
        s[r][yy][xx] = (gy >= 0 && gy < h && gx >= 0 && gx < wd) ? ip[(long)gy * wd + gx] : 0.0f;
    }
    __syncthreads();

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int oy = y0 + ty;
    const int ox = x0 + tx;
    if (oy >= h || ox >= wd) return;

    const float *wk = w + ch * 9;
    float acc = bias[ch];
    // Accumulation order matches the CPU twin: ky, then kx.
    #pragma unroll
    for (int ky = 0; ky < 3; ky++) {
        #pragma unroll
        for (int kx = 0; kx < 3; kx++) {
            acc += wk[ky * 3 + kx] * s[ky][ty + ky][tx + kx];
        }
    }
    op[(long)oy * wd + ox] = acc;
}

// ---------------------------------------------------------------------------
// nf_pixel_shuffle2 - depth-to-space, the inverse of the toolkit's
// lg_pixel_unshuffle2.
//
//   in  [C * 4][h][wd]  ->  out [C][2*h][2*wd]
//   out[c][2*y + dy][2*x + dx] = in[c * 4 + dy * 2 + dx][y][x]
//
// THE PERMUTATION IS THE CONTRACT, not just the shape: the 1x1 conv that
// precedes this op orders its output channels by it, so a different tap order
// produces a plausible image from the wrong weights. This matches PyTorch's
// `nn.PixelShuffle(2)`, whose channel order is `c * r^2 + dy * r + dx` for
// output (c, y*r + dy, x*r + dx).
//
// One output pixel per thread, block 32x8, grid (ceil(2*wd/32), ceil(2*h/8),
// C). The gather is a single load per output element: the four source values
// that land in a 2x2 output neighbourhood are four different channels, and
// staging them would cost more than the load.
// ---------------------------------------------------------------------------

extern "C" __global__ void nf_pixel_shuffle2(
    const float *__restrict__ in,
    float *__restrict__ out,
    int c,
    int h,
    int wd)
{
    const int C = blockIdx.z;
    const int oy = blockIdx.y * NF_DW_TY + threadIdx.y;
    const int ox = blockIdx.x * NF_DW_TX + threadIdx.x;
    const int oh = h * 2;
    const int ow = wd * 2;
    if (C >= c || oy >= oh || ox >= ow) return;

    const int y = oy >> 1;
    const int x = ox >> 1;
    const int dy = oy & 1;
    const int dx = ox & 1;

    const long src = ((long)(C * 4 + dy * 2 + dx) * h + y) * wd + x;
    out[((long)C * oh + oy) * ow + ox] = in[src];
}

// ---------------------------------------------------------------------------
// nf_down2x2s2 - stride-2 2x2 convolution, no padding. The encoder's downsample
// (`nn.Conv2d(c, 2*c, 2, stride=2)`), and one of the three ops the family had no
// kernel for.
//
//   out[oc][y][x] = bias[oc] + sum_ic sum_ky,kx w[oc][ic][ky][kx] * in[ic][2y+ky][2x+kx]
//
// IT CANNOT BE REACHED THROUGH lg_conv3x3s1p1. A zero-padded 3x3 looks like the
// obvious stand-in for a 2x2 - put the four taps at (ky,kx) = (1,1),(1,2),(2,1),
// (2,2) - but the kernel's READ offsets are fixed at stride 1, so what it would
// compute is `sum w[ky][kx] * in[y+ky-1][x+kx-1]`, and matching
// `in[2y+ky'][2x+kx']` requires the INPUT to be nearest-upsampled first. That
// costs a full extra plane per downsample and an `lg_upsample2x_nearest` launch
// on the ENCODER side, where the toolkit's own conv family is documented as
// losing. A 2x2 stride-2 gather is nine multiplies cheaper than the padded 3x3
// and reads a quarter of the input, so it stays a kernel of its own.
//
// The output is h/2 x wd/2; h and wd are even by construction (the padder makes
// the input a multiple of 2**levels), so no tail case exists. The guard below
// is still written out rather than assumed.
//
// TILED OVER OUTPUT CHANNELS, FOR THE SAME REASON AS nf_conv1x1_oc. One thread
// per output element means every thread walks all `c_in` input planes, and the
// whole input is walked AGAIN for each of the `c_out` output channels. Here the
// arithmetic says exactly how much that costs, and unlike the 1x1 conv the
// prediction HELD: at 1280x736 the four downsample launches read
// c_out x input x 4 levels = 4 x 7.7 GB = 31 GB, which at this device's
// ~320 GB/s is ~96 ms - and the measured cost was 98 ms. So this kernel really
// was 100% memory stall, and a thread that now holds NF_DW_TILE accumulators
// for one output position reads the input `ceil(c_out / NF_DW_TILE)` times
// instead of `c_out` times.
//
// THE FOUR-PRODUCT EXPRESSION IS UNCHANGED, and it deliberately stays ONE
// expression rather than becoming four `acc +=` statements. nvcc groups it into
// a sub-tree of its own, which is why `nf_down2x2s2` disagrees with its CPU twin
// by 2 ulp where every other op agrees exactly - writing it as separate adds
// would change the rounding and lose that known quantity for no benefit.
// ---------------------------------------------------------------------------

// Output channels per thread here. MUST match DW_OC_TILE in src/gpu.rs.
#define NF_DW_TILE 16

extern "C" __global__ void nf_down2x2s2(
    const float *__restrict__ in,
    const float *__restrict__ w,
    const float *__restrict__ bias,
    float *__restrict__ out,
    int c_in,
    int c_out,
    int h,
    int wd)
{
    const int oh = h / 2;
    const int ow = wd / 2;
    const long plane = (long)oh * ow;
    // ONE THREAD PER OUTPUT POSITION, not per output element: the grid's y axis
    // carries the output channels now.
    const long pos = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= plane) return;

    const int x = (int)(pos % ow);
    const int y = (int)(pos / ow);
    const int oc0 = blockIdx.y * NF_DW_TILE;

    float acc[NF_DW_TILE];
    int oc[NF_DW_TILE];
    #pragma unroll
    for (int t = 0; t < NF_DW_TILE; t++) {
        // Clamped for the READ path only; the store re-checks the real bound.
        oc[t] = (oc0 + t < c_out) ? (oc0 + t) : (c_out - 1);
        acc[t] = bias ? bias[oc[t]] : 0.0f;
    }

    const size_t in_plane = (size_t)h * wd;
    const size_t top = (size_t)(2 * y) * wd + 2 * x;

    for (int ic = 0; ic < c_in; ++ic) {
        // The four taps, loaded ONCE for all NF_DW_TILE output channels.
        const float *ip = in + (size_t)ic * in_plane + top;
        const float x0 = ip[0], x1 = ip[1], x2 = ip[wd], x3 = ip[wd + 1];
        // Order: ic, then ky, then kx - the same order the CPU twin accumulates
        // in, and the same four-product grouping as the untiled kernel.
        #pragma unroll
        for (int t = 0; t < NF_DW_TILE; t++) {
            const float *wk = w + (size_t)oc[t] * c_in * 4 + (size_t)ic * 4;
            acc[t] += wk[0] * x0 + wk[1] * x1 + wk[2] * x2 + wk[3] * x3;
        }
    }

    #pragma unroll
    for (int t = 0; t < NF_DW_TILE; t++) {
        if (oc0 + t < c_out) {
            out[(long)oc[t] * plane + pos] = acc[t];
        }
    }
}

// ---------------------------------------------------------------------------
// nf_conv1x1_oc - the same 1x1 convolution as the toolkit's lg_conv1x1, with the
// work blocked over OUTPUT CHANNELS.
//
// WHY THIS EXISTS, IN ONE MEASUREMENT: on the 1280x736 width-32 pass,
// lg_conv1x1 was 4000 ms of a 5100 ms forward pass - 78% of it - across 184
// launches. The kernel gives one thread one output element, so a thread walks
// all `c_in` inputs at stride `plane`, and the whole input is walked AGAIN for
// every one of the `c_out` output channels: `c_out` passes over the input for
// one pass over the output.
//
// THE FIX IS REUSE. A thread holds NF_OC_TILE accumulators for one spatial
// position, so a single load of `in[ic][pos]` feeds NF_OC_TILE fused
// multiply-adds instead of one, and the input is read
// `ceil(c_out / NF_OC_TILE)` times rather than `c_out` times. NF_OC_SP then
// covers that many CONSECUTIVE positions per thread, which divides both the
// input loads and the weight fetches by another NF_OC_SP.
//
// THE PARAMETERS ARE MEASURED, NOT REASONED. Pass wall at 1280x725, best of 3,
// with the other parameter held at its final value:
//
//   NF_OC_TILE:  4 -> 1.41 s   8 -> 1.26 s   16 -> 1.12 s   32 -> 1.54 s
//   NF_OC_SP:    2 -> 0.95 s   4 -> 0.80 s    8 -> 0.78 s   16 -> 3.54 s
//
// `ptxas -Xptxas -v` explains both turn-overs, and neither is what the obvious
// model predicts: registers are 96/128/214/255+spill for SP = 2/4/8/16, so
// SP=16 spills to local memory and falls off a cliff, while SP=8 runs at HALF
// the resident threads of SP=4 (214 registers allow one 256-thread block) and is
// marginally FASTER - occupancy is not the limiter. Four models were tried
// against the residual cost and all four failed: input re-read volume (4x less
// re-reading bought 1.5x), DRAM bandwidth (~13.5 GB is 42 ms of a 431 ms
// kernel), FMA throughput (77e9 FMAs is 17 ms of 431 ms at peak) and occupancy.
// What is left - instruction issue, LSU/L1 throughput, TLB over 256
// simultaneously-strided planes - needs hardware counters, and Nsight Compute
// does not support Pascal (this is a GP104), so it cannot be settled here.
//
// NF_OC_SP IS 4 RATHER THAN 8 DESPITE BEING MARGINALLY SLOWER, for portability:
// 128 registers instead of 214 keeps two blocks resident per SM on every
// architecture this fatbin targets (sm_61/75/80), and the difference is 2.5% of
// the pass.
//
// THE ACCUMULATION ORDER IS UNCHANGED, AND THAT IS DELIBERATE: each output
// element still starts at its bias and folds in one fused multiply-add per `ic`
// in ascending order, exactly as lg_conv1x1 does and exactly as the CPU twin
// does. Only the LOOP STRUCTURE around that sum moved, never the sum itself. So
// this kernel is BIT-IDENTICAL to the one it replaced, and the 46-stage dump is
// how that is checked rather than assumed - it was byte-compared against the
// pre-change dump at every step of the tuning above.
//
// `bias` may be null (the bias-free upsample conv), matching lg_conv1x1's
// contract. Neither need `c_out` be a multiple of NF_OC_TILE nor `hw` a multiple
// of NF_OC_SP: the final tile clamps its oc to a valid row for the LOADS, and
// both the trailing positions and the trailing channels are guarded on the
// STORES only, so no read runs past the weight, bias or input arrays. The launch
// grid is rounded UP, so the last block may hold threads whose every store is
// guarded off - which is why no thread needs an early return.
//
// grid (ceil(hw / (blockDim.x * NF_OC_SP)), ceil(c_out / NF_OC_TILE), 1).
// ---------------------------------------------------------------------------

#define NF_OC_TILE 16
// Positions per thread along x: a thread covers NF_OC_SP CONSECUTIVE columns.
// See the measured table and the register counts above for both constants.
#define NF_OC_SP 4

extern "C" __global__ void nf_conv1x1_oc(
    const float *__restrict__ in,
    const float *__restrict__ w,
    const float *__restrict__ bias,
    float *__restrict__ out,
    int c_in,
    int c_out,
    int h,
    int wd)
{
    const long hw = (long)h * wd;
    const long base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * NF_OC_SP;

    const int oc0 = blockIdx.y * NF_OC_TILE;

    float acc[NF_OC_TILE][NF_OC_SP];
    int oc[NF_OC_TILE];
    #pragma unroll
    for (int t = 0; t < NF_OC_TILE; t++) {
        // Clamp for the READ path only; the store below re-checks the real bound.
        oc[t] = (oc0 + t < c_out) ? (oc0 + t) : (c_out - 1);
        #pragma unroll
        for (int s = 0; s < NF_OC_SP; s++) {
            acc[t][s] = bias ? bias[oc[t]] : 0.0f;
        }
    }

    // The `ic` loop is the only one that touches memory per iteration, and each
    // iteration now pays for NF_OC_TILE * NF_OC_SP fused multiply-adds.
    // THE ORDER PER OUTPUT ELEMENT IS UNCHANGED - bias, then one fused
    // multiply-add per `ic` ascending - because only the LOOP STRUCTURE around
    // that sum moved, not the sum itself.
    for (int ic = 0; ic < c_in; ic++) {
        const float *xp = in + (long)ic * hw + base;
        float xv[NF_OC_SP];
        #pragma unroll
        for (int s = 0; s < NF_OC_SP; s++) {
            xv[s] = (base + s < hw) ? xp[s] : 0.0f;
        }
        #pragma unroll
        for (int t = 0; t < NF_OC_TILE; t++) {
            const float wv = w[(long)oc[t] * c_in + ic];
            #pragma unroll
            for (int s = 0; s < NF_OC_SP; s++) {
                acc[t][s] += wv * xv[s];
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < NF_OC_TILE; t++) {
        if (oc0 + t < c_out) {
            #pragma unroll
            for (int s = 0; s < NF_OC_SP; s++) {
                if (base + s < hw) {
                    out[(long)oc[t] * hw + base + s] = acc[t][s];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// nf_residual - `y = a + b * s[ch]`, the per-channel residual that ends both
// halves of every NAFBlock.
//
// WHY IT IS NOT `lg_add_scaled`: that toolkit kernel takes a WHOLE-PLANE scalar
// `s`, and this op's scale is per CHANNEL (`beta` and `gamma` are [c] vectors).
// gpu.rs's header records that as the reason the block fell back to two kernels,
// `lg_channel_scale` then `lg_add`.
//
// WHAT FUSING BUYS: two launches become one, and 2 planes read + 1 written plus
// 2 read + 1 written becomes 3 read + 1 written - so per residual site the
// traffic drops from 270 MB to 180 MB at 1280x725, and the pass loses 72
// launches.
//
// THE ROUNDING IS THE TRAP AND IT IS WHY THE TWO INTRINSICS ARE HERE. The pair
// it replaces rounds TWICE: `lg_channel_scale` computes `out = in * s[ch]` as a
// stored, rounded f32, and `lg_add` then rounds the sum. Writing the same thing
// as `out[i] = a[i] + b[i] * s[ch]` would let nvcc - which this project builds
// with `--fmad=true` - contract it into ONE FFMA, which rounds once and is a
// different number. `__fadd_rn` and `__fmul_rn` are the explicitly-rounded
// forms, which are never contracted, so this kernel reproduces the two-kernel
// result bit for bit. That is checked by the 46-stage dump, not assumed.
//
// The elementwise shape is deliberately the same as `lg_add`'s: one thread per
// element, flat grid.
// ---------------------------------------------------------------------------

extern "C" __global__ void nf_residual(
    const float *__restrict__ a,
    const float *__restrict__ b,
    const float *__restrict__ s,
    float *__restrict__ out,
    int c,
    int hw)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * hw;
    if (idx >= total) return;
    const int ch = (int)(idx / hw);
    out[idx] = __fadd_rn(a[idx], __fmul_rn(b[idx], s[ch]));
}
