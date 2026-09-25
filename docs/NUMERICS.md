# Numerical notes

Why the CPU and CUDA backends agree as closely as they do, the few places where
they deliberately do not, and how to check that a change has not moved either.
None of this is needed to run the engine; it is here for anyone editing a kernel.

## The compiler is part of the arithmetic

The CUDA kernels are compiled by `nvcc` with `--prec-div=true --prec-sqrt=true
--fmad=true`, so an `acc += w * x` in a kernel contracts into a single-rounded
FFMA - no intermediate rounding of the product. Rust does not contract float
operations, so the same expression in a CPU kernel rounds twice.

The CPU kernels therefore fuse the matching multiply-adds explicitly, through a
cfg-gated `fma()` helper that falls back to `a * b + c` where the target has no
FMA. This is not a portability afterthought: `f32::mul_add` on a target without
`+fma` lowers to a *call* into `compiler_builtins`' soft-float `fmaf`, which also
blocks vectorisation. The same pass went from 96 s to 550 s of user time that way,
for identical results. `.cargo/config.toml` sets `target-feature=+fma` so the
real instruction is used, which means the CPU binary wants a Haswell or
Piledriver (2012 or later); on anything older the fallback compiles and runs,
just more slowly.

Fusing chose itself in one place and had to be rejected in another, and only
measurement settled both. In `lg_channel_layer_norm`, ptxas contracts
`s2 += v * v` and fusing it improves the two backends' agreement; fusing
`s2 / n - mean * mean` makes it *worse* (worst stage 7.32e-03 to 1.05e-02), so
that kernel fuses one and not the other.

## Where the backends still differ

Two operations are left more than 1 ulp apart, both documented in the source:

* `nf_down2x2s2` - `nvcc` groups its four-product expression into a sub-tree that
  no portable Rust form reproduces, so it sits at **2 ulp**.
* `lg_channel_layer_norm` - the kernel uses CUDA's `rsqrtf()`, which has no Rust
  equivalent, so it sits at **2.384e-07** relative.

Neither is a free choice: matching them would mean a Rust kernel that no longer
reads like the arithmetic it expresses. They are bounded at 2 ulp, and the
stage-by-stage comparison below sees them.

`nf_residual` (`y = a + b * s[ch]`) takes the opposite approach deliberately: it
writes `__fadd_rn(a, __fmul_rn(b, s[ch]))`, the explicitly-rounded forms, because
the obvious `a + b * s[ch]` *would* be contracted by `--fmad=true` into one FFMA
and would then disagree with the two operations it replaced.

## Accumulation order is contractual

Several CPU kernels depend on the order they accumulate in. Each of these was a
real 2-ulp bug when it changed, so they are not to be "tidied":

* `conv3x3` accumulates `ky`, then `kx`, then `ic`.
* `conv3x3_dw` accumulates `ky`, then `kx`.
* `down2x2s2` keeps `acc := bias` and then four fused per-`ic` accumulates in
  (ky, kx) order.
* `lg_channel_mean`'s summation order is fixed for the same reason.
* `lg_channel_layer_norm`'s is *not* contractual, and its order changed
  deliberately when it was rewritten from one block per position to one thread
  per position.

## How to check a change

The record is `tools/reference.py`, a PyTorch transcription of the official
network, plus a checkpoint. `--dump` writes every intermediate activation as flat
float32 with a text index, and the two sides are compared **stage by stage, by
name** - by name, not by byte offset, because the CPU dump has an extra leading
`in` stage and the offsets therefore do not line up.

* `--cuda-selftest` compares each CUDA kernel against its CPU twin on identical
  inputs at real magnitudes, with a **relative** tolerance of 1e-5. The earlier
  absolute tolerance with tiny shapes (a `6x4x5` tensor, `hw = 20`) could not see
  a summation-order difference at all.
* The two backends' 128x128 dumps agree to 7.324e-03 worst stage. Against the
  reference the worst stage is 2.740e-02 at 32x32, and both backends sit ~97 dB
  from it at the output - the CPU twin reproduces the CUDA kernel rather than
  PyTorch, which is the point of fusing the FMA.
* The end-to-end check is the output PNG: at 1280x725 the CPU and GPU paths
  produce byte-identical files, and both are reproducible from a fresh clone.

A comment-only edit changes the binary's bytes (the build-id and metadata
sections are hashed from the source), so **binary byte-identity is not a valid
regression check** across source edits. The output PNG is.
