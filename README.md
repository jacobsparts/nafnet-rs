# nafnet-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs),
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs),
[locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs) and
maxim-rs (a sibling checkout so far, not published yet); they share the
[lightgpu toolkit](https://github.com/jacobsparts/lightgpu).
[pixeldeck](https://github.com/jacobsparts/pixeldeck) is a local web app for
cleaning up product photos that drives all of these engines.

[NAFNet](https://github.com/megvii-research/NAFNet) image restoration - deblur
and denoise - as a single self-contained binary. Feed it a PNG, get back a
restored PNG. No Python at inference, no PyTorch, no ONNX Runtime, no CUDA
toolkit needed to run it.

```
nafnet -m nafnet-gopro-width32.safetensors -i blurry.png -o sharp.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels, selected at run time with `--device cpu|gpu`.
* 1.60 MiB binary (1,677,632 bytes), statically linked except `libc` and
  `libgcc_s`. The CUDA kernels are embedded as two fatbins - only the ones this
  engine calls - and `libcuda.so.1` is `dlopen`ed, so the CPU path works on a
  machine with no NVIDIA driver at all. (The CPU-only build is 1.05 MiB
  / 1,098,000 bytes.)
* Coded for several published NAFNet configurations (`--task`/`--width` in
  `tools/convert.py`); the accuracy figures below are for the GoPro width-32
  one.
* **Both backends are faster than PyTorch on the machine this was built on**
  (see Performance).

## Download

Prebuilt binaries and the converted checkpoint are attached to the
[releases](https://github.com/jacobsparts/nafnet-rs/releases):

| asset | contents | runs on |
|---|---|---|
| `nafnet-linux-x86_64` | CPU + CUDA, selected with `--device` | any x86-64 Linux with glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); the GPU path needs an NVIDIA driver and a compute capability 6.1+ GPU |
| `nafnet-linux-x86_64-cpu-only` | CPU only | same, but nothing NVIDIA-related is ever touched - `--device gpu` is refused rather than failing obscurely |
| `nafnet-gopro-width32.safetensors` | the converted GoPro width-32 checkpoint | the engine reads this file directly, so nothing needs converting to try it |

```sh
./nafnet-linux-x86_64 -m nafnet-gopro-width32.safetensors -i blurry.png -o sharp.png
```

**Both** binaries carry the CPU backend; the difference is whether the CUDA
sections are in them, so a machine with no NVIDIA driver can still use the first
one with `--device cpu`.

## Build

```sh
cargo build --release
# CPU only, no CUDA toolkit or driver needed at build time either:
cargo build --release --no-default-features
```

The default build needs `nvcc` (set `NVCC=` if it is not on `PATH`) and a
CUDA toolkit; it produces one binary with both backends. The
`--no-default-features` build produces a binary containing only the CPU path,
which reports `this build has no cuda feature; use --device cpu` if asked for
the GPU rather than failing obscurely.

The kernels are compiled for `sm_61`, `sm_75`, `sm_80` and compute capability
8.0 PTX, so the GPU path runs on Pascal (GTX 10-series) through Ampere, and on
anything newer via the PTX.

`lightgpu` is a sibling checkout rather than a git dependency, so build this
next to the toolkit:

```
lightgpu-family/
├── lightgpu/
├── nafnet-rs/     <- here
└── ...
```

Switching the dependency to `git = "https://github.com/jacobsparts/lightgpu.git"`
is a one-line change, but do not make it casually: this engine launches
`lg_channel_layer_norm` with a grid of `hw / 256` blocks of 256 threads, which
matches the toolkit's current one-thread-per-position form. An older toolkit
kernel with one *block* per position would be under-covered by that grid and
would leave most of the output unwritten **without reporting an error**, because
the kernel has no way to know the host intended more blocks.

### Development build

The flags used to verify the engine are not in a release binary - the release
build is 100 KB smaller for it. They are behind a non-default feature, and a
release build **refuses them by name** (with the rebuild command) rather than
ignoring them, so a script that asked for a dump cannot carry on as though it
had got one:

```sh
cargo build --release --features dev
```

| flag | what it is for |
|---|---|
| `--dump <path>` | every intermediate activation as flat f32 plus a text index, for stage-by-stage comparison against `tools/reference.py`. Both backends implement it. |
| `--raw <path> --size <h> <w>` | run the network on a `[3][h][w]` f32 plane instead of a PNG |
| `--cuda-selftest` | compare each CUDA kernel against its CPU twin and exit |
| `--profile` | per-kernel GPU time, longest first |

`--raw` plus the reference's `--seed`/`--dump` is what makes a per-stage
comparison meaningful at all: it gives both sides byte-identical input, so a
difference is the model rather than the PNG loader.

## Model

The engine reads a `.safetensors` file converted from an official NAFNet
checkpoint, and the architecture constants are read from that file rather than
inferred, so a converted file cannot be shape-checked against the wrong config:

```sh
python3 tools/convert.py NAFNet-GoPro-width32.pth nafnet-gopro-width32.safetensors
```

`--task` (gopro, sidd, reds) and `--width` (32 or 64) are recorded in the
converted file's metadata; both default to being inferred from the weights and
the source file name. The official `.pth` checkpoints come from
[megvii-research/NAFNet](https://github.com/megvii-research/NAFNet), and the
original files are not redistributed here - but the **converted**
`nafnet-gopro-width32.safetensors` is attached to the releases, because the
conversion is mechanical and having it means the engine can be run without a
PyTorch install anywhere in the loop. `NAFNet-GoPro-width32` is the one this
engine was validated against, so it is the one to start with.

## Usage

```sh
nafnet -m nafnet-gopro-width32.safetensors -i blurry.png -o sharp.png
nafnet -m model.safetensors -i in.png -o out.png --device cpu
```

```
-m, --model <path>    converted .safetensors checkpoint
-i, --input <path>    input PNG, or - for stdin (default: stdin)
-o, --output <path>   output PNG, or - for stdout (default: stdout)
    --device <dev>    gpu or cpu
    --pad <mode>      reflect (default, what the reference uses) or zero
-q, --quiet           no progress output
```

Nothing else is in a release binary. In particular `--tile` is **not** accepted:
the other engines in the family take it, so a shared driver script may pass it,
but this engine runs a whole image at once and a flag that is swallowed in
silence would let you believe it had bounded the memory this process uses.
Memory is bounded by an allocation plan instead (see `gpu::Plan`), and a release
build answers `--tile` with that reason rather than a fake success.

The input is padded up to a multiple of 16 by reflection, which is what the
reference does; zero-padding changes the first and last rows the network sees
and therefore the output near the border.

## Performance

NAFNet-GoPro-width32, 1280x725 input, on an i7-13700K (8 P-cores + 8 E-cores)
with a GTX 1080 (Pascal, sm_61):

| | this engine | PyTorch 2.6 (cu124) |
|---|---|---|
| GPU | **0.70-0.72 s**, 250 MB peak RSS | 1.68 s |
| CPU | **2.2-2.3 s**, 37 s user time, 1.34 GB peak RSS | 3.46 s |

The CPU figure moves with machine load (24 hardware threads are in play); the
user time and the peak RSS are the stable numbers.

`--profile` (a development build) reports per-kernel GPU time, longest first.
The single largest cost
is the 1x1 convolution that every NAFBlock uses four times, `nf_conv1x1_oc`, at
283-289 ms of a ~490 ms pass over 184 launches; its tile sizes were chosen by a
two-dimensional sweep, and re-sweeping that grid confirms the current point is
the optimum. Behind it sit five passes in the 20-42 ms range - the depthwise 3x3,
the channel LayerNorm, the SimpleGate multiply, the per-channel residual and the
2x2 stride-2 downsample - which is what a memory-bound pass over tensors this
size costs on this card. On the CPU the same 1x1 convolution dominates even more
heavily (1310 ms of 1655 ms of instrumented op time), which is why that is where
the tiling effort went on both backends.

Set `NAFNET_CPU_PROFILE=1` for the CPU equivalent: it prints the same
longest-first table for the CPU kernels.

## Verification

The correctness record is `tools/reference.py`: a PyTorch NAFNet transcribed
from the official sources, plus the checkpoint, which are the only two things
the engine and the reference are given in common. A second implementation is
needed because an engine agreeing with its own CPU twin proves only that the
two share a reading of the paper.

* `--cuda-selftest` (development build) compares each CUDA kernel against its CPU
  twin on identical inputs, at real magnitudes with a RELATIVE tolerance. 20
  checks, 0 failed.
* `--dump` (development build) writes every intermediate activation of a pass as
  flat float32 with a text index, and the two are compared **stage by stage by
  name**. Both backends implement it.
* On the real checkpoint: CPU against the PyTorch reference is 2.740e-02 worst
  stage (at `decoders.0.0` of 32x32), and the two backends agree with each other
  to 7.324e-03 worst stage (of 128x128), which is the float32 accumulation order
  and nothing else.
* At full 1280x725 the output PNG is byte-identical between this engine's CPU
  and GPU paths, at 1.34 GB peak host memory (CPU) and 250 MB (GPU).

The divergence that remains is documented rather than hidden: the CUDA kernels
are built with `--fmad=true` so their `acc += w * x` contracts to a single
rounded FFMA, while Rust does not contract, and the CPU kernels therefore fuse
the matching multiply-adds explicitly (`.cargo/config.toml` enables `+fma` for
that reason, and the code has a documented non-FMA fallback). Two operations
still differ by 2 ulp: the depthwise downsample, whose four-product expression
nvcc groups into a sub-tree no portable Rust form matches, and the toolkit's
`lg_channel_layer_norm`, which uses CUDA's `rsqrtf` and has no Rust equivalent.

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license;
see [LICENSE](LICENSE).

This is an independent reimplementation of the NAFNet architecture, which is by
[megvii-model](https://github.com/megvii-research/NAFNet) and MIT licensed
(© 2022 megvii-model). `tools/reference.py` is a PyTorch transcription of their
network and is therefore a derived work, not covered by this repository's
copyright; `tools/convert.py` transcribes their published configuration
constants. The **checkpoints** are the NAFNet authors' work as well. The
converted `nafnet-gopro-width32.safetensors` attached to the releases is a
format conversion of the official `NAFNet-GoPro-width32.pth` and is redistributed
under the same MIT terms as the upstream release; the original `.pth` is not
redistributed here.
