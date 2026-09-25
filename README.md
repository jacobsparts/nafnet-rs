# nafnet-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs),
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs) and
[locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs); they
share the [lightgpu toolkit](https://github.com/jacobsparts/lightgpu).
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
* 1.60 MiB binary (1,677,728 bytes), statically linked except `libc` and
  `libgcc_s`. `libcuda.so.1` is `dlopen`ed, so the CPU path works on a machine
  with no NVIDIA driver at all. (The CPU-only build is 1.05 MiB.)
* All five published NAFNet configurations, converted: deblur (GoPro, REDS) and
  denoise (SIDD), each in a **width 32** build for speed and a **width 64** build
  for quality. They are attached to the releases; see Choosing a checkpoint.
* **Both backends are faster than PyTorch on the machine this was built on**
  (see Performance).

## Download

Prebuilt binaries and the converted checkpoints are attached to the
[releases](https://github.com/jacobsparts/nafnet-rs/releases):

| asset | contents | notes |
|---|---|---|
| `nafnet-linux-x86_64` | CPU + CUDA, selected with `--device` | any x86-64 Linux with glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); the GPU path needs an NVIDIA driver and a compute capability 6.1+ GPU |
| `nafnet-linux-x86_64-cpu-only` | CPU only | same, but nothing NVIDIA-related is ever touched - `--device gpu` is refused rather than failing obscurely |
| `nafnet-gopro-width32.safetensors` | checkpoint: **deblur - speed** | GoPro, width 32 |
| `nafnet-gopro-width64.safetensors` | checkpoint: **deblur - quality** | GoPro, width 64 |
| `nafnet-reds-width64.safetensors` | checkpoint: deblur | REDS, width 64; JPEG-damaged video |
| `nafnet-sidd-width32.safetensors` | checkpoint: **denoise - speed** | SIDD, width 32 |
| `nafnet-sidd-width64.safetensors` | checkpoint: **denoise - quality** | SIDD, width 64 |

The engine reads these files directly, so nothing needs converting to try one.

```sh
./nafnet-linux-x86_64 -m nafnet-gopro-width32.safetensors -i blurry.png -o sharp.png
```

**Both** binaries carry the CPU backend; the difference is whether the CUDA
sections are in them.

## Build

```sh
cargo build --release
# CPU only, no CUDA toolkit or driver needed at build time either:
cargo build --release --no-default-features
```

The default build needs `nvcc` (set `NVCC=` if it is not on `PATH`) and produces
one binary with both backends. The `--no-default-features` build contains only
the CPU path, which reports `this build has no cuda feature; use --device cpu` if
asked for the GPU rather than failing obscurely. The kernels cover `sm_61`,
`sm_75`, `sm_80` and compute capability 8.0 PTX, so the GPU path runs on Pascal
(GTX 10-series) through Ampere, and on anything newer via the PTX.

`lightgpu` is a normal Cargo dependency on
[its repository](https://github.com/jacobsparts/lightgpu), so a clone of this
project builds on its own.

### Development build

The flags used to verify the engine are not in a release binary. They are behind
a non-default feature, and a release build **refuses them by name** (with the
rebuild command) rather than ignoring them:

```sh
cargo build --release --features dev
```

| flag | what it is for |
|---|---|
| `--dump <path>` | every intermediate activation as flat f32 plus a text index, for stage-by-stage comparison against `tools/reference.py`. Both backends implement it. |
| `--raw <path> --size <h> <w>` | run the network on a `[3][h][w]` f32 plane instead of a PNG |
| `--cuda-selftest` | compare each CUDA kernel against its CPU twin and exit |
| `--profile` | per-kernel GPU time, longest first |

`--raw` plus the reference's `--seed`/`--dump` gives both sides byte-identical
input, so a per-stage difference is the model rather than the PNG loader.

## Choosing a checkpoint

Which task, and how much quality, are both decided by which file you pass to
`-m`. There is no flag for either: the task picks the training distribution and
the width picks quality against speed.

| checkpoint | what it is | params | upstream PSNR |
|---|---|---|---|
| `nafnet-gopro-width32` | deblur, GoPro motion blur - **speed** | 17.1 M | 32.8705 dB |
| `nafnet-gopro-width64` | deblur, GoPro motion blur - **quality** | 67.9 M | 33.7103 dB |
| `nafnet-reds-width64` | deblur, JPEG-damaged video | 67.9 M | 29.0903 dB |
| `nafnet-sidd-width32` | denoise, real camera noise - **speed** | 29.2 M | 39.9672 dB |
| `nafnet-sidd-width64` | denoise, real camera noise - **quality** | 116.0 M | 40.3045 dB |

**Width 64 is the quality model and width 32 is the speed model.** Width is the
network's base channel count - the architecture, not a tuning knob - so the 64
models are ~4x the parameters and cost ~2.6x the GPU time and ~3.8x the CPU
time: a width-32 model takes 0.74-0.78 s on the GPU and 2.3-2.6 s on the CPU at
1280x725, a width-64 model about 1.9 s and 9-14 s. Start with a width-32 file
and move up when you want the last fraction of a dB.

Pick the task by what the picture actually is. The models are not
interchangeable: a GoPro model run on a JPEG-damaged frame, or a REDS model on
clean sensor noise, is off its training distribution and can make the image
worse rather than better. The PSNR figures are the upstream authors' own, not
measured here.

### Where the checkpoints come from

The engine reads a `.safetensors` file converted from an official NAFNet
checkpoint. The architecture constants are read from that file rather than
inferred, so a converted file cannot be checked against the wrong config:

```sh
python3 tools/convert.py NAFNet-GoPro-width32.pth nafnet-gopro-width32.safetensors
```

`--task` (gopro, sidd, reds) and `--width` (32 or 64) are recorded in the file's
metadata and default to being inferred from the weights and the file name, so
converting an unlisted checkpoint needs no flags. The official `.pth` files come
from [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) and are
not redistributed here; the converted ones are attached to the releases.

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

Both figures move with machine load, and the wider models are slower: the
width-64 checkpoints take 1.9-2.1 s on the GPU and 8.5-14 s on the CPU for the
same image. See Choosing a checkpoint.

`--profile` (a development build) reports per-kernel GPU time longest first, and
`NAFNET_CPU_PROFILE=1` does the same for the CPU. On both backends the single
largest cost is the 1x1 convolution every NAFBlock uses four times, which is
where the tiling effort went.

## Verification

The correctness record is `tools/reference.py` - a PyTorch NAFNet transcribed
from the official sources - plus a checkpoint, which are the only two things the
engine and the reference have in common. An engine agreeing with its own CPU twin
would prove only that the two share a reading of the paper.

* `--cuda-selftest` (development build) compares each CUDA kernel against its CPU
  twin on identical inputs at real magnitudes, with a relative tolerance: 20
  checks, 0 failed.
* `--dump` (development build) writes every intermediate activation and the two
  sides are compared **stage by stage, by name**; both backends implement it.
* Against the reference at 32x32 the worst stage is 2.740e-02, and the two
  backends agree with each other to 7.324e-03 worst stage of 128x128, which is
  float32 accumulation order and nothing else. At 1280x725 the CPU and GPU output
  PNGs are byte-identical.
* A second geometry was validated the same way, because the checkpoints are not
  all the same network: `nafnet-sidd-width32` has `enc [2,2,4,8]` and 12 middle
  blocks against GoPro's `enc [1,1,1,28]` and 1. Its output is 119.5 dB from the
  reference, the same order as the GoPro config's own, so a configuration the
  engine was not written against behaves like the one it was.

Where the two backends still differ, and why, is in
[docs/NUMERICS.md](docs/NUMERICS.md) - along with the accumulation orders that
must not be changed, and how to check an edit to a kernel.

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license;
see [LICENSE](LICENSE).

This is an independent reimplementation of the NAFNet architecture, which is by
[megvii-model](https://github.com/megvii-research/NAFNet) and MIT licensed
(© 2022 megvii-model). `tools/reference.py` is a PyTorch transcription of their
network and is therefore a derived work, not covered by this repository's
copyright; `tools/convert.py` transcribes their published configuration
constants. The **checkpoints** are the NAFNet authors' work as well. The five converted
`.safetensors` files attached to the releases are format conversions of the
official `NAFNet-GoPro-width32`, `NAFNet-GoPro-width64`, `NAFNet-REDS-width64`,
`NAFNet-SIDD-width32` and `NAFNet-SIDD-width64` `.pth` files, and are
redistributed under the same MIT terms as the upstream release; the original
`.pth` files are not redistributed here.

The upstream PSNR figures quoted above are from the NAFNet paper and repository
and are reproduced as the authors report them, not measured here.
