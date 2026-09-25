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
  hand-written kernels. The GPU is used when the CUDA driver can be brought up
  and the CPU path otherwise, so one binary covers a machine with no NVIDIA
  driver at all; `--device cpu|gpu` overrides that choice.
* 1.59 MiB binary (1,666,880 bytes), statically linked except `libc` and
  `libgcc_s`. `libcuda.so.1` is `dlopen`ed, so the CPU path works on a machine
  with no NVIDIA driver at all. (The CPU-only build is 1.05 MiB.)
* All five published NAFNet configurations, converted from the official `.pth`
  files: deblur (GoPro, REDS) and denoise (SIDD), in **width 32** builds for
  speed and **width 64** for quality. They are attached to the releases; see
  Choosing a checkpoint.
* **Both backends are faster than PyTorch on the machine this was built on**
  (see Performance).

## Download

Prebuilt binaries and the converted checkpoints are attached to the
[releases](https://github.com/jacobsparts/nafnet-rs/releases). Both binaries
carry the CPU backend; the difference is whether the CUDA sections are in them.

| asset | contents | notes |
|---|---|---|
| `nafnet-linux-x86_64` | CPU + CUDA, auto-selected | any x86-64 Linux with glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+). The CUDA sections do not oblige you to have a driver: with none, the CPU path runs and says so. The GPU path needs a compute capability 6.1+ GPU |
| `nafnet-linux-x86_64-cpu-only` | CPU only | same, but nothing NVIDIA-related is ever touched - `--device gpu` is refused rather than failing obscurely |
| the 5 `*.safetensors` checkpoints | deblur (GoPro, REDS) and denoise (SIDD), width 32 and 64 | see Choosing a checkpoint |

```sh
./nafnet-linux-x86_64 -m nafnet-gopro-width32.safetensors -i blurry.png -o sharp.png
```

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

## Usage

```sh
nafnet -m nafnet-gopro-width32.safetensors -i blurry.png -o sharp.png
nafnet -m model.safetensors -i in.png -o out.png --device cpu
```

```
-m, --model <path>    converted .safetensors checkpoint (see tools/convert.py)
-i, --input <path>    input PNG, or - for stdin (default: stdin)
-o, --output <path>   output PNG, or - for stdout (default: stdout)
    --device <dev>    gpu or cpu (default: gpu when the CUDA driver can be
                      brought up, cpu otherwise; a CPU-only build is always
                      cpu)
    --cpu             same as --device cpu
    --gpu             same as --device gpu, and refuses to fall back
-q, --quiet           no progress output
-h, --help            this text
-V, --version         print the version
```

Nothing else is in a release binary. There is no `--tile` (other engines in the
family have one): this engine runs a whole image at once, so a memory-control
flag swallowed in silence would let a caller believe it had bounded this
process's memory. Asking for `--tile` gets that reason rather than a fake
success.

The input is padded up to a multiple of 16 by reflection, which is what the
reference does.

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

On both backends the largest single cost is the 1x1 convolution every NAFBlock
uses four times, which is where the tiling effort went.

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license;
see [LICENSE](LICENSE).

This is an independent reimplementation of the NAFNet architecture, which is by
[megvii-model](https://github.com/megvii-research/NAFNet) and MIT licensed
(© 2022 megvii-model). `tools/reference.py` is a PyTorch transcription of their
network and is therefore a derived work, not covered by this repository's
copyright; `tools/convert.py` transcribes their published configuration
constants. The **checkpoints** are the NAFNet authors' work as well: the five
converted `.safetensors` files attached to the releases are format conversions of
the official `NAFNet-GoPro-width32`, `NAFNet-GoPro-width64`, `NAFNet-REDS-width64`,
`NAFNet-SIDD-width32` and `NAFNet-SIDD-width64` `.pth` files, redistributed under
the same MIT terms as the upstream release. The original `.pth` files are not
redistributed here.

The upstream PSNR figures quoted above are from the NAFNet paper and repository
and are reproduced as the authors report them, not measured here.
