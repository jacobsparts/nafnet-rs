# nafnet-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs),
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs),
[locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs) and
[maxim-rs](https://github.com/jacobsparts/maxim-rs), all
built on the [lightgpu toolkit](https://github.com/jacobsparts/lightgpu);
[pixeldeck](https://github.com/jacobsparts/pixeldeck) is a local web app for
cleaning up product photos that drives them all.

[NAFNet](https://github.com/megvii-research/NAFNet) image restoration - deblur
and denoise - as a single self-contained binary. Feed it a PNG, get back a
restored PNG. No Python, PyTorch, ONNX Runtime, or CUDA toolkit needed at
runtime.

```
nafnet -m nafnet-gopro-width32.safetensors -i blurry.png -o sharp.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when a CUDA driver is available and
  the CPU path otherwise, so one binary covers a machine with no NVIDIA driver
  at all; `--device cpu|gpu` overrides that choice.
* 1.59 MiB binary, statically linked except `libc` and `libgcc_s`.
  `libcuda.so.1` is `dlopen`ed, so no driver is required on disk. (The
  CPU-only build is 1.05 MiB.)
* All five published NAFNet configurations, converted from the official `.pth`
  files: deblur (GoPro, REDS) and denoise (SIDD), in **width 32** builds for
  speed and **width 64** for quality. They are attached to the releases; see
  Choosing a checkpoint below.
* **Both backends are faster than PyTorch on the machine this was built on**
  (see Performance below).

## Download

Prebuilt binaries and the converted checkpoints are attached to the
[releases](https://github.com/jacobsparts/nafnet-rs/releases). Both binaries
run on the CPU; they differ only in whether CUDA support is compiled in.

| asset | contents | notes |
|---|---|---|
| `nafnet-linux-x86_64` | CPU + CUDA, auto-selected | x86-64 Linux with glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); falls back to the CPU path when no NVIDIA driver is present. GPU path needs a compute capability 6.1+ GPU |
| `nafnet-linux-x86_64-cpu-only` | CPU only | same, with nothing NVIDIA-related included - `--device gpu` is refused |
| 5 `*.safetensors` checkpoints | deblur (GoPro, REDS) and denoise (SIDD), width 32 and 64 | see Choosing a checkpoint |

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
the CPU path, which reports `this build has no cuda feature; use --device cpu`
if asked for the GPU rather than failing obscurely. The kernels cover `sm_61`,
`sm_75`, `sm_80` and compute capability 8.0 PTX, so the GPU path runs on Pascal
(GTX 10-series) through Ampere, and on anything newer via the PTX.

`lightgpu` is a normal Cargo dependency on
[its repository](https://github.com/jacobsparts/lightgpu), so a clone of this
project builds on its own.

## Choosing a checkpoint

The task and the quality level are both decided by which file you pass to
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

## Large images

The whole image is resident on the device at once, so VRAM grows with the input.
Every buffer the pass will use is enumerated up front - as shapes, before a byte
is allocated - which makes the requirement a number rather than a matter of luck:

| checkpoint | 1280x853 | 1920x1080 | 2560x1440 | 2048x2048 |
| --- | --- | --- | --- | --- |
| width 32 (Deblur/Denoise fast) | 1.2 GB | 2.3 GB | 4.0 GB | 4.6 GB |
| width 64 (Deblur/Denoise best, Video Deblur) | 2.4 GB | 4.5 GB | 7.9 GB | 9.0 GB |

Those are plan totals and they are what the card must have FREE. Three things
keep them down, and none of them changes the arithmetic:

* the encoder's activation IS the skip the decoder reads back and IS the
  downsample's destination, so one buffer carries three roles and the pass
  contains no full-size copy at all;
* the decoder's staging slots are ONE PAIR for every level, sized to the largest,
  because the levels are strictly ordered - level `l` finishes reading before
  level `l+1` writes;
* every block shape shares ONE workspace pool sized to the largest, and inside a
  block the four planes are the live set rather than the role list.

On an 8 GB card the width-32 checkpoints run at every size the editor produces,
up to and including 2048x2048 - which is what its Photo Box 2048 routine outputs
- and the width-64 ones reach 2560x1440. 4K wants about 12 GB at width 32.

### A pass that will not fit is refused, with the numbers

There is no fallback. A plan larger than the free VRAM fails before it allocates
anything, and says what it needed and what there was:

```
nafnet: plan 24320 MiB (12032 activations + 12288 workspace), 7937 MiB free
nafnet: not enough device memory for a 4096x4096 pass
nafnet: the plan needs 24320 MiB (12032 activations + 12288 workspace); 7937 MiB is free
nafnet: the plan is exact - it is what the driver would be asked for - so this is a hard limit, not a guess
nafnet: a smaller image, a narrower checkpoint, or a freer card is what fits
```

The first line is printed on every GPU run, whether it fits or not, so which plan
was used and against how much free memory is always on the record. Free VRAM is
reported by the driver and can move by a gigabyte or two between runs on a
machine whose desktop compositor is holding device memory, so the number is worth
reading next to the plan rather than assuming.

`--device cpu` runs the same network on the host. It is slower - 5.4 s against
1.3 s on a 1920x1080 image - and it is guarded too: the CPU pass is refused
before it starts if its modelled footprint will not fit in the machine's
available memory, because a pass that runs a machine out of RAM does not fail,
it swaps. The model is conservative by design, about 1.1x the measured peak.

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
