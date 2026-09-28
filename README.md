# nafnet-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

NAFNet image restoration in one self-contained binary: deblur a motion-blurred
photo or denoise a noisy one. No Python, PyTorch, ONNX Runtime, or CUDA toolkit
needed.

```
nafnet -m nafnet-gopro-width32.safetensors -i blurry.png -o sharp.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when a CUDA driver is available and
  the CPU path otherwise, so one binary covers a machine with no NVIDIA driver
  at all; `--device cpu|gpu` overrides that choice.
* 1.59 MiB binary, statically linked except `libc` and `libgcc_s`.
  `libcuda.so.1` is `dlopen`ed, so no driver is required on disk.
* All five published NAFNet configurations, converted from the official `.pth`
  files: deblur (GoPro, REDS) and denoise (SIDD), in width-32 builds for speed
  and width-64 for quality.

Both backends reproduce the upstream PyTorch implementation's output to within
one level of 255 on a handful of values per image, none off by more.

## Download

Prebuilt binary and the converted checkpoints are attached to the
[release](https://github.com/jacobsparts/nafnet-rs/releases).

| asset | what it is |
|---|---|
| `nafnet-linux-x86_64` | the engine: x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); falls back to the CPU path when no NVIDIA driver is present, the GPU path needs a compute capability 6.1+ GPU |
| `nafnet-gopro-width32.safetensors` | deblur, GoPro motion blur - speed |
| `nafnet-gopro-width64.safetensors` | deblur, GoPro motion blur - quality |
| `nafnet-reds-width64.safetensors` | deblur, JPEG-damaged video |
| `nafnet-sidd-width32.safetensors` | denoise, real camera noise - speed |
| `nafnet-sidd-width64.safetensors` | denoise, real camera noise - quality |

```sh
chmod +x nafnet-linux-x86_64
./nafnet-linux-x86_64 -m nafnet-gopro-width32.safetensors -i blurry.png -o sharp.png
```

## Models

Five checkpoints, converted from the official `.pth` files; `-m` is the whole
choice. Width is the network's base channel count, not a tuning knob: width 64 is
~4x the parameters and 2-4x the cost for the last fraction of a dB, so start with
width 32.

| checkpoint | what it is | upstream PSNR |
|---|---|---|
| `nafnet-gopro-width32` | deblur, GoPro motion blur - **speed** | 32.8705 dB |
| `nafnet-gopro-width64` | deblur, GoPro motion blur - **quality** | 33.7103 dB |
| `nafnet-reds-width64` | deblur, JPEG-damaged video | 29.0903 dB |
| `nafnet-sidd-width32` | denoise, real camera noise - **speed** | 39.9672 dB |
| `nafnet-sidd-width64` | denoise, real camera noise - **quality** | 40.3045 dB |

The models are not interchangeable: a GoPro model run on a JPEG-damaged frame, or
a REDS model on clean sensor noise, is off its training distribution and can make
the image worse. The PSNR figures are the upstream authors' own, not measured
here.

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
                      brought up, cpu otherwise)
    --cpu             same as --device cpu
    --gpu             same as --device gpu, and refuses to fall back
-q, --quiet           no progress output
-h, --help            this text
-V, --version         print the version
```

## Large images

The whole image is resident on the device at once, so memory grows with the
input. Every buffer the pass will use is enumerated up front, which makes the
requirement a number rather than a matter of luck:

| checkpoint | 1280x853 | 1920x1080 | 2560x1440 | 2048x2048 |
| --- | --- | --- | --- | --- |
| width 32 | 1.2 GB | 2.3 GB | 4.0 GB | 4.6 GB |
| width 64 | 2.4 GB | 4.5 GB | 7.9 GB | 9.0 GB |

Those are plan totals and they are what the card (or, on `--device cpu`, the
host) must have free. On an 8 GB card the width-32 checkpoints run at every size
up to and including 2048x2048; 4K wants about 12 GB. A plan larger than the free
memory is refused with both numbers before it allocates anything:

```
nafnet: plan 24320 MiB (12032 activations + 12288 workspace), 7937 MiB free
nafnet: not enough device memory for a 4096x4096 pass
```

The input is padded up to a multiple of 16 by reflection, which is what the
reference does.

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license; see
[LICENSE](LICENSE). This is an independent reimplementation of the NAFNet
architecture, which is by [megvii-research](https://github.com/megvii-research/NAFNet)
and MIT licensed (© 2022 megvii-model). The five converted `.safetensors`
checkpoints are format conversions of the official `.pth` files, redistributed
under the same MIT terms. The original `.pth` files are not redistributed here.
