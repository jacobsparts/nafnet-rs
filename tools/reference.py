#!/usr/bin/env python3
"""A PyTorch NAFNet, transcribed from the official sources, as the correctness
record for this engine.

PROVENANCE, because this file is a derivate work: the network and the block
structure are transcribed from NAFNet (https://github.com/megvii-research/NAFNet,
MIT licensed, Copyright (c) 2022 megvii-model). The transcription is deliberate -
an engine checked against its own CPU twin only proves the two share a reading -
but it also means this file is NOT covered by nafnet-rs's own copyright. Keep the
attribution if it is ever moved.

WHY A SECOND IMPLEMENTATION: nafnet-rs agreeing with its own CPU twin proves
only that the two share a reading of the paper. The paper's reading has a
specific shape - LayerNorm over the CHANNEL axis at each position, a bias-free
upsample conv, SimpleGate splitting the channel axis - and each of those is a
place where a port can be self-consistent and still wrong. So the check is
against an independent transcription, and the checkpoint is the only thing the
two are given in common.

    python3 tools/reference.py --model NAFNet-GoPro-width32.pth \
        --image test.png --out ref.png [--crop 128 128] [--dump tensors.pt]

`--dump` writes every intermediate tensor so a Rust-side comparison can find
WHICH stage diverges rather than only that something did.
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """The reference's LayerNorm2d: per-position, over the CHANNEL axis."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, C, H, W]. Mean/var over C (dim=1), which is what makes this
        # different from a BatchNorm-like spatial normalisation.
        mean = x.mean(1, keepdim=True)
        var = (x * x).mean(1, keepdim=True) - mean * mean
        y = (x - mean) / torch.sqrt(var.clamp_min(0.0) + self.eps)
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        dw = 2 * c
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1, bias=True)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1, bias=True))
        self.conv3 = nn.Conv2d(c, c, 1, bias=True)
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, dw, 1, bias=True)
        self.conv5 = nn.Conv2d(c, c, 1, bias=True)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = SimpleGate()(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = SimpleGate()(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=(1, 1, 1, 28), dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.middle_blks = nn.ModuleList()

        n = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(n) for _ in range(num)]))
            self.downs.append(nn.Conv2d(n, 2 * n, 2, stride=2))
            n *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(n) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            # BIAS-FREE, as in the reference: this is the one conv whose bias
            # absence is load-bearing (the checkpoint has no `ups.*.0.bias`).
            self.ups.append(nn.Sequential(nn.Conv2d(n, n * 2, 1, bias=False), nn.PixelShuffle(2)))
            n //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(n) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, dump=None):
        b, c, h, w = inp.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        if ph or pw:
            inp = F.pad(inp, (0, pw, 0, ph), mode="reflect")
        x = self.intro(inp)
        if dump is not None:
            dump("intro", x)

        skips = []
        for i, (enc, down) in enumerate(zip(self.encoders, self.downs)):
            for j, blk in enumerate(enc):
                x = blk(x)
                if dump is not None:
                    dump(f"encoders.{i}.{j}", x)
            skips.append(x)
            x = down(x)
            if dump is not None:
                dump(f"downs.{i}", x)

        for j, blk in enumerate(self.middle_blks):
            x = blk(x)
            if dump is not None:
                dump(f"middle_blks.{j}", x)

        for i, (up, dec) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            if dump is not None:
                dump(f"ups.{i}", x)
            x = x + skips[-i - 1]
            for j, blk in enumerate(dec):
                x = blk(x)
                if dump is not None:
                    dump(f"decoders.{i}.{j}", x)

        x = self.ending(x)
        x = x + inp
        if dump is not None:
            dump("out", x)
        return x


def build(width, enc, mid, dec):
    return NAFNet(width=width, middle_blk_num=mid, enc_blk_nums=enc, dec_blk_nums=dec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image")
    ap.add_argument("--out")
    ap.add_argument("--crop", nargs=2, type=int, default=None, help="top-left x y")
    ap.add_argument("--size", nargs=2, type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="use random input instead of an image")
    ap.add_argument("--dump", help="torch.save every intermediate tensor here")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--enc", default="1,1,1,28")
    ap.add_argument("--mid", type=int, default=1)
    ap.add_argument("--dec", default="1,1,1,1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "params" in state:
        state = state["params"]
    width = args.width or int(state["intro.weight"].shape[0])
    enc = [int(v) for v in args.enc.split(",")]
    dec = [int(v) for v in args.dec.split(",")]

    net = build(width, enc, args.mid, dec)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"state_dict: {len(missing)} missing, {len(unexpected)} unexpected", file=sys.stderr)
        for k in list(missing)[:5]:
            print(f"  missing {k}", file=sys.stderr)
        for k in list(unexpected)[:5]:
            print(f"  unexpected {k}", file=sys.stderr)
        return 1
    net.eval().to(args.device)

    if args.seed is not None:
        g = torch.Generator().manual_seed(args.seed)
        h, w = args.size or (32, 32)
        x = torch.rand(1, 3, h, w, generator=g).to(args.device)
    else:
        if not args.image:
            print("--image or --seed is required", file=sys.stderr)
            return 2
        from PIL import Image
        im = Image.open(args.image).convert("RGB")
        a = np.asarray(im, dtype=np.float32) / 255.0
        if args.crop:
            cx, cy = args.crop
            a = a[cy:cy + (args.size[1] if args.size else a.shape[0]), cx:cx + (args.size[0] if args.size else a.shape[1])]
        if args.size:
            a = a[: args.size[1], : args.size[0]]
        x = torch.from_numpy(a).permute(2, 0, 1)[None].to(args.device)

    d = None
    if args.dump:
        store = {}
        d = lambda name, t: store.__setitem__(name, t.detach().float().cpu())

    with torch.no_grad():
        y = net(x, dump=d)
    y = y.clamp(0, 1)

    if args.dump:
        torch.save({"in": x.detach().float().cpu(), **store}, args.dump)
        print(f"{args.dump}: {len(store) + 1} tensors")

    if args.out:
        a = (y[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        from PIL import Image
        Image.fromarray(a).save(args.out)
        print(f"{args.out}: {a.shape[1]}x{a.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
