#!/usr/bin/env python3
"""Convert a NAFNet .pth checkpoint into the .safetensors this engine loads.

The released configurations below are taken from NAFNet's own
`options/test/*/NAFNet-*.yml` (https://github.com/megvii-research/NAFNet, MIT
licensed); the checkpoint tensors themselves are the NAFNet authors' work and are
covered by that project's licence and the terms on its model releases, not by
nafnet-rs's.

The Rust side reads the architecture constants out of `__metadata__`, so they
are written here rather than inferred at load: there are four published configs
and a converted file that does not say which one it is cannot be shape-checked
against its own weights.

    python3 tools/convert.py NAFNet-GoPro-width32.pth nafnet-gopro-width32.safetensors

`--task` is recorded for the accuracy table in the README (it names which
published figure the checkpoint is the source of). The default guesses from the
file name.

The tensor names are written through unchanged, including the leading
`params.` stripped by the official checkpoints. Nothing is transposed: the
checkpoint already stores conv weights as [c_out][c_in][kh][kw] and the
depthwise one as [c_out][1][3][3], which is what both backends read.
"""
import argparse
import json
import re
import struct
import sys

import numpy as np
import torch

# The released configurations, from NAFNet's options/test/*/NAFNet-*.yml.
CONFIGS = {
    ("gopro", 32): dict(enc=[1, 1, 1, 28], mid=1, dec=[1, 1, 1, 1]),
    ("gopro", 64): dict(enc=[1, 1, 1, 28], mid=1, dec=[1, 1, 1, 1]),
    ("reds", 64): dict(enc=[1, 1, 1, 28], mid=1, dec=[1, 1, 1, 1]),
    ("sidd", 32): dict(enc=[2, 2, 4, 8], mid=12, dec=[2, 2, 2, 2]),
    ("sidd", 64): dict(enc=[2, 2, 4, 8], mid=12, dec=[2, 2, 2, 2]),
}


def guess_task(path: str):
    n = path.lower()
    for t in ("gopro", "sidd", "reds"):
        if t in n:
            return t
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--task", default=None, help="gopro, sidd or reds (default: from the file name)")
    ap.add_argument("--width", type=int, default=None, help="32 or 64 (default: inferred from the weights)")
    args = ap.parse_args()

    state = torch.load(args.src, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "params" in state:
        state = state["params"]
    if not isinstance(state, dict):
        print(f"{args.src}: not a state dict ({type(state)})", file=sys.stderr)
        return 1
    state = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

    task = args.task or guess_task(args.src) or "unknown"

    # Width from intro.weight's output channels; the block counts from the
    # encoders/decoders key sets, so a converted file's metadata always matches
    # the tensors actually in it.
    if "intro.weight" not in state:
        print(f"{args.src}: no intro.weight - not a NAFNet checkpoint?", file=sys.stderr)
        return 1
    width = args.width or int(state["intro.weight"].shape[0])
    levels = len({int(k.split(".")[1]) for k in state if k.startswith("encoders.")})
    enc = [0] * levels
    dec = [0] * levels
    for k in state:
        parts = k.split(".")
        if k.startswith("encoders.") and len(parts) > 2:
            enc[int(parts[1])] = max(enc[int(parts[1])], int(parts[2]) + 1)
        if k.startswith("decoders.") and len(parts) > 2:
            dec[int(parts[1])] = max(dec[int(parts[1])], int(parts[2]) + 1)
    mid = len({k.split(".")[1] for k in state if k.startswith("middle_blks.")})

    known = CONFIGS.get((task, width))
    if known and (known["enc"] != enc or known["dec"] != dec or known["mid"] != mid):
        print(
            f"warning: the weights are enc={enc} mid={mid} dec={dec} but the published "
            f"{task}-width{width} config is enc={known['enc']} mid={known['mid']} dec={known['dec']}",
            file=sys.stderr,
        )

    metadata = {
        "width": str(width),
        "enc_blk_nums": ",".join(str(v) for v in enc),
        "middle_blk_num": str(mid),
        "dec_blk_nums": ",".join(str(v) for v in dec),
        "task": task,
        "format": "pt",
    }

    # safetensors: an 8-byte little-endian header length, the JSON header, then
    # the raw tensor payloads. Written directly rather than through the
    # safetensors package so this script needs only torch and numpy.
    tensors = []
    for name, t in state.items():
        a = t.detach().to(torch.float32).contiguous().numpy()
        tensors.append((name, a))
    tensors.sort(key=lambda kv: kv[0])

    offset = 0
    header = {}
    for name, a in tensors:
        n = a.size * 4
        header[name] = {
            "dtype": "F32",
            "shape": list(a.shape),
            "data_offsets": [offset, offset + n],
        }
        # PAD EACH TENSOR TO A 4-BYTE BOUNDARY. safetensors does not require it,
        # but the engine mmaps the file and hands out `&[f32]` slices, so a
        # tensor starting mid-word would be an unaligned read - which the loader
        # refuses rather than silently producing garbage. A checkpoint writer
        # that packs tightly (as this one did before) makes the second tensor
        # onward unreadable.
        offset += (n + 3) & ~3
    header["__metadata__"] = metadata
    hjson = json.dumps(header, separators=(",", ":")).encode()

    header_bytes = len(struct.pack("<Q", 0)) + len(hjson)
    pad = (-header_bytes) & 7
    with open(args.dst, "wb") as f:
        f.write(struct.pack("<Q", len(hjson) + pad))
        f.write(hjson)
        f.write(b" " * pad)
        written = 0
        for name, a in tensors:
            raw = np.ascontiguousarray(a, dtype="<f4").tobytes()
            want = header[name]["data_offsets"][0]
            if written < want:
                f.write(b"\0" * (want - written))
                written = want
            f.write(raw)
            written += len(raw)

    total = sum(a.size for _, a in tensors)
    print(f"{args.dst}: {len(tensors)} tensors, {total} values, task {task}, width {width}")
    print(f"  enc {enc} mid {mid} dec {dec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
