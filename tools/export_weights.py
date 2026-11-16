#!/usr/bin/env python3
"""FlexGlue checkpoint -> flat weights file (same container as just-keypoints .jkpw; see jmtfile.py).

    python tools/export_weights.py                 # -> weights/flexglue.jmtw       (float32, 567 MB)
    python tools/export_weights.py --fp16          # -> weights/flexglue-fp16.jmtw  (float16, 284 MB)

Changes made to the checkpoint, none of which alters the function:
  - ConvNeXt layer scale `gamma` is folded into `pwconv2` (weight rows and bias).
  - `Wqkv` rows are reordered from (head, dim, q|k|v) to (q|k|v, head, dim), so q, k and v are contiguous.
  - 1x1 convolution weights are stored as (out, in) matrices.
  - heads.N.disambiguate is named heads.N.covisibility, as in the upstream module.
  - names: extractor.backbone.* -> backbone.*, extractor.head.* -> neck.*, extractor.descriptor_head -> desc0,
    matcher.* -> *.

The files are not committed: the backbone tensors are under the DINOv3 License and the file is large.
"""
import argparse
import os

import numpy as np
import torch

import jmtfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMIT = "99920296d082c9405ebfd612d674efe288e74881"
N_HEADS = 8


def convert(sd):
    out = {}
    gammas = {k[: -len("gamma")]: v for k, v in sd.items() if k.endswith(".gamma")}
    for k, v in sd.items():
        v = v.float()
        if k.endswith(".gamma"):
            continue
        for prefix, g in gammas.items():
            if k == prefix + "pwconv2.weight":
                v = v * g[:, None]
            elif k == prefix + "pwconv2.bias":
                v = v * g
        if k.endswith("self_attn.Wqkv.weight") or k.endswith("self_attn.Wqkv.bias"):
            dim = v.shape[0]
            perm = torch.arange(dim).view(N_HEADS, dim // (3 * N_HEADS), 3).permute(2, 0, 1).reshape(-1)
            v = v[perm]
        if v.dim() == 4 and v.shape[2:] == (1, 1):
            v = v[:, :, 0, 0]
        name = k.replace(".disambiguate.", ".covisibility.")
        for a, b in (("extractor.backbone.", "backbone."), ("extractor.head.", "neck."),
                     ("extractor.descriptor_head.", "desc0."), ("matcher.", "")):
            if name.startswith(a):
                name = b + name[len(a):]
                break
        out[name] = v.numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "third_party", "flexglue.pth"))
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tensors = convert(sd)
    if args.fp16:
        tensors = {k: v.astype(np.float16) for k, v in tensors.items()}
    out = args.out or os.path.join(ROOT, "weights", "flexglue-fp16.jmtw" if args.fp16 else "flexglue.jmtw")
    meta = {
        "kind": "weights", "model": "flexglue", "n_layers": 9, "dim": 512, "heads": N_HEADS,
        "feature_dim": 256, "search_dim": 64, "stride": 8,
        "depths": "3,3,27,3", "widths": "128,256,512,1024",
        "source": "https://github.com/cvg/FlexGlue/releases/download/v0.1/flexglue.pth",
        "source_commit": COMMIT,
        "license": "backbone.*: DINOv3 License; everything else: MIT",
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    jmtfile.write(out, meta, tensors)
    n = sum(v.size for v in tensors.values())
    print(f"{out}: {len(tensors)} tensors, {n:,} values, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
