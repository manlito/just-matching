#!/usr/bin/env python3
"""Golden test vectors from the reference forward pass (tools/folded.py), for the C++ backends.

    python tools/dump_golden.py A B [--size 256] [--layers 9] [--keypoints 512]   # -> build/golden/<A>-<B>.jmtt

One file per pair, in the weights container format (jmtfile.py), holding for both images the network input,
every backbone stage, the neck and the features, then the tokens after every transformer block, the head's
coarse positions and window mass, and the final predictions and matches. Keypoints are stored as given to
the network (corner convention). The default size is 256 px to keep the file near 100 MB; at 512 px it is
four times that. Files are not committed.

The reference pass itself is checked against the original with tools/compare.py.
"""
import argparse
import os

import numpy as np
import torch

import common
import folded
import jmtfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--out")
    args = ap.parse_args()
    w = folded.load(common.WEIGHTS)
    tensors, feats, ims, xys = {}, [], [], []
    with torch.no_grad():
        for i, path in enumerate((args.a, args.b)):
            im = common.load_image(path)
            st = {}
            feats.append(folded.describe(w, im, (args.size, args.size), st))
            tensors[f"image{i}"] = (im * 255).round().byte().permute(1, 2, 0).numpy()
            tensors.update({f"im{i}.{k}": v.numpy() for k, v in st.items()})
            ims.append(im), xys.append(common.keypoints(path, args.keypoints))
            tensors[f"keypoints{i}"] = xys[-1].numpy()
        st = {}
        out = folded.match(w, feats[0], xys[0], common.size_of(ims[0]), feats[1], xys[1], common.size_of(ims[1]),
                           n_layers=args.layers, stages=st)
    tensors.update({f"pair.{k}": v.numpy() for k, v in st.items()})
    for k, v in out.items():
        tensors[f"out.{k}"] = v.numpy() if isinstance(v, torch.Tensor) else np.float32([v])
    name = "-".join(os.path.splitext(os.path.basename(p))[0] for p in (args.a, args.b))
    path = args.out or os.path.join(common.ROOT, "build", "golden", f"{name}.jmtt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"kind": "golden", "model": "flexglue", "size": args.size, "layers": args.layers,
            "filter_threshold": 0.1, "max_distance_px": 3.0}
    jmtfile.write(path, meta, tensors)
    print(f"{path}: {len(tensors)} tensors, {os.path.getsize(path) / 1e6:.0f} MB, "
          f"{int((out['matches0'] > -1).sum())} matches, covisibility {out['covisibility']:.4f}")


if __name__ == "__main__":
    main()
