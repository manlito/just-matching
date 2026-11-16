#!/usr/bin/env python3
"""Covisibility levels 1 and 2 of `jmt shortlist` against the reference (tools/folded.py, as tools/covis.py).

    python tools/compare_covis.py IMAGE... [--backend cpu] [--size 512] [--device cpu]

Runs `jmt shortlist --k N --rerank` (every pair, scored with level 2) and compares, pair by pair, the level-1
matrix and the level-2 scores with the reference computed from the same images. Level 2 counts grid points,
so one point that crosses the one-cell bar moves a score by 1/512. Bars: level 1 within 1e-4 (1e-3 in FP16),
level 2 within 4 points (8 in FP16).
"""
import argparse
import itertools
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

import common
import covis
import folded
import jmtfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cpu", help="device of the reference")
    ap.add_argument("--jmt", default=os.path.join(common.ROOT, "build", "release", "jmt"))
    args = ap.parse_args()
    n = len(args.images)
    out = os.path.join(common.ROOT, "build", "compare_covis.jmtt")
    cmd = [args.jmt, "shortlist", *args.images, "--k", str(n), "--rerank", "--backend", args.backend,
           "--size", str(args.size), "--weights", common.WEIGHTS, "--out", out]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode:
        sys.exit(run.stderr)
    print(run.stdout.splitlines()[-1])
    _, ours = jmtfile.read(out)
    pairs = ours["pairs"].astype(int)
    level2 = np.zeros((n, n))
    level2[pairs[:, 0], pairs[:, 1]] = ours["scores"]

    w = folded.load(common.WEIGHTS, args.device)
    with torch.no_grad():
        feats = [folded.describe(w, common.load_image(p), (args.size, args.size)) for p in args.images]
        mean = torch.stack([F.normalize(f.flatten(1).mean(1), dim=0) for f in feats]).cpu()
        cf = [folded.chw(folded.linear(w, "input_proj", folded.hwc(f))) for f in feats]
        pts = covis.grid_points(16, feats[0].device).to(feats[0])
        ref1 = (mean @ mean.T).numpy()
        ref2 = np.zeros((n, n))
        for i, j in itertools.combinations(range(n), 2):
            ref2[i, j] = covis.grid0(w, cf[i], cf[j], pts)
    iu = np.triu_indices(n, 1)
    d1 = np.abs(ours["level1"][iu] - ref1[iu])
    d2 = np.abs(level2[iu] - ref2[iu])
    print(f"{len(iu[0])} pairs; level 1: max difference {d1.max():.2e}; level 2: max difference {d2.max():.4f} "
          f"({int(round(d2.max() * 512))} of 512 points), {int((d2 > 0).sum())} pairs not identical")
    fp16 = args.backend.endswith(":fp16")  # features stored as half: about 1e-3 relative
    ok = d1.max() < (1e-3 if fp16 else 1e-4) and d2.max() <= (8 if fp16 else 4) / 512
    print("level 1 and 2 match the reference" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
