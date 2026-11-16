#!/usr/bin/env python3
"""Accuracy and time of FlexGlue against input size, number of layers and precision.

    python tools/presets.py IMAGES... [--impl upstream:mps] [--neg A B ...]

Ground truth comes from synthetic pairs: every image is warped by known homographies (one mild, one strong:
rotation, scale, perspective, brightness), so the true position of every keypoint in the other image is exact.
This measures the network, not a real capture: real pairs have depth, occlusion and lighting changes that a
homography does not. Read the numbers as trends between presets.

Per preset (size, layers, precision), over all pairs:
    err        median error of the predicted point, in pixels of the original image, over keypoints of image A
               whose true position lies inside image B
    <1px <3px  share of those keypoints predicted within 1 and 3 pixels
    matches    sparse matches per pair (mutual nearest neighbours, default thresholds)
    prec       share of sparse matches within 3 px of the truth
    covis+     lowest covisibility score over the positive pairs
    covis-     highest covisibility score over the negative pairs (--neg)
    describe   time per image, match time per pair (median of warm runs, 2048 keypoints per image)
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch

import common


def homography(rng, w, h, strength):
    """Random homography about the image centre. strength 1 = mild, 2 = strong."""
    a = np.deg2rad(rng.uniform(-12, 12) * strength)
    s = np.exp(rng.uniform(-0.15, 0.15) * strength)
    c = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1.0]])
    r = np.array([[s * np.cos(a), -s * np.sin(a), rng.uniform(-0.05, 0.05) * strength * w],
                  [s * np.sin(a), s * np.cos(a), rng.uniform(-0.05, 0.05) * strength * h],
                  [rng.uniform(-1, 1) * 8e-5 * strength, rng.uniform(-1, 1) * 8e-5 * strength, 1]])
    return c @ r @ np.linalg.inv(c)


def make_pairs(images, out_dir, seed=0):
    """[(path_a, path_b, H)] with H mapping pixel-corner coordinates of a to b."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pairs = []
    for path in images:
        bgr = cv2.imread(path)
        h, w = bgr.shape[:2]
        for strength in (1, 2):
            H = homography(rng, w, h, strength)
            # cv2 works with pixel-centre coordinates: shift by half a pixel on both sides
            T = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1.0]])
            warped = cv2.warpPerspective(bgr, T @ H @ np.linalg.inv(T), (w, h), flags=cv2.INTER_LINEAR)
            gain, bias = rng.uniform(0.8, 1.2), rng.uniform(-15, 15)
            warped = np.clip(warped.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
            out = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(path))[0]}_h{strength}.png")
            cv2.imwrite(out, warped)
            pairs.append((path, out, H))
    return pairs


def project(H, xy):
    p = np.concatenate([xy, np.ones((len(xy), 1))], 1) @ H.T
    return p[:, :2] / p[:, 2:]


def evaluate(impl, feats, items, layers, kp):
    errs, n_matches, n_correct, covis = [], [], [], []
    for a, b, H in items:
        ima, imb = feats[a][1], feats[b][1]
        xa, xb = kp[a], kp[b]
        o = impl.match(feats[a][0], xa, ima, feats[b][0], xb, imb, n_layers=layers)
        covis.append(o["covisibility"])
        if H is None:
            continue
        gt = project(H, xa.numpy().astype(np.float64))
        inside = (gt[:, 0] > 0) & (gt[:, 0] < imb[0]) & (gt[:, 1] > 0) & (gt[:, 1] < imb[1])
        errs.append(np.linalg.norm(o["p2d0_i1"].numpy() - gt, axis=1)[inside])
        m = o["matches0"].numpy()
        ok = m > -1
        d = np.linalg.norm(xb.numpy()[m[ok]] - gt[ok], axis=1)
        n_matches.append(int(ok.sum())), n_correct.append(int((d < 3).sum()))
    return errs, n_matches, n_correct, covis


def timed(fn, sync, runs=5):
    fn(), sync()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(), sync()
        t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--neg", nargs="*", default=[], help="real pairs known not to be covisible")
    ap.add_argument("--impl", default="upstream:mps")
    ap.add_argument("--keypoints", type=int, default=2048)
    ap.add_argument("--presets", default="512x9,640x9,384x9,256x9,512x7,512x5,512x3,512x2,512x1,512x0,"
                                         "256x5,256x3,512x9xfp16,256x9xfp16")
    args = ap.parse_args()
    pos = make_pairs(args.images, os.path.join(common.ROOT, "build", "synth"))
    neg = list(zip(args.neg[::2], args.neg[1::2]))
    items = pos + [(a, b, None) for a, b in neg]
    paths = sorted({p for a, b, _ in items for p in (a, b)})
    kp = {p: common.keypoints(p, args.keypoints) for p in paths}
    images = {p: common.load_image(p) for p in paths}
    print(f"{len(pos)} positive pairs, {len(neg)} negative pairs, {args.impl}")
    print("| size | layers | precision | err px | <1px | <3px | matches | prec | covis+ | covis- | describe ms | match ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    impls = {}
    for preset in args.presets.split(","):
        size, layers, *rest = preset.split("x")
        size, layers, fp16 = int(size), int(layers), bool(rest)
        spec = args.impl + (",fp16" if fp16 else "")
        impl = impls.get(spec) or impls.setdefault(spec, common.Impl(spec))
        feats = {p: (impl.describe(images[p], resize=(size, size)), common.size_of(images[p])) for p in paths}
        errs, nm, nc, covis = evaluate(impl, feats, items, layers, kp)
        e = np.concatenate(errs)
        a, b, _ = pos[0]
        t_desc = timed(lambda: impl.describe(images[a], resize=(size, size)), impl.sync)
        t_match = timed(lambda: impl.match(feats[a][0], kp[a], feats[a][1], feats[b][0], kp[b], feats[b][1],
                                           n_layers=layers), impl.sync)
        cp, cn = covis[:len(pos)], covis[len(pos):]
        print(f"| {size} | {layers} | {'fp16' if fp16 else 'fp32'} | {np.median(e):.2f} | {100 * (e < 1).mean():.1f} % "
              f"| {100 * (e < 3).mean():.1f} % | {np.mean(nm):.0f} | {100 * sum(nc) / max(sum(nm), 1):.1f} % "
              f"| {min(cp):.3f} | {max(cn) if cn else float('nan'):.3f} | {t_desc:.0f} | {t_match:.0f} |", flush=True)


if __name__ == "__main__":
    main()
