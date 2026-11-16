#!/usr/bin/env python3
"""Evaluate cheap covisibility scores against FlexGlue's own score, on folders of images.

    python tools/covis.py SCENE_DIR... [--single IMAGE ...] [--device mps] [--size 512]

Ground truth is coarse: two images are "covisible" when they are in the same SCENE_DIR. Each --single image is
a scene of its own. This says whether a score separates scenes, not how well it ranks partial overlap.

Scores, all from the features that `describe` already computes (no other network):
    mean      level 1: cosine of the mean feature vector (256-dim)
    vlad      level 1: VLAD of the 64-dim layer-0 descriptors, vocabulary of K centres clustered from these
              images themselves (intra-normalized)
    grid0     level 2: a 16x16 grid of queries through the layer-0 head (no transformer), there and back;
              share of queries that return within one cell (8 resized pixels), mean of both directions
    fgK       level 3: FlexGlue's covisibility score after K transformer layers (K = 1..9)

Reported per score: ROC AUC, average precision, and shortlist recall: the share of covisible pairs kept when
every image keeps its k best partners (a pair is kept when either image keeps it).
Raw scores go to build/covis.npz.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common
import folded

EXT = (".png", ".jpg", ".jpeg")


def kmeans(x, k, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        for j in range(k):
            if (a == j).any():
                c[j] = F.normalize(x[a == j].mean(0), dim=0)
    return c


def vlad(tokens, centres):
    a = (tokens @ centres.T).argmax(1)
    v = torch.zeros(len(centres), tokens.shape[1])
    for j in range(len(centres)):
        if (a == j).any():
            v[j] = F.normalize((tokens[a == j] - centres[j]).sum(0), dim=0)
    return F.normalize(v.flatten(), dim=0)


def grid_points(n, device):
    return folded.cell_centers(n, n, n, n, device)


def grid0(w, cf0, cf1, pts, cell_px=8.0):
    """Layer-0 round trip of a query grid; share of queries that come back within one cell."""
    out = []
    for a, b in ((cf0, cf1), (cf1, cf0)):
        there, _ = folded.head(w, 0, pts, a, b)
        back, _ = folded.head(w, 0, there, b, a)
        size = torch.tensor([8.0 * a.shape[-1], 8.0 * a.shape[-2]], device=pts.device)
        err = ((back - pts) / 2 * size).norm(dim=-1)
        out.append((err < cell_px).float().mean().item())
    return 0.5 * (out[0] + out[1])


def layer_scores(w, f0, f1, n_layers=9):
    """FlexGlue covisibility score after each layer."""
    x0 = folded.linear(w, "input_proj", folded.hwc(f0).flatten(0, 1))
    x1 = folded.linear(w, "input_proj", folded.hwc(f1).flatten(0, 1))
    enc0 = folded.rotary_encoding(w, *f0.shape[-2:], f0.device)
    enc1 = folded.rotary_encoding(w, *f1.shape[-2:], f1.device)
    scores = []
    for k in range(1, n_layers + 1):
        p = f"transformers.{k - 1}"
        x0 = folded.self_block(w, p + ".self_attn", x0, enc0)
        x1 = folded.self_block(w, p + ".self_attn", x1, enc1)
        x0, x1 = folded.cross_block(w, p + ".cross_attn", x0, x1)
        scores.append(folded.covisibility(w, k, x0, x1))
    return scores


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    pos = label.sum()
    return (rank[label].sum() - pos * (pos + 1) / 2) / (pos * (len(label) - pos))


def average_precision(score, label):
    order = np.argsort(-score)
    hit = label[order]
    prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float((prec * hit).sum() / hit.sum())


def shortlist_recall(S, L, k):
    n = len(S)
    keep = np.zeros_like(L)
    for i in range(n):
        s = S[i].copy()
        s[i] = -np.inf
        keep[i, np.argsort(-s)[:k]] = True
    keep |= keep.T
    iu = np.triu_indices(n, 1)
    return (keep[iu] & L[iu]).sum() / L[iu].sum(), keep[iu].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--single", nargs="*", default=[])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--vlad-k", type=int, default=16)
    ap.add_argument("--layers", type=int, default=9, help="0 skips level 3")
    args = ap.parse_args()
    paths, scene = [], []
    for s, d in enumerate(args.scenes):
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            if p.lower().endswith(EXT):
                paths.append(p), scene.append(s)
    for j, p in enumerate(args.single):
        paths.append(p), scene.append(len(args.scenes) + j)
    n = len(paths)
    scene = np.array(scene)
    L = scene[:, None] == scene[None, :]
    w = folded.load(common.WEIGHTS, args.device)
    with torch.no_grad():
        t0 = time.perf_counter()
        feats = [folded.describe(w, common.load_image(p), (args.size, args.size)) for p in paths]
        print(f"{n} images, {len(set(scene))} scenes, describe {1e3 * (time.perf_counter() - t0) / n:.0f} ms/image")
        mean = torch.stack([F.normalize(f.flatten(1).mean(1), dim=0) for f in feats]).cpu()
        tokens = [folded.global_tokens(w, f).cpu() for f in feats]
        centres = kmeans(torch.cat(tokens)[::7], args.vlad_k)
        v = torch.stack([vlad(t, centres) for t in tokens])
        S = {"mean": (mean @ mean.T).numpy(), "vlad": (v @ v.T).numpy(), "grid0": np.eye(n)}
        for k in range(1, args.layers + 1):
            S[f"fg{k}"] = np.eye(n)
        cf = [folded.chw(folded.linear(w, "input_proj", folded.hwc(f))) for f in feats]
        pts = grid_points(16, feats[0].device).to(feats[0])
        t = {"grid0": 0.0, "fg": 0.0}
        for i in range(n):
            for j in range(i + 1, n):
                t0 = time.perf_counter()
                S["grid0"][i, j] = S["grid0"][j, i] = grid0(w, cf[i], cf[j], pts)
                t["grid0"] += time.perf_counter() - t0
                if args.layers:
                    t0 = time.perf_counter()
                    for k, s in enumerate(layer_scores(w, feats[i], feats[j], args.layers), 1):
                        S[f"fg{k}"][i, j] = S[f"fg{k}"][j, i] = s
                    t["fg"] += time.perf_counter() - t0
    iu = np.triu_indices(n, 1)
    n_pairs = len(iu[0])
    print(f"{n_pairs} pairs, {int(L[iu].sum())} covisible; grid0 {1e3 * t['grid0'] / n_pairs:.1f} ms/pair, "
          f"FlexGlue {args.layers} layers {1e3 * t['fg'] / n_pairs:.0f} ms/pair")
    os.makedirs(os.path.join(common.ROOT, "build"), exist_ok=True)
    np.savez(os.path.join(common.ROOT, "build", "covis.npz"), paths=np.array(paths), scene=scene, **S)
    ks = [k for k in (3, 5, 10, 20) if k < n]
    print("| score | AUC | AP | " + " | ".join(f"recall, k={k}" for k in ks) + " |")
    print("|---|---|---|" + "---|" * len(ks))
    for name, M in S.items():
        cells = [f"{100 * shortlist_recall(M, L, k)[0]:.1f} %" for k in ks]
        print(f"| {name} | {auc(M[iu], L[iu]):.4f} | {average_precision(M[iu], L[iu]):.4f} | " + " | ".join(cells) + " |")
    print("pairs kept: " + ", ".join(f"k={k}: {shortlist_recall(S['mean'], L, k)[1]}" for k in ks))


if __name__ == "__main__":
    main()
