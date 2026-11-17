#!/usr/bin/env python3
"""Compare two FlexGlue implementations on image pairs.

    python tools/compare.py A1 B1 [A2 B2 ...] [--a upstream] [--b folded] [--layers 9] [--size 512]

Implementations (--a / --b):
    upstream[:cpu|mps|cuda][,fp16]   the original repository under PyTorch
    folded[:cpu|mps|cuda][,fp16]     reference forward pass from weights/flexglue.jmtw (tools/folded.py)
    cmd:<command>                    external program, e.g. "cmd:build/release/jmt match"; it is run as
                                     <command> A.png B.png --kpts-a A.jkpt --kpts-b B.jkpt --out OUT.jmtt
                                     --size S --layers L --dump-features (tokens are not compared)

Keypoints come from the just-keypoints CLI (RaCo, --keypoints N per image; $JKP overrides its path).
Reported per pair, a against b:
    feat      largest absolute difference of the dense features (their values are ~1)
    tokens    largest absolute difference of the tokens after the last layer (values ~1-10)
    pos       largest / median difference of the predicted points, in image pixels, over confident queries
    conf      largest difference of the query confidences
    covis     covisibility score of a and of b
    layer     layers run by a and by b (fewer than --layers when --early-stop stops the pair)
    matches   number of matches of a and of b, and how many are the same pair

--save DIR writes the matches of each implementation as text, one match per line (keypoint index in A,
keypoint index in B, their pixel positions in FlexGlue's convention, the score), to
DIR/<A>-<B>.<size>x<layers>.a.txt and .b.txt; `diff` on the two files without the score column shows
exactly which matches differ (see docs/reproducing.md).
"""
import argparse
import os

import torch

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--a", default="upstream")
    ap.add_argument("--b", default="folded")
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=2048)
    ap.add_argument("--min-covisibility", type=float, default=0.0)
    ap.add_argument("--early-stop", type=float, help="early_stop_decay, in [0, 1); needs --min-covisibility")
    ap.add_argument("--save", help="directory for the match lists of both implementations, as text")
    args = ap.parse_args()
    if len(args.images) % 2:
        ap.error("images must come in pairs")
    a, b = common.Impl(args.a), common.Impl(args.b)
    resize = (args.size, args.size)
    worst = {"feat": 0.0, "tokens": 0.0, "pos": 0.0, "conf": 0.0, "same": 1.0, "layers": 0}
    for p0, p1 in zip(args.images[::2], args.images[1::2]):
        im = [common.load_image(p) for p in (p0, p1)]
        xy = [common.keypoints(p, args.keypoints) for p in (p0, p1)]
        out, feats, stages = [], [], []
        for impl in (a, b):
            f = [impl.describe(i, resize=resize) for i in im]
            st = {}
            out.append(impl.match(f[0], xy[0], common.size_of(im[0]), f[1], xy[1], common.size_of(im[1]),
                                  n_layers=args.layers, min_covisibility=args.min_covisibility,
                                  early_stop_decay=args.early_stop, stages=st))
            feats.append([impl.features(x) for x in f])
            stages.append(st)
        feat = max((feats[0][i] - feats[1][i]).abs().max().item() for i in (0, 1))
        key = f"tokens0_l{args.layers}"
        both = args.layers and key in stages[0] and key in stages[1]
        tokens = (stages[0][key] - stages[1][key]).abs().max().item() if both else float("nan")
        sure = out[0]["confidence0"] > 0.1
        d = (out[0]["p2d0_i1"] - out[1]["p2d0_i1"]).norm(dim=-1)[sure]
        pos, med = (d.max().item(), d.median().item()) if len(d) else (0.0, 0.0)
        conf = max((out[0][k] - out[1][k]).abs().max().item() for k in ("confidence0", "confidence1"))
        ma, mb = out[0]["matches0"], out[1]["matches0"]
        na, nb = int((ma > -1).sum()), int((mb > -1).sum())
        same = int(((ma == mb) & (ma > -1)).sum())
        if args.save:
            save_matches(args.save, f"{args.size}x{args.layers}", p0, p1, out, xy)
        print(f"{p0.split('/')[-1]} - {p1.split('/')[-1]}: feat {feat:.2e}  tokens {tokens:.2e}  "
              f"pos max {pos:.2e} median {med:.2e} px  conf {conf:.2e}  "
              f"covis {out[0]['covisibility']:.4f} / {out[1]['covisibility']:.4f}  "
              f"layer {out[0]['stop_layer']} / {out[1]['stop_layer']}  "
              f"matches {na} / {nb}, same {same}")
        worst["feat"], worst["tokens"] = max(worst["feat"], feat), max(worst["tokens"], tokens if tokens == tokens else 0.0)
        worst["pos"], worst["conf"] = max(worst["pos"], pos), max(worst["conf"], conf)
        worst["same"] = min(worst["same"], same / max(na, nb) if max(na, nb) else 1.0)
        worst["layers"] += out[0]["stop_layer"] != out[1]["stop_layer"]
    print(f"worst: feat {worst['feat']:.2e}  tokens {worst['tokens']:.2e}  pos {worst['pos']:.2e} px  "
          f"conf {worst['conf']:.2e}  same matches {100 * worst['same']:.2f} %  "
          f"stop layer differs on {worst['layers']} pairs")


def save_matches(directory, setting, p0, p1, out, xy):
    os.makedirs(directory, exist_ok=True)
    name = f"{os.path.splitext(os.path.basename(p0))[0]}-{os.path.splitext(os.path.basename(p1))[0]}.{setting}"
    for tag, o in zip("ab", out):
        m, sc = o["matches0"], o["scores0"]
        with open(os.path.join(directory, f"{name}.{tag}.txt"), "w") as f:
            f.write("# idx_a idx_b x_a y_a x_b y_b score\n")
            for i in torch.nonzero(m > -1).flatten().tolist():
                j = int(m[i])
                f.write(f"{i} {j} {xy[0][i, 0]:.2f} {xy[0][i, 1]:.2f} {xy[1][j, 0]:.2f} {xy[1][j, 1]:.2f} {float(sc[i]):.4f}\n")


if __name__ == "__main__":
    main()
