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
    matches   number of matches of a and of b, and how many are the same pair
"""
import argparse

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
    args = ap.parse_args()
    if len(args.images) % 2:
        ap.error("images must come in pairs")
    a, b = common.Impl(args.a), common.Impl(args.b)
    resize = (args.size, args.size)
    worst = {"feat": 0.0, "tokens": 0.0, "pos": 0.0, "conf": 0.0, "same": 1.0}
    for p0, p1 in zip(args.images[::2], args.images[1::2]):
        im = [common.load_image(p) for p in (p0, p1)]
        xy = [common.keypoints(p, args.keypoints) for p in (p0, p1)]
        out, feats, stages = [], [], []
        for impl in (a, b):
            f = [impl.describe(i, resize=resize) for i in im]
            st = {}
            out.append(impl.match(f[0], xy[0], common.size_of(im[0]), f[1], xy[1], common.size_of(im[1]),
                                  n_layers=args.layers, stages=st))
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
        print(f"{p0.split('/')[-1]} - {p1.split('/')[-1]}: feat {feat:.2e}  tokens {tokens:.2e}  "
              f"pos max {pos:.2e} median {med:.2e} px  conf {conf:.2e}  "
              f"covis {out[0]['covisibility']:.4f} / {out[1]['covisibility']:.4f}  "
              f"matches {na} / {nb}, same {same}")
        worst["feat"], worst["tokens"] = max(worst["feat"], feat), max(worst["tokens"], tokens if tokens == tokens else 0.0)
        worst["pos"], worst["conf"] = max(worst["pos"], pos), max(worst["conf"], conf)
        worst["same"] = min(worst["same"], same / max(na, nb, 1))
    print(f"worst: feat {worst['feat']:.2e}  tokens {worst['tokens']:.2e}  pos {worst['pos']:.2e} px  "
          f"conf {worst['conf']:.2e}  same matches {100 * worst['same']:.2f} %")


if __name__ == "__main__":
    main()
