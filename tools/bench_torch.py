#!/usr/bin/env python3
"""Time of the original FlexGlue under PyTorch on the pair tools/bench.sh uses, for the benchmark table.

    python tools/bench_torch.py [--impl upstream:mps] [--presets 512x9,512x5,384x9] [--runs 8]

Per preset (size x layers), in FP32 and FP16: describe per image and match per pair, best and median of
the warm runs, 2048 RaCo keypoints per image.
"""
import argparse
import os
import time

import numpy as np

import common


def timed(fn, sync, runs):
    fn(), sync()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(), sync()
        t.append(1e3 * (time.perf_counter() - t0))
    return min(t), float(np.median(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="upstream:mps")
    ap.add_argument("--presets", default="512x9,512x5,384x9")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--keypoints", type=int, default=2048)
    args = ap.parse_args()
    assets = os.path.join(common.ROOT, "third_party", "FlexGlue", "assets")
    paths = [os.path.join(assets, "berlin1.jpg"), os.path.join(assets, "berlin2.jpg")]
    images = [common.load_image(p) for p in paths]
    kp = [common.keypoints(p, args.keypoints) for p in paths]
    sizes = [common.size_of(i) for i in images]
    print("| size, layers | precision | describe best / median | match best / median |")
    print("|---|---|---|---|")
    for fp16 in (False, True):
        impl = common.Impl(args.impl + (",fp16" if fp16 else ""))
        for preset in args.presets.split(","):
            size, layers = (int(v) for v in preset.split("x"))
            f = [impl.describe(i, resize=(size, size)) for i in images]
            d = timed(lambda: impl.describe(images[0], resize=(size, size)), impl.sync, args.runs)
            m = timed(lambda: impl.match(f[0], kp[0], sizes[0], f[1], kp[1], sizes[1], n_layers=layers), impl.sync, args.runs)
            print(f"| {size}, {layers} | {'FP16' if fp16 else 'FP32'} | {d[0]:.0f} / {d[1]:.0f} ms | {m[0]:.0f} / {m[1]:.0f} ms |",
                  flush=True)


if __name__ == "__main__":
    main()
