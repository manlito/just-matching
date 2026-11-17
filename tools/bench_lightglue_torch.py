#!/usr/bin/env python3
"""Time of the original LightGlue under PyTorch, for the M9 speed table.

    python tools/bench_lightglue_torch.py A.jkpt B.jkpt [--device mps|cuda|cpu] [--model aliked]

The keypoint files are just-keypoints outputs with descriptors (as tools/dump_golden_lightglue.py caches
them in build/keypoints). Prints the best of --runs runs per pair, at fixed depth and with upstream's
adaptive defaults, in float32 and in float16 (autocast, as upstream's `mp` option does on CUDA).
"""
import argparse
import time

import torch

import dump_golden_lightglue as d
import jmtfile


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", choices=sorted(d.WEIGHTS), default="aliked")
    ap.add_argument("--runs", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)
    data = {}
    for i, path in enumerate((args.a, args.b)):
        meta, t = jmtfile.read(path)
        data[f"image{i}"] = {"keypoints": torch.from_numpy(t["keypoints"].copy())[None].to(device),
                             "descriptors": torch.from_numpy(t["descriptors"].copy())[None].to(device),
                             "image_size": torch.tensor([[float(meta["width"]), float(meta["height"])]], device=device)}
    for adaptive in (False, True):
        conf = {} if adaptive else {"depth_confidence": -1, "width_confidence": -1}
        net = d.load_model(args.model, **conf)
        net.conf.flash = True
        net = net.to(device)
        for fp16 in (False, True):
            best, out = float("inf"), None
            for _ in range(args.runs):
                sync(device)
                t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                    out = net(data)
                sync(device)
                best = min(best, time.perf_counter() - t0)
            n = int((out["matches0"] > -1).sum())
            print(f"{args.device} {'fp16' if fp16 else 'fp32'} {'adaptive' if adaptive else 'fixed   '}: {best * 1e3:.0f} ms, "
                  f"{int(out['stop'])} layers, {n} matches")


if __name__ == "__main__":
    main()
