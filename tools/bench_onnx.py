#!/usr/bin/env python3
"""Time the upstream ONNX models (third_party/onnx, release v0.1) under ONNX Runtime on one image pair.

    python tools/bench_onnx.py A B [--providers cpu coreml] [--runs 5]

Reports, per provider and matcher file: time to describe one image, time to match the pair (best of --runs,
after one warm-up run), the number of matches and the covisibility score. Keypoints: 2048 RaCo keypoints per
image from the just-keypoints CLI, as in the other tools. 512 px, 9 layers (what the models are exported for).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

import common

sys.path.insert(0, os.path.join(common.ROOT, "third_party", "FlexGlue", "onnx"))
import match_pair as up  # noqa: E402  (upstream's example: describe() and match())

DIR = os.path.join(common.ROOT, "third_party", "onnx")
PROVIDERS = {"cpu": ["CPUExecutionProvider"], "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
             "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"]}


def best(fn, runs):
    out = fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return 1e3 * min(times), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--providers", nargs="+", default=["cpu", "coreml"])
    ap.add_argument("--matchers", nargs="+", default=["flexglue_matcher.onnx", "flexglue_matcher_fp16.onnx"])
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    images = [up.read_image(p) for p in (args.a, args.b)]
    kpts = [common.keypoints(p).numpy() for p in (args.a, args.b)]
    sizes = [(im.shape[1], im.shape[0]) for im in images]
    for name in args.providers:
        try:
            desc = ort.InferenceSession(os.path.join(DIR, "flexglue_descriptor.onnx"), providers=PROVIDERS[name])
            t_desc, f0 = best(lambda: up.describe(desc, images[0]), args.runs)
            f1 = up.describe(desc, images[1])
        except Exception as e:  # a provider that cannot run the model
            print(f"{name}: descriptor failed: {str(e)[:200]}")
            continue
        used = desc.get_providers()[0]
        for matcher in args.matchers:
            try:
                sess = ort.InferenceSession(os.path.join(DIR, matcher), providers=PROVIDERS[name])
                t_match, (m, _, covis) = best(lambda: up.match(sess, f0, f1, kpts[0], kpts[1], sizes[0], sizes[1]),
                                              args.runs)
                print(f"{name} ({used}, matcher on {sess.get_providers()[0]}) {matcher}: describe {t_desc:.0f} ms, "
                      f"match {t_match:.0f} ms, {len(m)} matches, covisibility {covis:.4f}", flush=True)
            except Exception as e:
                print(f"{name} {matcher}: failed: {str(e)[:200]}")


if __name__ == "__main__":
    main()
