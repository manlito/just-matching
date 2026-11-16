"""Runs the original FlexGlue (third_party/FlexGlue) with the same interface as folded.py.

    m = load_model(device="cpu", mp=None)            # mp: None = float32 reference, "float16" = upstream default
    f = describe(m, image)                           # (3,H,W) in [0,1] -> features (256,h,w)
    out = match(m, f0, xy0, size0, f1, xy1, size1)   # same dict as folded.match
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "FlexGlue"))
CHECKPOINT = os.path.join(ROOT, "third_party", "flexglue.pth")


def load_model(device="cpu", mp=None):
    from flexglue import FlexGlue
    return FlexGlue(weights=CHECKPOINT, mp=mp).eval().to(device)


def device_of(m):
    return next(m.parameters()).device


@torch.no_grad()
def describe(m, image, resize=(512, 512), stages=None):
    f = m.descriptor.describe(image.to(device_of(m)), resize=resize)["features"][0]
    if stages is not None:
        stages["features"] = f.float().cpu()
    return f


@torch.no_grad()
def match(m, f0, xy0, size0, f1, xy1, size1, n_layers=9, filter_threshold=0.1, max_distance_px=3.0,
          min_covisibility=0.0, early_stop_decay=None, stages=None):
    dev = device_of(m)
    data = [{"features": f[None], "keypoints": xy[None].to(dev, torch.float32),
             "image_size": torch.tensor([size], device=dev, dtype=torch.float32)}
            for f, xy, size in ((f0, xy0, size0), (f1, xy1, size1))]
    hooks = []
    if stages is not None:
        for k, block in enumerate(m.transformers, 1):
            def hook(_, __, out, k=k):
                stages[f"tokens0_l{k}"], stages[f"tokens1_l{k}"] = out[0][0].float().cpu(), out[1][0].float().cpu()
            hooks.append(block.register_forward_hook(hook))
    try:
        pred = m.match(data[0], data[1], sparse=True, n_layers=n_layers, filter_threshold=filter_threshold,
                       max_distance_px=max_distance_px, min_covisibility=min_covisibility,
                       early_stop_decay=early_stop_decay)
    finally:
        for h in hooks:
            h.remove()
    return {
        "matches0": pred["matches0"][0].cpu(), "scores0": pred["matching_scores0"][0].float().cpu(),
        "p2d0_i1": pred["p2d0_i1"][0].float().cpu(), "p2d1_i0": pred["p2d1_i0"][0].float().cpu(),
        "confidence0": pred["confidence0"][0].float().cpu(), "confidence1": pred["confidence1"][0].float().cpu(),
        "covisibility": float(pred["covisibility_score"][0]), "stop_layer": int(pred["stop_layer"]),
    }
