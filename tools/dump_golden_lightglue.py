#!/usr/bin/env python3
"""Golden test vectors from the original LightGlue (third_party/LightGlue, tools/fetch_lightglue.sh), for the
C++ backends.

    python tools/dump_golden_lightglue.py A B [--model aliked|raco-aliked] [--keypoints 2048]
        -> build/golden/lightglue-<model>/<A>-<B>.jmtt

Keypoints and 128-dim descriptors come from the just-keypoints CLI: aliked-n16 alone, or RaCo keypoints
described by aliked-n16. Coordinates are just-keypoints' (0,0) = centre of the top-left pixel, which is also
upstream ALIKED's. LightGlue runs on the CPU in FP32.

One file per pair, in the weights container format (jmtfile.py). Two runs:
  - fixed depth (adaptivity off): the inputs, the encoding, the tokens after the input projection and after
    every layer, the log assignment of the last layer, matchability, and matches and scores;
  - adaptive (upstream defaults, depth 0.95 / width 0.99): stop layer, the points kept, matches and scores.
Files are not committed.
"""
import argparse
import importlib.util
import os
import subprocess

import numpy as np
import torch

import common
import jmtfile

LG_DIR = os.path.join(common.ROOT, "third_party", "LightGlue")
WEIGHTS = {m: os.path.join(common.ROOT, "third_party", "lightglue-weights", f"{m.replace('-', '_')}_lightglue.pth")
           for m in ("aliked", "raco-aliked")}


def load_lightglue_module():
    """lightglue/lightglue.py alone: the package's __init__ imports the extractors, which need kornia."""
    spec = importlib.util.spec_from_file_location("lightglue_upstream", os.path.join(LG_DIR, "lightglue", "lightglue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(model, **conf):
    lg = load_lightglue_module()
    net = lg.LightGlue(features=None, input_dim=128, flash=False, **conf).eval()
    sd = torch.load(WEIGHTS[model], map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "confidence_thresholds"]  # a buffer, recomputed in __init__
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
    return net


def features(path, model, n):
    """(xy (N,2), desc (N,128), (width, height)) from the just-keypoints CLI, cached in build/keypoints."""
    cache = os.path.join(common.ROOT, "build", "keypoints")
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, f"{os.path.basename(path)}.{model}.{n}.jkpt")
    if not os.path.exists(out):
        cmd = [common.JKP, "extract", path, "--out", out, "--top-k", str(n)]
        cmd += ["--model", "raco", "--descriptor", "aliked-n16", "--no-covariance", "--no-ranker"] \
            if model == "raco-aliked" else ["--model", "aliked-n16"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    meta, t = jmtfile.read(out)
    return t["keypoints"].copy(), t["descriptors"].copy(), (int(meta["width"]), int(meta["height"]))


def run(net, xy, desc, size, stages=None):
    data = {f"image{i}": {"keypoints": torch.from_numpy(xy[i])[None], "descriptors": torch.from_numpy(desc[i])[None],
                          "image_size": torch.tensor([size[i]], dtype=torch.float32)} for i in range(2)}
    hooks = []
    if stages is not None:
        def keep(name):
            return lambda _m, _i, o: stages.__setitem__(name, o)
        hooks.append(net.input_proj.register_forward_hook(
            lambda _m, i, o: stages.setdefault("input_proj0", o) if "input_proj0" not in stages else stages.__setitem__("input_proj1", o)))
        hooks.append(net.posenc.register_forward_hook(
            lambda _m, i, o: stages.setdefault("encoding0", o) if "encoding0" not in stages else stages.__setitem__("encoding1", o)))
        for k, layer in enumerate(net.transformers):
            hooks.append(layer.register_forward_hook(keep(f"layer{k}")))
        for k, head in enumerate(net.log_assignment):
            hooks.append(head.register_forward_hook(keep(f"assignment{k}")))
    with torch.no_grad():
        out = net(data)
    for h in hooks:
        h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--model", choices=sorted(WEIGHTS), default="aliked")
    ap.add_argument("--keypoints", type=int, default=2048)
    ap.add_argument("--out")
    args = ap.parse_args()

    xy, desc, size = zip(*(features(p, args.model, args.keypoints) for p in (args.a, args.b)))
    tensors = {}
    for i in range(2):
        tensors[f"keypoints{i}"], tensors[f"descriptors{i}"] = xy[i], desc[i]
        tensors[f"image_size{i}"] = np.float32(size[i])

    fixed = load_model(args.model, depth_confidence=-1, width_confidence=-1)
    st = {}
    out = run(fixed, xy, desc, size, st)
    for i in range(2):
        tensors[f"pair.input_proj{i}"] = st[f"input_proj{i}"][0].numpy()
        tensors[f"pair.encoding{i}"] = st[f"encoding{i}"][:, 0].numpy()  # (2, N, 64): cos, sin
    for k in range(fixed.conf.n_layers):
        d0, d1 = st[f"layer{k}"]
        tensors[f"pair.layer{k}.desc0"], tensors[f"pair.layer{k}.desc1"] = d0[0].numpy(), d1[0].numpy()
    last = fixed.conf.n_layers - 1
    tensors["pair.log_assignment"] = st[f"assignment{last}"][0].numpy()  # (N0+1, N1+1)
    d0, d1 = st[f"layer{last}"]
    head = fixed.log_assignment[last]
    with torch.no_grad():
        tensors["pair.matchability0"] = head.get_matchability(d0)[0].numpy()
        tensors["pair.matchability1"] = head.get_matchability(d1)[0].numpy()
    tensors["out.matches0"] = out["matches0"][0].int().numpy()
    tensors["out.matches1"] = out["matches1"][0].int().numpy()
    tensors["out.matching_scores0"] = out["matching_scores0"][0].numpy()
    tensors["out.matching_scores1"] = out["matching_scores1"][0].numpy()

    adaptive = load_model(args.model)
    out_a = run(adaptive, xy, desc, size)
    tensors["adaptive.matches0"] = out_a["matches0"][0].int().numpy()
    tensors["adaptive.matching_scores0"] = out_a["matching_scores0"][0].numpy()
    tensors["adaptive.stop"] = np.int32([int(out_a["stop"])])
    tensors["adaptive.prune0"] = out_a["prune0"][0].int().numpy()  # layer at which each point was dropped
    tensors["adaptive.prune1"] = out_a["prune1"][0].int().numpy()

    name = "-".join(os.path.splitext(os.path.basename(p))[0] for p in (args.a, args.b))
    path = args.out or os.path.join(common.ROOT, "build", "golden", f"lightglue-{args.model}", f"{name}.jmtt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"kind": "golden", "model": f"lightglue-{args.model}", "layers": fixed.conf.n_layers,
            "filter_threshold": fixed.conf.filter_threshold, "keypoints": args.keypoints,
            "source_commit": "eb42fee2d71449efb0aa5c10549752b5d75384d8"}
    jmtfile.write(path, meta, tensors)
    n_fixed, n_adapt = int((out["matches0"] > -1).sum()), int((out_a["matches0"] > -1).sum())
    kept = [int((out_a[f"prune{i}"][0] == int(out_a["stop"])).sum()) for i in range(2)]
    print(f"{path}: {os.path.getsize(path) / 1e6:.0f} MB; fixed depth {n_fixed} matches; adaptive: stop after "
          f"{int(out_a['stop'])} layers, {kept[0]} + {kept[1]} points kept to the end, {n_adapt} matches")


if __name__ == "__main__":
    main()
