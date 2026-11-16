"""Shared helpers of the dev tools: images, keypoints from just-keypoints, implementation selection."""
import os
import shlex
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import torch

import jmtfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JKP = os.environ.get("JKP", os.path.join(ROOT, "..", "just-keypoints", "build", "release", "jkp"))
WEIGHTS = os.path.join(ROOT, "weights", "flexglue.jmtw")


def load_image(path):
    """(3,H,W) float32 RGB in [0,1]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    return torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1].transpose(2, 0, 1))).float() / 255


def keypoints(path, n=2048, model="raco", cache=None):
    """Keypoints (N,2) of an image from the just-keypoints CLI, in FlexGlue's convention: (0,0) is the
    top-left corner of the top-left pixel (just-keypoints puts (0,0) at that pixel's centre: +0.5)."""
    cache = cache or os.path.join(ROOT, "build", "keypoints")
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, f"{os.path.basename(path)}.{model}.{n}.jkpt")
    if not os.path.exists(out):
        cmd = [JKP, "extract", path, "--out", out, "--model", model, "--top-k", str(n)]
        if model == "raco":
            cmd += ["--no-covariance", "--no-ranker"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    _, t = jmtfile.read(out)
    return torch.from_numpy(t["keypoints"].copy()).float() + 0.5


class Impl:
    """One implementation: describe(image) -> features, match(...) -> dict (see folded.match)."""

    def __init__(self, spec):
        kind, _, rest = spec.partition(":")
        opts = rest.split(",") if rest else []
        self.device = next((o for o in opts if o in ("cpu", "mps", "cuda")), "cpu")
        fp16 = "fp16" in opts
        self.spec = spec
        if kind == "upstream":
            import upstream
            self.mod, self.model = upstream, upstream.load_model(self.device, "float16" if fp16 else None)
        elif kind == "folded":
            import folded
            self.mod = folded
            self.model = folded.load(WEIGHTS, self.device, torch.float16 if fp16 else torch.float32)
        elif kind == "cmd":
            self.mod, self.model, self.device, self.cmd = None, None, "cpu", shlex.split(rest)
        else:
            raise ValueError(f"unknown implementation {spec}")

    @torch.no_grad()
    def describe(self, image, **kw):
        if self.mod is None:  # the program describes and matches in one run
            return {"image": image, "size": kw.get("resize", (512, 512))[0]}
        return self.mod.describe(self.model, image, **kw)

    @torch.no_grad()
    def match(self, *a, **kw):
        if self.mod is None:
            return self.run_cmd(*a, **kw)
        return self.mod.match(self.model, *a, **kw)

    def features(self, f):
        """Dense features (256, h, w) of what describe returned (cmd: available after match)."""
        return (f["features"] if self.mod is None else f).float().cpu()

    def run_cmd(self, f0, xy0, size0, f1, xy1, size1, n_layers=9, stages=None, **kw):
        """<cmd> A.png B.png --kpts-a A.jkpt --kpts-b B.jkpt --out OUT --size S --layers L --dump-features.
        Images go as lossless PNG, so both sides see the same pixels; keypoints in the just-keypoints
        convention (centre of the top-left pixel = (0,0))."""
        d = tmpdir()
        paths = []
        for i, (f, xy) in enumerate(((f0, xy0), (f1, xy1))):
            rgb = (f["image"] * 255).round().byte().permute(1, 2, 0).numpy()
            cv2.imwrite(os.path.join(d, f"{i}.png"), np.ascontiguousarray(rgb[..., ::-1]))
            jmtfile.write(os.path.join(d, f"{i}.jkpt"), {"kind": "keypoints"}, {"keypoints": (xy - 0.5).numpy()})
            paths.append(os.path.join(d, f"{i}"))
        out = os.path.join(d, "out.jmtt")
        cmd = self.cmd + [paths[0] + ".png", paths[1] + ".png", "--kpts-a", paths[0] + ".jkpt", "--kpts-b",
                          paths[1] + ".jkpt", "--out", out, "--size", str(f0["size"]), "--layers", str(n_layers),
                          "--dump-features"]
        for key, flag in (("filter_threshold", "--filter"), ("max_distance_px", "--max-px"),
                          ("min_covisibility", "--min-covisibility"), ("early_stop_decay", "--early-stop")):
            if kw.get(key) is not None:
                cmd += [flag, str(kw[key])]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        _, t = jmtfile.read(out)
        t = {k: torch.from_numpy(v.copy()) for k, v in t.items()}
        f0["features"], f1["features"] = t["features_a"].permute(2, 0, 1), t["features_b"].permute(2, 0, 1)
        matches0 = torch.full((len(xy0),), -1, dtype=torch.long)
        scores0 = torch.zeros(len(xy0))
        idx = t["matches"].long()
        matches0[idx[:, 0]], scores0[idx[:, 0]] = idx[:, 1], t["scores"]
        shutil.rmtree(d)
        return {"matches0": matches0, "scores0": scores0, "p2d0_i1": t["a_in_b"] + 0.5, "p2d1_i0": t["b_in_a"] + 0.5,
                "confidence0": t["conf_a"], "confidence1": t["conf_b"],
                "covisibility": float(t["covisibility"][0]), "stop_layer": int(t["stop_layer"][0])}

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()


def size_of(image):
    return (image.shape[-1], image.shape[-2])


def tmpdir():
    return tempfile.mkdtemp(prefix="jmt-")
