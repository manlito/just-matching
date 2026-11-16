"""Shared helpers of the dev tools: images, keypoints from just-keypoints, implementation selection."""
import os
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
        else:
            raise ValueError(f"unknown implementation {spec}")

    @torch.no_grad()
    def describe(self, image, **kw):
        return self.mod.describe(self.model, image, **kw)

    @torch.no_grad()
    def match(self, *a, **kw):
        return self.mod.match(self.model, *a, **kw)

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()


def size_of(image):
    return (image.shape[-1], image.shape[-2])


def tmpdir():
    return tempfile.mkdtemp(prefix="jmt-")
