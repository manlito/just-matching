"""just-matching: FlexGlue and LightGlue image matching with minimal dependencies.

The compiled core is in ``just_matching._core``; everything in it is re-exported here.

LightGlue matches keypoints and descriptors from just-keypoints (ALIKED, alone or describing RaCo's keypoints)::

    import just_matching as jmt

    m = jmt.Matcher("aliked_lightglue.pth")           # the checkpoint as released, read without PyTorch
    a = m.describe_keypoints(w_a, h_a, xy_a, desc_a)  # xy (n, 2), desc (n, 128)
    b = m.describe_keypoints(w_b, h_b, xy_b, desc_b)
    r = m.match(a, b)                                 # r.idx (n, 2), r.score (n,)

FlexGlue describes whole images and predicts where each keypoint lands in the other image::

    m = jmt.Matcher("flexglue.pth", backend="cuda")
    a, b = m.describe(image_a), m.describe(image_b)   # uint8 (h, w), (h, w, 3) or (h, w, 4)
    r = m.match(a, b, xy_a, xy_b)
    pairs, scores = m.shortlist([a, b, ...], k=10)    # which pairs are worth matching

Pixel coordinates: x right, y down, (0, 0) the centre of the top-left pixel (as just-keypoints). No weights are
in the package; see the README for where to get them and their licences.
"""

from __future__ import annotations

from ._build_info import CUDA_VERSION, VARIANT, __version__
from ._core import *  # noqa: F401,F403
from ._core import Features, Matcher, Matches, backend_available, compiled_backends

__all__ = [
    "CUDA_VERSION",
    "VARIANT",
    "__version__",
    "Features",
    "Matcher",
    "Matches",
    "backend_available",
    "compiled_backends",
]
