# just-matching

FlexGlue and LightGlue image matching in C++ with minimal dependencies, for Python. No PyTorch: the library
reads the released checkpoints itself and runs on the CPU, on Apple GPUs (Metal) and on NVIDIA GPUs (CUDA).

| Package | Platforms | Backends |
|---|---|---|
| `just-matching` | Linux x86_64, macOS arm64 (13 or later), Windows x86_64 | CPU; Metal on macOS |
| `just-matching-cu12` | Linux x86_64, Windows x86_64 | CPU, CUDA 12 (NVIDIA driver R525 or later) |
| `just-matching-cu13` | Linux x86_64, Windows x86_64 | CPU, CUDA 13 (NVIDIA driver R580 or later) |

All three provide the module `just_matching`; install one. The CUDA packages need only the NVIDIA driver (the
CUDA runtime is built in) and a GPU of compute capability 7.5 (Turing) or later. Python 3.12 or later.

```bash
pip install just-matching        # or just-matching-cu12, just-matching-cu13
```

## Weights

No weights are in the packages. Download them from their authors and give the file (or its bytes) to `Matcher`;
the library reads the checkpoints as released, without PyTorch.

**LightGlue** for ALIKED descriptors, under the Apache License 2.0 (Computer Vision and Geometry Lab), from
[LightGlue's v0.1_arxiv release](https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv):

| File | Model | SHA-256 |
|---|---|---|
| `aliked_lightglue.pth` | `lightglue-aliked`: ALIKED keypoints and descriptors | `d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503` |
| `raco_aliked_lightglue.pth` | `lightglue-raco-aliked`: RaCo keypoints described by ALIKED | `a804c3fe2851cc3c765cda6a9f412845893c8d8e2a3048f848d7f6ba172e6312` |

**FlexGlue**: `flexglue.pth` (567 MB) from
[FlexGlue's v0.1 release](https://github.com/cvg/FlexGlue/releases/tag/v0.1), SHA-256
`d7371f743b8cdc4adf723c2aa05b181587fe4bc177fced139697975dff790068`. Its matcher and descriptor heads are MIT;
its backbone is Meta's DINOv3 ConvNeXt-B, under the
[DINOv3 License](https://github.com/cvg/FlexGlue/blob/main/DINOv3_LICENSE.md), which you accept by using the
file: read it first. just-matching never distributes these weights.

The keypoints and descriptors LightGlue matches come from
[just-keypoints](https://github.com/manlito/just-keypoints) (`aliked-n16`, alone or describing RaCo's
keypoints).

## LightGlue

```python
import just_matching as jmt

m = jmt.Matcher("aliked_lightglue.pth")                   # backend="cuda" or "metal" for the GPU
a = m.describe_keypoints(width_a, height_a, xy_a, desc_a)  # xy (n, 2), desc (n, 128) float32
b = m.describe_keypoints(width_b, height_b, xy_b, desc_b)
r = m.match(a, b)
r.idx      # (n, 2) int32: keypoint of a, keypoint of b
r.score    # (n,) in (0, 1]
```

## FlexGlue

FlexGlue describes whole images once, then matches any keypoints of them and predicts, for every keypoint, where
it lands in the other image. It also says which pairs of a set are worth matching.

```python
m = jmt.Matcher("flexglue.pth", backend="cuda", precision="fp16")
a, b = m.describe(image_a), m.describe(image_b)  # uint8 (h, w), (h, w, 3) RGB or (h, w, 4) RGBA
r = m.match(a, b, xy_a, xy_b)
r.a_in_b, r.conf_a, r.covisibility

features = [m.describe(img) for img in images]
pairs, scores = m.shortlist(features, k=10, neighbours=2)  # (n, 2) image indices, best first
m.covisibility(a, b, level=2)
```

`match_batch([(a, b), (c, d, xy_c, xy_d), ...])` matches many pairs in one call (pipelined on a GPU). Options of
`match` and `match_batch` (`n_layers`, `filter_threshold`, `depth_confidence`, `min_covisibility`, ...) are
keyword arguments, documented in `help(jmt.Matcher.match)`.

Pixel coordinates are those of the original image: x right, y down, (0, 0) the centre of the top-left pixel.
A `Matcher` is safe to share between threads (calls take turns); separate Matchers run in parallel.

`jmt.compiled_backends()` lists the backends of the installed package, `jmt.VARIANT` and `jmt.CUDA_VERSION` say
which package it is.
