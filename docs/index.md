# just-matching

Image matching with the FlexGlue network, with minimal dependencies, for bundle adjustment. It also finds
which image pairs are worth matching. No PyTorch, no ONNX Runtime, no cuDNN: one C++ library with hand-written
kernels for each backend. Keypoints come from
[just-keypoints](https://just-keypoints.pages.dev); only their positions are used.

:::{important}
**Use [FlexGlue](https://github.com/cvg/FlexGlue) and [LightGlue](https://github.com/cvg/LightGlue).** This
project is a test bed for batching and GPU kernels, not a replacement for them. It adds nothing to them except
speed: no new networks, training or weights, and only a subset of what they offer (inference, no training). Its one
extra, the shortlist of pairs, is built from FlexGlue's own features. The methods, and the credit for them, are
their authors'. For real work, FlexGlue and LightGlue are the libraries to depend on. LightGlue is not ported yet.
:::

- **Backends**: the CPU (plain C++, the reference), CUDA on NVIDIA GPUs, Metal on Apple Silicon, and
  WebGPU, natively and in the browser through WebAssembly.
- **Same results as the original**: in FP32 every backend returns the original FlexGlue's matches on the
  test pairs, with predicted points within 4e-4 px. FP16 is the fast mode on the GPU backends.
- **Fast**: a pair at full settings (512 px, 9 layers, 2048 keypoints per image) takes 105 ms on an M5 Pro
  (Metal, FP16) and 141 ms on an RTX 2070 laptop (CUDA, FP16), 1.5 times as fast as the original under
  PyTorch on the same GPU.
- **Which pairs to match**: three levels of covisibility, each built on what the one before computed. A mean
  feature vector per image and a coarse grid of points shortlist the pairs; FlexGlue's own score gives the
  verdict while matching, and stops early on pairs it rejects.
- **Your weights, as released**: the library reads FlexGlue's checkpoint, `flexglue.pth`, as its authors
  release it. Its backbone is under Meta's DINOv3 License, so the weights are not distributed with the library.
- **One small C API**: describe each image once, then match any pair of described images. Give it keypoint
  positions and get back matches, a predicted point in the other image for every keypoint, and the pair's
  covisibility.

:::{admonition} Try it in your browser
<a href="demo/index.html">The live demo</a> runs the library on your GPU through WebGPU. Pick a set of images
and it shows which pairs are worth matching, FlexGlue's verdict on each one, the matches, and how long each
step takes. It needs a browser with WebGPU and JavaScript promise integration (Chrome or Edge 137 or later),
and FlexGlue's weights, which you download from its authors and load into the page ([Licences](licences.md)).
:::

```{toctree}
:maxdepth: 2
:caption: Guide

install
quickstart
covisibility
browser
accuracy-and-speed
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
weights-format
references
licences
```
