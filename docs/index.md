# just-matching

Image matching for bundle adjustment with minimal dependencies, with two matchers: LightGlue on ALIKED
descriptors, whose weights come with the library, and FlexGlue, which also tells look-alike places apart and
finds which image pairs are worth matching. No PyTorch, no ONNX Runtime, no cuDNN: one C++ library with
hand-written kernels for each backend. Keypoints come from [just-keypoints](https://just-keypoints.pages.dev).

- **Backends**: the CPU (plain C++, the reference), CUDA on NVIDIA GPUs, Metal on Apple Silicon, and
  WebGPU, natively and in the browser through WebAssembly.
- **Two matchers**: LightGlue matches just-keypoints' ALIKED keypoints and descriptors; its weights are
  Apache-2.0 and can be built into the library. FlexGlue reads the images themselves and judges whether two
  images show the same place, which keeps look-alike places apart; its weights come from you.
- **Same results as the original**: in FP32 every backend returns the original FlexGlue's and LightGlue's
  matches on the test pairs. FP16 is the fast mode on the GPU backends.
- **Fast**: LightGlue matches a pair of 2048 keypoints per image in 16 ms on an M5 Pro (Metal, FP16) and in
  18 ms on an RTX 2070 laptop (CUDA, FP16), and less when it stops early on an easy pair: in FP16, 4 to 5 times
  as fast as the original under PyTorch on the same GPU. FlexGlue at full settings (512 px, 9 layers) takes 105 and 141 ms, 1.5 times as
  fast as the original.
- **Which pairs to match**: three levels of covisibility, each built on what the one before computed. A mean
  feature vector per image and a coarse grid of points shortlist the pairs; FlexGlue's own score gives the
  verdict while matching, and stops early on pairs it rejects.
- **Weights as released**: the library reads the checkpoints as their authors release them, without
  PyTorch. FlexGlue's backbone is under Meta's DINOv3 License, so FlexGlue's weights are not distributed with
  the library: you download `flexglue.pth` yourself.
- **One small C API**: describe each image once (its pixels for FlexGlue, its keypoints and descriptors for
  LightGlue), then match any pair of described images.

:::{admonition} Try it in your browser
<a href="demo/index.html">The live demo</a> runs the library on your GPU through WebGPU. Pick a set of images
and it matches every pair with LightGlue, or, with FlexGlue, shows which pairs are worth matching, FlexGlue's
verdict on each one and the matches; with how long each step takes. It needs a browser with WebGPU and
JavaScript promise integration (Chrome or Edge 137 or later). LightGlue's weights come with the page; FlexGlue's
you download from its authors and load into the page ([Licences](licences.md)).
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
