# References and acknowledgements

just-matching runs a network other people designed and trained, and it is checked against their code. This page
credits that work, the images the demo uses, and the software the library and this site are built with.

## Networks

### FlexGlue

Philipp Lindenberger, Paul-Edouard Sarlin, Ondrej Miksik and Marc Pollefeys. "FlexGlue: Sparse-to-Dense Feature
Matching and Visual Disambiguation." arXiv:2610.32976, 2026. <https://github.com/cvg/FlexGlue>.

- The library runs the released v0.1 checkpoint, rearranged as described in [Weights format](weights-format.md).
  The matcher and the descriptor heads are under the MIT licence.
- The network, the matching and the covisibility score with its early-stop rule follow the original code, and
  every backend is checked against it.

### DINOv3

Oriane Siméoni and others. "DINOv3." arXiv:2508.10104, 2025. <https://github.com/facebookresearch/dinov3>.

- FlexGlue's backbone is a DINOv3 ConvNeXt-B. Its weights are under the DINOv3 License, which applies to anyone
  who uses or passes on the weights ([Licences](licences.md)).

### RaCo and just-keypoints

The keypoints in the demo, and in every measurement on this site, are RaCo's, computed with
[just-keypoints](https://just-keypoints.pages.dev). Abhiram Shenoi, Philipp Lindenberger, Paul-Edouard Sarlin and
Marc Pollefeys. "RaCo: Ranking and Covariance for Practical Learned Keypoints." 3DV 2026.
<https://github.com/cvg/RaCo>. Apache 2.0 licence.

## Images in the demo

- Four frames of the TUM RGB-D benchmark. Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard and
  Daniel Cremers. "A Benchmark for the Evaluation of RGB-D SLAM Systems." IROS 2012.
  <https://cvg.cit.tum.de/data/datasets/rgbd-dataset>. CC BY 4.0.
- Four frames of the KITTI odometry benchmark. Andreas Geiger, Philip Lenz and Raquel Urtasun. "Are we ready for
  Autonomous Driving? The KITTI Vision Benchmark Suite." CVPR 2012. <https://www.cvlibs.net/datasets/kitti/>.
  CC BY-NC-SA 3.0.
- Two views of one place in Berlin, from the sample images of the FlexGlue repository.
- Two images of the LaMAR dataset (the CAB building), from the sample images of the FlexGlue repository. Paul-Edouard
  Sarlin, Mihai Dusmanu and others. "LaMAR: Benchmarking Localization and Mapping for Augmented Reality." ECCV 2022.
  <https://lamar.ethz.ch>.

The TUM and KITTI frames are among the sample images of the ALIKED repository; they were re-encoded as JPEG.

## Software

- **Dawn** and **emdawnwebgpu**, Google's WebGPU implementation and its Emscripten bindings: the WebGPU backend,
  natively and in the browser. <https://dawn.googlesource.com/dawn>. BSD 3-Clause licence.
- **Emscripten**: the browser build. <https://emscripten.org>. MIT and University of Illinois/NCSA licences.
- **stb_image** by Sean Barrett: image decoding in the command-line tool only. Public domain or MIT.
- **PyTorch**: exporting the weights, and running the original network when checking results. It is never a
  dependency of the library.
- **Sphinx** and **MyST-Parser**: this site. The theme follows the documentation site of just-ba, a bundle
  adjustment library by the same authors.
- **Inter** and **IBM Plex Mono**, from Google Fonts: the site's typefaces. SIL Open Font License.
