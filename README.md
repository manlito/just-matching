# just-matching

Image matching (FlexGlue) and covisibility with minimal dependencies, the companion of
[just-keypoints](../just-keypoints). No PyTorch, no ONNX Runtime at run time. Plan and status: [PLAN.md](PLAN.md).

Status: the reference is locked (M1). The weights export, a standalone reference forward pass that matches the
original FlexGlue, golden dumps and the measurement tools exist in `tools/`. There is no C++ library yet.

See [tools/README.md](tools/README.md) for how to fetch the original, export the weights and run the checks.

## Licenses

Code in this repository: to be decided. FlexGlue's code and matcher weights are MIT (`weights/LICENSE-FLEXGLUE`);
its backbone weights are under the DINOv3 License (`weights/LICENSE-DINOv3.md`). No weights are committed.
