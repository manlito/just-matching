// Released PyTorch checkpoints, read without PyTorch: FlexGlue's flexglue.pth, which users fetch from FlexGlue's
// authors themselves and give to the library as it is, and LightGlue's checkpoints for ALIKED descriptors
// (aliked_lightglue.pth, raco_aliked_lightglue.pth). The result is the same as tools/export_weights.py makes:
// the same tensors, bit for bit, and the same metadata but for the source.
#pragma once

#include <cstddef>

#include "core/weights.h"

namespace jmt {

// True when the blob is a zip archive, the container of torch.save.
bool is_torch_checkpoint(const void* data, size_t size);

// FlexGlue's or LightGlue's checkpoint, told apart by their tensors. Throws std::runtime_error.
Weights weights_from_checkpoint(const void* data, size_t size);

// Reads the state dict of flexglue.pth and applies the export's changes (layer scale folded into pwconv2,
// Wqkv rows grouped as q | k | v, 1x1 convolutions as matrices, names). Throws std::runtime_error.
Weights flexglue_from_checkpoint(const void* data, size_t size);

// Reads the state dict of a LightGlue checkpoint for 128-dim descriptors and applies the export's changes
// (Wqkv rows grouped as q | k | v; the stored confidence thresholds left out). Throws std::runtime_error.
Weights lightglue_from_checkpoint(const void* data, size_t size);

}  // namespace jmt
