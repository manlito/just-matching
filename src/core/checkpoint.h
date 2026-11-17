// FlexGlue's released PyTorch checkpoint (flexglue.pth), read without PyTorch: users fetch the weights from
// FlexGlue's authors themselves and give the file to the library as it is. The result is the same as
// tools/export_weights.py makes: the same tensors, bit for bit, and the same metadata.
#pragma once

#include <cstddef>

#include "core/weights.h"

namespace jmt {

// True when the blob is a zip archive, the container of torch.save.
bool is_torch_checkpoint(const void* data, size_t size);

// Reads the state dict of flexglue.pth and applies the export's changes (layer scale folded into pwconv2,
// Wqkv rows grouped as q | k | v, 1x1 convolutions as matrices, names). Throws std::runtime_error.
Weights flexglue_from_checkpoint(const void* data, size_t size);

}  // namespace jmt
