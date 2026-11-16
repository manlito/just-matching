// What the CLI and the tests need beyond the public C API.
#pragma once

#include <vector>

#include "core/ops.h"
#include "jmt/jmt.h"

struct jmt_features {
  jmt::TensorPtr tensor;  // (size / 8, size / 8, 256)
  int width = 0, height = 0;  // of the original image
  // Made on first use by the covisibility functions and kept: the L2-normalized mean feature vector
  // (level 1) and the layer-0 search map (level 2, see FlexGlue::search0).
  mutable std::vector<float> global;
  mutable jmt::TensorPtr search0;
};

namespace jmt {

class FlexGlue;

// 8-bit image of any supported format -> height x width x 3 floats in [0, 1].
std::vector<float> to_rgb01(const jmt_image& image);

FlexGlue& model_of(jmt_ctx* ctx);
Ops& ops_of(jmt_ctx* ctx);

}  // namespace jmt
