// What the CLI and the tests need beyond the public C API.
#pragma once

#include <memory>
#include <vector>

#include "core/ops.h"
#include "jmt/jmt.h"
#include "models/lightglue/lightglue.h"

struct jmt_features {
  jmt::TensorPtr tensor;  // FlexGlue: (size / 8, size / 8, 256)
  std::shared_ptr<const jmt::LightGlue::Image> keypoints;  // LightGlue: what jmt_describe_keypoints made
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

FlexGlue& model_of(jmt_ctx* ctx);  // throws on a LightGlue context
bool is_lightglue(const jmt_ctx* ctx);
Ops& ops_of(jmt_ctx* ctx);

}  // namespace jmt
