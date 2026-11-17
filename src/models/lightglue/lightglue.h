// LightGlue graph (cvg/LightGlue at eb42fee, features with 128-dim descriptors), written against core/ops.h.
// tools/dump_golden_lightglue.py runs the original and is the specification.
//
// Coordinates are pixels of the original image, given as they are: the keypoints only enter through
// LightGlue's normalization by the image size, and just-keypoints' convention ((0,0) = centre of the
// top-left pixel) is also upstream ALIKED's.
#pragma once

#include <string>
#include <vector>

#include "core/ops.h"
#include "core/weights.h"
#include "models/flexglue/flexglue.h"  // Trace

namespace jmt {

class LightGlue {
 public:
  struct Options {
    int n_layers = 9;               // layers to run, 1..9
    float filter_threshold = 0.1f;  // minimum match score
  };

  // One image's keypoints made ready for matching: the projected descriptors and the rotary encoding of the
  // normalized positions. Made once per image and used by every pair it is in.
  struct Image {
    int n = 0;
    TensorPtr tokens;          // 1 x n x 256: input_proj(descriptors)
    TensorPtr cos, sin;        // 1 x n x 32
    std::vector<float> cos_h, sin_h;  // the same on the host (n x 32), for tests
  };

  struct Result {
    std::vector<int> matches0, matches1;  // per keypoint: index into the other image, or -1
    std::vector<float> scores0, scores1;  // per keypoint: the match score, 0 when unmatched
    std::vector<float> matchability0, matchability1;  // sigmoid of the last layer's matchability
    int stop_layer = 0;                    // layers run
  };

  explicit LightGlue(const Weights& weights);

  // xy: n keypoints (x, y) in pixels of an image of width x height; desc: n x input_dim descriptors.
  Image prepare(Ops& ops, const float* xy, const float* desc, int n, float width, float height,
                const Trace& trace = nullptr, const std::string& tag = "") const;

  void match(Ops& ops, const Image& im0, const Image& im1, const Options& options, Result* out,
             const Trace& trace = nullptr) const;

  int max_layers() const { return n_layers_; }
  int input_dim() const { return input_dim_; }

 private:
  TensorPtr lin(Ops& ops, const std::string& name, const TensorPtr& x, Act act = Act::None,
                const TensorPtr& residual = nullptr) const;
  TensorPtr ffn(Ops& ops, const std::string& block, const TensorPtr& x, const TensorPtr& message) const;
  TensorPtr self_block(Ops& ops, const std::string& p, const TensorPtr& x, const Image& im) const;

  const Weights& w_;
  int n_layers_, heads_, dim_, input_dim_;
};

}  // namespace jmt
