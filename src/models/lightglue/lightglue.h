// LightGlue graph (cvg/LightGlue at eb42fee, features with 128-dim descriptors), written against core/ops.h.
// The same graph runs LighterGlue (verlab/accelerated_features at e92685f): XFeat's 64-dim descriptors, 6 layers
// of width 96 with one head; the sizes come from the weights file.
// tools/dump_golden_lightglue.py runs the original and is the specification.
//
// Coordinates are pixels of the original image, given as they are: the keypoints only enter through
// LightGlue's normalization by the image size, and just-keypoints' convention ((0,0) = centre of the
// top-left pixel) is also upstream ALIKED's.
#pragma once

#include <limits>
#include <string>
#include <vector>

#include "core/ops.h"
#include "core/weights.h"
#include "models/flexglue/flexglue.h"  // Trace

namespace jmt {

class LightGlue {
 public:
  struct Options {
    int n_layers = 9;               // layers to run, from 1; more than the model has (9, LighterGlue 6): all
    float filter_threshold = 0.1f;  // minimum match score
    // Adaptive depth: after each layer but the last, stop when more than this share of the keypoints are
    // confident (upstream's depth_confidence). <= 0: off. NaN: the model's own (LightGlue 0.95, LighterGlue
    // off), from the weights file.
    float depth_confidence = std::numeric_limits<float>::quiet_NaN();
    // Adaptive width: drop confident keypoints whose matchability is at most 1 - this (upstream's
    // width_confidence) before the next layer. <= 0: off. NaN: the model's own (LightGlue 0.99, LighterGlue
    // 0.95).
    float width_confidence = std::numeric_limits<float>::quiet_NaN();
  };

  // One image's keypoints made ready for matching: the projected descriptors and the rotary encoding of the
  // normalized positions. Made once per image and used by every pair it is in.
  struct Image {
    int n = 0;
    TensorPtr tokens;          // 1 x n x dim: input_proj(descriptors)
    TensorPtr cos, sin;        // 1 x n x (dim / heads / 2)
    std::vector<float> cos_h, sin_h;  // the same on the host (n x 32), for tests
  };

  struct Result {
    std::vector<int> matches0, matches1;  // per keypoint: index into the other image, or -1
    std::vector<float> scores0, scores1;  // per keypoint: the match score, 0 when unmatched
    // Sigmoid of the matchability at the last layer each keypoint took part in (dropped keypoints keep the
    // value they were dropped with).
    std::vector<float> matchability0, matchability1;
    // Per keypoint, upstream's prune0 / prune1: 1 + the layers after which it was kept, while points were
    // being dropped; equal to the layers run for every keypoint when width is not adaptive.
    std::vector<int> layers0, layers1;
    int stop_layer = 0;                    // layers run
  };

  explicit LightGlue(const Weights& weights);

  // xy: n keypoints (x, y) in pixels of an image of width x height; desc: n x input_dim descriptors.
  Image prepare(Ops& ops, const float* xy, const float* desc, int n, float width, float height,
                const Trace& trace = nullptr, const std::string& tag = "") const;

  void match(Ops& ops, const Image& im0, const Image& im1, const Options& options, Result* out,
             const Trace& trace = nullptr) const;

  int max_layers() const { return n_layers_; }
  float default_depth_confidence() const { return depth_confidence_; }
  float default_width_confidence() const { return width_confidence_; }
  int input_dim() const { return input_dim_; }

 private:
  TensorPtr lin(Ops& ops, const std::string& name, const TensorPtr& x, Act act = Act::None,
                const TensorPtr& residual = nullptr) const;
  TensorPtr ffn(Ops& ops, const std::string& block, const TensorPtr& x, const TensorPtr& message) const;
  TensorPtr self_block(Ops& ops, const std::string& p, const TensorPtr& x, const TensorPtr& cos,
                       const TensorPtr& sin) const;
  // Reads back sigmoid(linear(name, x)): one value per token.
  std::vector<float> probabilities(Ops& ops, const std::string& name, const TensorPtr& x) const;

  const Weights& w_;
  int n_layers_, heads_, dim_, input_dim_;
  float depth_confidence_ = 0.95f, width_confidence_ = 0.99f;  // upstream's defaults for this model
  std::vector<float> thresholds_;  // confidence threshold after each layer, as upstream's float32 buffer
};

}  // namespace jmt
