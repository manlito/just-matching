// FlexGlue graph, written against core/ops.h. Mirrors tools/folded.py, which is the specification.
//
// Coordinates in this header are FlexGlue's: pixels of the original image with (0,0) at the top-left corner
// of the top-left pixel. The C API converts from and to the just-keypoints convention.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/ops.h"
#include "core/weights.h"

namespace jmt {

struct MatchOptions {
  int n_layers = 9;
  float filter_threshold = 0.1f;   // minimum match confidence
  float max_distance_px = 3.0f;    // maximum mean transfer error, in pixels of the network input; <= 0: off
  float min_covisibility = 0.0f;   // pairs scoring below it get no matches
  float early_stop_decay = -1.0f;  // in [0, 1): stop early on non-covisible pairs; < 0: off
};

struct MatchResult {
  std::vector<int> matches0;                 // per keypoint of image 0: index into image 1, or -1
  std::vector<float> scores0;                // per keypoint of image 0; 0 when unmatched
  std::vector<float> p2d0_i1, p2d1_i0;       // predicted point of every keypoint in the other image, (n, 2)
  std::vector<float> confidence0, confidence1;
  float covisibility = 0;
  int stop_layer = 0;
};

// Called with the name of a stage (as in tools/dump_golden.py) and its tensor.
using Trace = std::function<void(const std::string&, const TensorPtr&)>;

class FlexGlue {
 public:
  explicit FlexGlue(const Weights& weights);

  // rgb: height x width x 3 floats in [0, 1] -> features (size / 8, size / 8, 256).
  TensorPtr describe(Ops& ops, const float* rgb, int height, int width, int size, const Trace& trace = nullptr) const;

  // xy0: n0 keypoints of image 0 (x, y), (w0, h0) its size in pixels; likewise for image 1.
  void match(Ops& ops, const TensorPtr& f0, const float* xy0, int n0, float w0, float h0, const TensorPtr& f1,
             const float* xy1, int n1, float w1, float h1, const MatchOptions& options, MatchResult* out,
             const Trace& trace = nullptr) const;

  int max_layers() const { return n_layers_; }

 private:
  struct Tokens {
    TensorPtr x0, x1;
    int layer;
    float covisibility;
  };
  TensorPtr lin(Ops& ops, const std::string& name, const TensorPtr& x) const;
  TensorPtr norm(Ops& ops, const std::string& name, const TensorPtr& x, float eps = 1e-5f) const;
  TensorPtr mlp3(Ops& ops, const std::string& name, const TensorPtr& x) const;
  TensorPtr ffn(Ops& ops, const std::string& block, const TensorPtr& x, const TensorPtr& message) const;
  TensorPtr self_block(Ops& ops, const std::string& p, const TensorPtr& x, const std::vector<float>& cos,
                       const std::vector<float>& sin) const;
  void rotary_encoding(int h, int w, std::vector<float>* cos, std::vector<float>* sin) const;
  float covisibility(Ops& ops, int layer, const TensorPtr& x0, const TensorPtr& x1) const;
  Tokens decode_pair(Ops& ops, const TensorPtr& f0, const TensorPtr& f1, const MatchOptions& options,
                     const Trace& trace) const;
  void head(Ops& ops, int layer, const std::vector<float>& pts, const TensorPtr& cf_i, const TensorPtr& cf_j,
            std::vector<float>* refined, std::vector<float>* confidence, const Trace& trace,
            const std::string& tag) const;

  const Weights& w_;
  int n_layers_, heads_;
};

}  // namespace jmt
