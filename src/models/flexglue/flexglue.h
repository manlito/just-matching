// FlexGlue graph, written against core/ops.h. Mirrors tools/folded.py, which is the specification.
//
// Coordinates in this header are FlexGlue's: pixels of the original image with (0,0) at the top-left corner
// of the top-left pixel. The C API converts from and to the just-keypoints convention.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
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

  // The same from 8-bit pixels (`channels` bytes each: 1 grey, 3 or 4 with R, G, B first; rows of `stride`
  // bytes), so a backend can do the conversion and the resize on the device.
  TensorPtr describe_u8(Ops& ops, const unsigned char* data, int height, int width, int stride, int channels,
                        int size) const;

  // A pair whose work is queued on the backend and not read back yet.
  struct Pair;
  // match in two steps, so the work of several pairs can be queued before any result is waited for:
  // match_queue issues every operation of a pair, match_read reads the results and does the host part
  // (offsets, mutual nearest neighbours). With early stop on, match_queue itself reads a score per layer.
  std::shared_ptr<Pair> match_queue(Ops& ops, const TensorPtr& f0, const float* xy0, int n0, float w0, float h0,
                                    const TensorPtr& f1, const float* xy1, int n1, float w1, float h1,
                                    const MatchOptions& options, const Trace& trace = nullptr) const;
  void match_read(Pair& pair, MatchResult* out) const;

  // xy0: n0 keypoints of image 0 (x, y), (w0, h0) its size in pixels; likewise for image 1.
  void match(Ops& ops, const TensorPtr& f0, const float* xy0, int n0, float w0, float h0, const TensorPtr& f1,
             const float* xy1, int n1, float w1, float h1, const MatchOptions& options, MatchResult* out,
             const Trace& trace = nullptr) const;

  int max_layers() const { return n_layers_; }

  // ---- Covisibility levels 1 and 2 ("mean" and "grid0" in tools/covis.py).

  // Level 1: the L2-normalized mean of the features over all cells (256 values). Reads the features back.
  std::vector<float> global_vector(const TensorPtr& features) const;
  // The layer-0 head's search descriptors of every cell, L2-normalized: (h, w, 64). desc0 is input_proj
  // followed by heads.0.to_search, so this is the head's own search map without the 512-channel tokens.
  TensorPtr search0(Ops& ops, const TensorPtr& features) const;
  struct Image0 {
    TensorPtr features, search;  // search: search0(features)
  };
  // Level 2 for every pair: a grid x grid lattice of cell centres of each image through the layer-0 head
  // to the other image and back; the share that returns within one cell, mean of both directions.
  std::vector<float> grid_scores(Ops& ops, const std::vector<std::pair<Image0, Image0>>& pairs, int grid = 16) const;

 private:
  struct Tokens {
    TensorPtr x0, x1;
    int layer;
    float covisibility;       // < 0: not read yet, see covis_logits
    TensorPtr covis_logits[2];
  };
  struct HeadWork {
    int n = 0, h = 0, w = 0, layer = 0;
    TensorPtr coarse_t, mass_t, logits, matchability;
    std::vector<float> coarse, mass;
  };
  TensorPtr backbone(Ops& ops, TensorPtr x, const Trace& trace) const;
  TensorPtr lin(Ops& ops, const std::string& name, const TensorPtr& x, Act act = Act::None,
                const TensorPtr& residual = nullptr) const;
  TensorPtr norm(Ops& ops, const std::string& name, const TensorPtr& x, float eps = 1e-5f, Act act = Act::None) const;
  TensorPtr mlp3(Ops& ops, const std::string& name, const TensorPtr& x) const;
  TensorPtr ffn(Ops& ops, const std::string& block, const TensorPtr& x, const TensorPtr& message) const;
  TensorPtr self_block(Ops& ops, const std::string& p, const TensorPtr& x, const std::vector<float>& cos,
                       const std::vector<float>& sin) const;
  struct Rotary {
    std::vector<float> cos, sin;
  };
  const Rotary& rotary_encoding(int h, int w) const;
  void covisibility_queue(Ops& ops, int layer, const TensorPtr& x0, const TensorPtr& x1, TensorPtr logits[2]) const;
  float covisibility_read(const TensorPtr logits[2]) const;
  Tokens decode_pair(Ops& ops, const TensorPtr& f0, const TensorPtr& f1, const MatchOptions& options,
                     const Trace& trace) const;
  HeadWork head_queue(Ops& ops, int layer, const std::vector<float>& pts, const TensorPtr& cf_i, const TensorPtr& cf_j,
                      const Trace& trace, const std::string& tag) const;
  // The layer-0 head straight from the features: points of image i against the search map of image j.
  HeadWork head0_queue(Ops& ops, const std::vector<float>& pts, const TensorPtr& f_i, const TensorPtr& f_j,
                       const TensorPtr& search_j) const;
  // What both forms share once the query descriptors are known: read-out and offsets. The match is sampled
  // from src_j, then passed through the linear layer match_proj when it is not empty.
  HeadWork head_tail(Ops& ops, int layer, int n, const TensorPtr& query, const TensorPtr& search,
                     const TensorPtr& f_query, const TensorPtr& src_j, const std::string& match_proj,
                     const Trace& trace, const std::string& tag) const;
  void head_read(HeadWork& work, std::vector<float>* refined, std::vector<float>* confidence) const;

  const Weights& w_;
  int n_layers_, heads_;
  mutable std::mutex rotary_mutex_;
  mutable std::map<std::pair<int, int>, std::unique_ptr<Rotary>> rotary_;
};

}  // namespace jmt
