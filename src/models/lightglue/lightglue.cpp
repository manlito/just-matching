#include "models/lightglue/lightglue.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace jmt {
namespace {

std::vector<float> read(const TensorPtr& t) {
  std::vector<float> v(t->size());
  if (!v.empty()) t->read(v.data());
  return v;
}

void emit(const Trace& trace, const std::string& name, const TensorPtr& t) {
  if (trace) trace(name, t);
}

std::string idx(const std::string& prefix, int i) { return prefix + std::to_string(i); }

float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }

}  // namespace

LightGlue::LightGlue(const Weights& weights) : w_(weights) {
  if (w_.meta("model") != "lightglue") throw std::runtime_error("not LightGlue weights");
  n_layers_ = w_.meta_int("n_layers");
  heads_ = w_.meta_int("heads");
  dim_ = w_.meta_int("dim");
  input_dim_ = w_.meta_int("input_dim");
  if (w_.metadata().count("depth_confidence")) depth_confidence_ = std::stof(w_.meta("depth_confidence"));
  if (w_.metadata().count("width_confidence")) width_confidence_ = std::stof(w_.meta("width_confidence"));
  for (int i = 0; i < n_layers_; ++i) {
    const double t = 0.8 + 0.1 * std::exp(-4.0 * i / n_layers_);
    thresholds_.push_back(static_cast<float>(std::min(std::max(t, 0.0), 1.0)));
  }
}

std::vector<float> LightGlue::probabilities(Ops& ops, const std::string& name, const TensorPtr& x) const {
  std::vector<float> v = read(lin(ops, name, x));
  for (float& p : v) p = sigmoid(p);
  return v;
}

TensorPtr LightGlue::lin(Ops& ops, const std::string& name, const TensorPtr& x, Act act, const TensorPtr& residual) const {
  return ops.linear_fused(x, w_.get(name + ".weight"), w_.find(name + ".bias"), act, residual);
}

// x + Linear(GELU(LayerNorm(Linear([x, message])))): entries 0, 1 and 3 of the upstream Sequential.
TensorPtr LightGlue::ffn(Ops& ops, const std::string& block, const TensorPtr& x, const TensorPtr& message) const {
  TensorPtr y = lin(ops, block + ".ffn.0", ops.concat(x, message));
  y = ops.layer_norm_act(y, w_.get(block + ".ffn.1.weight"), w_.get(block + ".ffn.1.bias"), 1e-5f, Act::Gelu);
  return lin(ops, block + ".ffn.3", y, Act::None, x);
}

TensorPtr LightGlue::self_block(Ops& ops, const std::string& p, const TensorPtr& x, const TensorPtr& cos,
                                const TensorPtr& sin) const {
  TensorPtr qkv = lin(ops, p + ".Wqkv", x);  // [q | k | v], each (heads, dim / heads)
  TensorPtr q = ops.rotary_keypoints(ops.slice(qkv, 0, dim_), cos, sin, heads_);
  TensorPtr k = ops.rotary_keypoints(ops.slice(qkv, dim_, dim_), cos, sin, heads_);
  TensorPtr message = lin(ops, p + ".out_proj", ops.attention(q, k, ops.slice(qkv, 2 * dim_, dim_), heads_));
  return ffn(ops, p, x, message);
}

LightGlue::Image LightGlue::prepare(Ops& ops, const float* xy, const float* desc, int n, float width, float height,
                                    const Trace& trace, const std::string& tag) const {
  if (n < 0 || (n && (!xy || !desc))) throw std::runtime_error("bad keypoints");
  if (!(width > 0 && height > 0)) throw std::runtime_error("image size must be positive");
  Image im;
  im.n = n;
  if (n == 0) return im;
  // Normalized as upstream: centred on the image, divided by half the longer side.
  const Param& wr = w_.get("posenc.Wr.weight");  // (pairs, 2), no bias
  const int pairs = wr.shape[0];
  const float shift_x = width / 2, shift_y = height / 2, scale = std::max(width, height) / 2;
  im.cos_h.resize(static_cast<size_t>(n) * pairs);
  im.sin_h.resize(im.cos_h.size());
  for (int i = 0; i < n; ++i) {
    const float x = (xy[2 * i] - shift_x) / scale, y = (xy[2 * i + 1] - shift_y) / scale;
    for (int j = 0; j < pairs; ++j) {
      const float angle = x * wr.data[2 * j] + y * wr.data[2 * j + 1];
      im.cos_h[static_cast<size_t>(i) * pairs + j] = std::cos(angle);
      im.sin_h[static_cast<size_t>(i) * pairs + j] = std::sin(angle);
    }
  }
  im.cos = ops.upload(im.cos_h.data(), 1, n, pairs);
  im.sin = ops.upload(im.sin_h.data(), 1, n, pairs);
  im.tokens = lin(ops, "input_proj", ops.upload(desc, 1, n, input_dim_));
  emit(trace, "input_proj" + tag, im.tokens);
  return im;
}

void LightGlue::match(Ops& ops, const Image& im0, const Image& im1, const Options& options, Result* out,
                      const Trace& trace) const {
  Options o = options;
  if (std::isnan(o.depth_confidence)) o.depth_confidence = depth_confidence_;
  if (std::isnan(o.width_confidence)) o.width_confidence = width_confidence_;
  if (o.n_layers < 1) throw std::runtime_error("n_layers out of range");
  o.n_layers = std::min(o.n_layers, n_layers_);  // more than the model has: all of them (LighterGlue has 6)
  const int m = im0.n, n = im1.n, last = o.n_layers - 1;
  out->matches0.assign(m, -1), out->matches1.assign(n, -1);
  out->scores0.assign(m, 0.0f), out->scores1.assign(n, 0.0f);
  out->matchability0.assign(m, 0.0f), out->matchability1.assign(n, 0.0f);
  out->layers0.assign(m, o.n_layers), out->layers1.assign(n, o.n_layers);
  out->stop_layer = 0;
  if (m == 0 || n == 0) return;  // nothing to attend to or match

  const bool early = o.depth_confidence > 0, prune = o.width_confidence > 0;
  // The keypoints still in play: their tokens, rotary tables and original indices.
  struct Side {
    TensorPtr x, cos, sin;
    std::vector<int> ind;
    std::vector<int>* layers;
    std::vector<float>* matchability;
  } side[2] = {{im0.tokens, im0.cos, im0.sin, std::vector<int>(m), &out->layers0, &out->matchability0},
               {im1.tokens, im1.cos, im1.sin, std::vector<int>(n), &out->layers1, &out->matchability1}};
  for (Side& sd : side) std::iota(sd.ind.begin(), sd.ind.end(), 0);
  if (prune) std::fill(out->layers0.begin(), out->layers0.end(), 1), std::fill(out->layers1.begin(), out->layers1.end(), 1);

  int layer = 0;  // upstream's i when the loop ends
  for (layer = 0; layer < o.n_layers; ++layer) {
    if (side[0].x->n() == 0 || side[1].x->n() == 0) break;
    std::string p = idx("transformers.", layer);
    for (Side& sd : side) sd.x = self_block(ops, p + ".self_attn", sd.x, sd.cos, sd.sin);
    p += ".cross_attn";
    // One similarity in both directions: the queries of one image are the keys of the other.
    TensorPtr qk0 = lin(ops, p + ".to_qk", side[0].x), qk1 = lin(ops, p + ".to_qk", side[1].x);
    TensorPtr v0 = lin(ops, p + ".to_v", side[0].x), v1 = lin(ops, p + ".to_v", side[1].x);
    TensorPtr m0 = lin(ops, p + ".to_out", ops.attention(qk0, qk1, v1, heads_));
    TensorPtr m1 = lin(ops, p + ".to_out", ops.attention(qk1, qk0, v0, heads_));
    side[0].x = ffn(ops, p, side[0].x, m0);
    side[1].x = ffn(ops, p, side[1].x, m1);
    emit(trace, idx("layer", layer) + ".desc0", side[0].x), emit(trace, idx("layer", layer) + ".desc1", side[1].x);
    if (layer == last) break;  // no early stop or dropping after the last layer

    const float th = thresholds_[layer];
    std::vector<float> conf[2];
    if (early) {
      // Stop when the confident share of all keypoints, dropped ones counted as confident, is high enough.
      int low = 0;
      for (int s = 0; s < 2; ++s) {
        conf[s] = probabilities(ops, idx("token_confidence.", layer) + ".token.0", side[s].x);
        for (float c : conf[s]) low += c < th;
      }
      const float ratio = 1.0f - static_cast<float>(low) / static_cast<float>(m + n);
      if (ratio > o.depth_confidence) break;
    }
    if (!prune) continue;
    // Drop the keypoints that are confident and unmatchable. Unconfident ones are always kept.
    const float min_matchability = static_cast<float>(1.0 - static_cast<double>(o.width_confidence));
    for (int s = 0; s < 2; ++s) {
      Side& sd = side[s];
      const std::vector<float> match_p = probabilities(ops, idx("log_assignment.", layer) + ".matchability", sd.x);
      std::vector<int> keep, ind;
      for (size_t k = 0; k < sd.ind.size(); ++k) {
        (*sd.matchability)[sd.ind[k]] = match_p[k];
        if (match_p[k] > min_matchability || (early && conf[s][k] <= th)) keep.push_back(static_cast<int>(k));
      }
      for (int k : keep) ind.push_back(sd.ind[k]), ++(*sd.layers)[sd.ind[k]];
      if (keep.size() != sd.ind.size()) {
        sd.x = ops.gather_rows(sd.x, keep);
        sd.cos = ops.gather_rows(sd.cos, keep), sd.sin = ops.gather_rows(sd.sin, keep);
      }
      sd.ind = std::move(ind);
    }
  }
  if (layer == o.n_layers) layer = last;  // the loop ran to the end without a break
  out->stop_layer = layer + 1;
  if (side[0].x->n() == 0 || side[1].x->n() == 0) return;

  const std::string head = idx("log_assignment.", layer);
  TensorPtr z0 = lin(ops, head + ".matchability", side[0].x), z1 = lin(ops, head + ".matchability", side[1].x);
  std::vector<int> best0, best1;
  std::vector<float> max0, max1;
  ops.assignment(lin(ops, head + ".final_proj", side[0].x), lin(ops, head + ".final_proj", side[1].x), z0, z1,
                 &best0, &max0, &best1, &max1);
  if (trace) {
    trace("max0", ops.upload(max0.data(), 1, static_cast<int>(max0.size()), 1));
    trace("max1", ops.upload(max1.data(), 1, static_cast<int>(max1.size()), 1));
  }
  // Mutual best matches whose score clears the threshold (upstream filter_matches), in original indices.
  const std::vector<int>&ind0 = side[0].ind, &ind1 = side[1].ind;
  for (size_t a = 0; a < best0.size(); ++a) {
    const int b = best0[a];
    if (best1[b] != static_cast<int>(a)) continue;
    const float score = std::exp(max0[a]);
    if (!(score > o.filter_threshold)) continue;
    out->matches0[ind0[a]] = ind1[b], out->scores0[ind0[a]] = score;
    out->matches1[ind1[b]] = ind0[a], out->scores1[ind1[b]] = score;
  }
  const std::vector<float> h0 = read(z0), h1 = read(z1);
  for (size_t a = 0; a < h0.size(); ++a) out->matchability0[ind0[a]] = sigmoid(h0[a]);
  for (size_t b = 0; b < h1.size(); ++b) out->matchability1[ind1[b]] = sigmoid(h1[b]);
}

}  // namespace jmt
