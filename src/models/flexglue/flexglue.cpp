#include "models/flexglue/flexglue.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace jmt {
namespace {

const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};
const int kDepths[4] = {3, 3, 27, 3};

std::vector<float> read(const TensorPtr& t) {
  std::vector<float> v(t->size());
  t->read(v.data());
  return v;
}

void emit(const Trace& trace, const std::string& name, const TensorPtr& t) {
  if (trace) trace(name, t);
}

std::string idx(const std::string& prefix, int i) { return prefix + std::to_string(i); }

}  // namespace

FlexGlue::FlexGlue(const Weights& weights) : w_(weights) {
  if (w_.meta("model") != "flexglue") throw std::runtime_error("not FlexGlue weights");
  n_layers_ = w_.meta_int("n_layers");
  heads_ = w_.meta_int("heads");
}

TensorPtr FlexGlue::lin(Ops& ops, const std::string& name, const TensorPtr& x) const {
  return ops.linear(x, w_.get(name + ".weight"), w_.find(name + ".bias"));
}

TensorPtr FlexGlue::norm(Ops& ops, const std::string& name, const TensorPtr& x, float eps) const {
  return ops.layer_norm(x, w_.get(name + ".weight"), w_.get(name + ".bias"), eps);
}

// Linear, LayerNorm, ReLU, Linear (entries 0, 1 and 3 of the upstream Sequential).
TensorPtr FlexGlue::mlp3(Ops& ops, const std::string& name, const TensorPtr& x) const {
  return lin(ops, name + ".3", ops.relu(norm(ops, name + ".1", lin(ops, name + ".0", x))));
}

// ---------------------------------------------------------------- describe

TensorPtr FlexGlue::describe(Ops& ops, const float* rgb, int height, int width, int size, const Trace& trace) const {
  if (size <= 0 || size % 32 != 0) throw std::runtime_error("input size must be a positive multiple of 32");
  TensorPtr x = ops.preprocess(rgb, height, width, size, size, kMean, kStd);
  emit(trace, "input", x);
  TensorPtr feats[4];
  for (int s = 0; s < 4; ++s) {
    std::string p = idx("backbone.downsample_layers.", s);
    if (s == 0) {
      x = ops.conv(x, w_.get(p + ".0.weight"), w_.get(p + ".0.bias"), 4, 4, 0);
      x = norm(ops, p + ".1", x, 1e-6f);
    } else {
      x = norm(ops, p + ".0", x, 1e-6f);
      x = ops.conv(x, w_.get(p + ".1.weight"), w_.get(p + ".1.bias"), 2, 2, 0);
    }
    emit(trace, idx("down", s), x);
    for (int b = 0; b < kDepths[s]; ++b) {
      p = idx("backbone.stages.", s) + "." + std::to_string(b);
      TensorPtr y = ops.dwconv7(x, w_.get(p + ".dwconv.weight"), w_.get(p + ".dwconv.bias"));
      y = norm(ops, p + ".norm", y, 1e-6f);
      y = lin(ops, p + ".pwconv2", ops.gelu(lin(ops, p + ".pwconv1", y)));  // layer scale is folded in
      x = ops.add(x, y);
    }
    feats[s] = x;
    emit(trace, idx("stage", s), x);
  }
  // Neck: merge the four stages at stride 8.
  x = norm(ops, "neck.stem.0", feats[0]);
  x = ops.conv(x, w_.get("neck.stem.1.weight"), w_.get("neck.stem.1.bias"), 3, 1, 1);
  x = ops.gelu(norm(ops, "neck.stem.2", x));
  x = lin(ops, "neck.fine_proj", ops.pixel_unshuffle2(x));
  emit(trace, "neck_fine", x);
  for (int i = 0; i < 3; ++i) {
    TensorPtr c = lin(ops, idx("neck.projs.", i), norm(ops, idx("neck.norms.", i), feats[i + 1]));
    if (c->h != x->h || c->w != x->w) c = ops.resize(c, x->h, x->w);
    x = ops.add(x, c);
  }
  emit(trace, "neck_sum", x);
  TensorPtr y = lin(ops, "neck.mlp.3", ops.gelu(lin(ops, "neck.mlp.1", norm(ops, "neck.mlp.0", x))));
  x = norm(ops, "neck.out_norm", lin(ops, "neck.down", ops.add(x, y)));
  emit(trace, "features", x);
  return x;
}

// ---------------------------------------------------------------- transformer

// cos and sin of the rotary angles of every token: (h * w, pairs). Positions are cell centres in [-1, 1],
// scaled by the longer side of the grid.
void FlexGlue::rotary_encoding(int h, int w, std::vector<float>* cos, std::vector<float>* sin) const {
  const Param& wr = w_.get("posenc.Wr.weight");  // (pairs, 2)
  const int pairs = wr.shape[0];
  const float scale = static_cast<float>(std::max(h, w));
  cos->resize(static_cast<size_t>(h) * w * pairs);
  sin->resize(cos->size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float cx = (static_cast<float>(x) + 0.5f) / scale * 2 - 1, cy = (static_cast<float>(y) + 0.5f) / scale * 2 - 1;
      for (int j = 0; j < pairs; ++j) {
        float angle = cx * wr.data[2 * j] + cy * wr.data[2 * j + 1];
        (*cos)[(static_cast<size_t>(y) * w + x) * pairs + j] = std::cos(angle);
        (*sin)[(static_cast<size_t>(y) * w + x) * pairs + j] = std::sin(angle);
      }
    }
}

// x + Linear(GELU(LayerNorm(Linear([x, LayerNorm(message)])))).
TensorPtr FlexGlue::ffn(Ops& ops, const std::string& block, const TensorPtr& x, const TensorPtr& message) const {
  TensorPtr y = ops.concat(x, norm(ops, block + ".message_norm", message));
  y = ops.gelu(norm(ops, block + ".ffn.1", lin(ops, block + ".ffn.0", y)));
  return ops.add(x, lin(ops, block + ".ffn.3", y));
}

TensorPtr FlexGlue::self_block(Ops& ops, const std::string& p, const TensorPtr& x, const std::vector<float>& cos,
                               const std::vector<float>& sin) const {
  TensorPtr qkv = lin(ops, p + ".Wqkv", x);  // [q | k | v], each (heads, dim / heads)
  const int d = x->c;
  TensorPtr q = ops.rotary(norm(ops, p + ".q_norm", ops.slice(qkv, 0, d)), cos, sin, heads_);
  TensorPtr k = ops.rotary(norm(ops, p + ".k_norm", ops.slice(qkv, d, d)), cos, sin, heads_);
  TensorPtr message = lin(ops, p + ".out_proj", ops.attention(q, k, ops.slice(qkv, 2 * d, d), heads_));
  return ffn(ops, p, x, message);
}

// Mean over the tokens of both images of sigmoid(linear(token)). Layer 0 has no such head.
float FlexGlue::covisibility(Ops& ops, int layer, const TensorPtr& x0, const TensorPtr& x1) const {
  if (layer == 0) return 1.0f;
  double sum = 0;
  size_t count = 0;
  for (const TensorPtr& x : {x0, x1}) {
    std::vector<float> v = read(lin(ops, idx("heads.", layer) + ".covisibility", x));
    for (float logit : v) sum += 1.0 / (1.0 + std::exp(-static_cast<double>(logit)));
    count += v.size();
  }
  return static_cast<float>(sum / static_cast<double>(count));
}

FlexGlue::Tokens FlexGlue::decode_pair(Ops& ops, const TensorPtr& f0, const TensorPtr& f1, const MatchOptions& o,
                                       const Trace& trace) const {
  if (o.n_layers < 0 || o.n_layers > n_layers_) throw std::runtime_error("n_layers out of range");
  if (o.early_stop_decay >= 1) throw std::runtime_error("early_stop_decay must be below 1");
  TensorPtr x0 = lin(ops, "input_proj", f0), x1 = lin(ops, "input_proj", f1);
  std::vector<float> cos0, sin0, cos1, sin1;
  rotary_encoding(f0->h, f0->w, &cos0, &sin0);
  rotary_encoding(f1->h, f1->w, &cos1, &sin1);
  emit(trace, "tokens0_l0", x0), emit(trace, "tokens1_l0", x1);
  Tokens t{nullptr, nullptr, o.n_layers, -1.0f};
  for (int k = 1; k <= o.n_layers; ++k) {
    std::string p = idx("transformers.", k - 1);
    x0 = self_block(ops, p + ".self_attn", x0, cos0, sin0);
    x1 = self_block(ops, p + ".self_attn", x1, cos1, sin1);
    emit(trace, idx("tokens0_l", k) + "_self", x0);
    p += ".cross_attn";
    TensorPtr qk0 = norm(ops, p + ".qk_norm", lin(ops, p + ".to_qk", x0));
    TensorPtr qk1 = norm(ops, p + ".qk_norm", lin(ops, p + ".to_qk", x1));
    TensorPtr v0 = lin(ops, p + ".to_v", x0), v1 = lin(ops, p + ".to_v", x1);
    TensorPtr m0 = lin(ops, p + ".to_out", ops.attention(qk0, qk1, v1, heads_));
    TensorPtr m1 = lin(ops, p + ".to_out", ops.attention(qk1, qk0, v0, heads_));
    x0 = ffn(ops, p, x0, m0);
    x1 = ffn(ops, p, x1, m1);
    emit(trace, idx("tokens0_l", k), x0), emit(trace, idx("tokens1_l", k), x1);
    if (o.early_stop_decay < 0 || k == 1 || k == o.n_layers) continue;
    // Stop on a confident non-pair: the bar is lower the earlier the layer.
    float score = covisibility(ops, k, x0, x1);
    if (score < o.min_covisibility * std::pow(1.0f - o.early_stop_decay, static_cast<float>(n_layers_ - k))) {
      t.layer = k, t.covisibility = score;
      break;
    }
  }
  if (t.covisibility < 0) t.covisibility = covisibility(ops, t.layer, x0, x1);
  t.x0 = x0, t.x1 = x1;
  return t;
}

// ---------------------------------------------------------------- head

// pts: n points of image i in [-1, 1] -> predicted points in image j in [-1, 1] and their confidence.
void FlexGlue::head(Ops& ops, int layer, const std::vector<float>& pts, const TensorPtr& cf_i, const TensorPtr& cf_j,
                    std::vector<float>* refined, std::vector<float>* confidence, const Trace& trace,
                    const std::string& tag) const {
  const std::string p = idx("heads.", layer);
  const int n = static_cast<int>(pts.size() / 2), h = cf_j->h, w = cf_j->w;
  refined->assign(pts.size(), 0.0f);
  confidence->assign(n, 0.0f);
  if (n == 0) return;
  TensorPtr search = ops.l2_normalize(lin(ops, p + ".to_search", cf_j));
  TensorPtr f_query = ops.sample(cf_i, pts.data(), n);
  TensorPtr query = ops.l2_normalize(lin(ops, p + ".to_search", f_query));
  std::vector<float> coarse, mass;
  ops.readout(query, search, w_.get(p + ".temperature").data[0], &coarse, &mass);
  if (trace) {
    trace("coarse" + tag, ops.upload(coarse.data(), 1, n, 2));
    trace("mass" + tag, ops.upload(mass.data(), 1, n, 1));
  }
  // Sub-cell offset: a distribution over 17 bin edges spanning +-1 cell per axis; the 18th output is unused.
  TensorPtr f_match = ops.sample(cf_j, coarse.data(), n);
  TensorPtr x = ops.concat(mlp3(ops, p + ".offset_head.query_encoder", f_query),
                           mlp3(ops, p + ".offset_head.match_encoder", f_match));
  std::vector<float> logits = read(mlp3(ops, p + ".offset_head.head", x));  // (n, 2 * 18)
  std::vector<float> matchability;
  if (layer > 0) matchability = read(lin(ops, p + ".matchability", f_query));
  for (int i = 0; i < n; ++i) {
    float delta[2];
    for (int axis = 0; axis < 2; ++axis) {
      const float* l = logits.data() + static_cast<size_t>(i) * 36 + axis * 18;
      float top = l[0];
      for (int b = 1; b < 17; ++b) top = std::max(top, l[b]);
      double sum = 0, expect = 0;
      for (int b = 0; b < 17; ++b) {
        double e = std::exp(static_cast<double>(l[b] - top));
        sum += e, expect += e * b;
      }
      delta[axis] = static_cast<float>((expect / sum / 16 - 0.5) * 2);
    }
    (*refined)[2 * i] = coarse[2 * i] + delta[0] * 2 / static_cast<float>(w);
    (*refined)[2 * i + 1] = coarse[2 * i + 1] + delta[1] * 2 / static_cast<float>(h);
    float m = layer > 0 ? 1.0f / (1.0f + std::exp(-matchability[i])) : 1.0f;
    (*confidence)[i] = m * mass[i];
  }
}

// ---------------------------------------------------------------- sparse matching

void FlexGlue::match(Ops& ops, const TensorPtr& f0, const float* xy0, int n0, float w0, float h0, const TensorPtr& f1,
                     const float* xy1, int n1, float w1, float h1, const MatchOptions& o, MatchResult* out,
                     const Trace& trace) const {
  Tokens t = decode_pair(ops, f0, f1, o, trace);
  out->covisibility = t.covisibility;
  out->stop_layer = t.layer;

  const float* xy[2] = {xy0, xy1};
  const int n[2] = {n0, n1};
  const float size[2][2] = {{w0, h0}, {w1, h1}};
  const float resized[2][2] = {{8.0f * f0->w, 8.0f * f0->h}, {8.0f * f1->w, 8.0f * f1->h}};
  const TensorPtr cf[2] = {t.x0, t.x1};
  std::vector<float>* pred[2] = {&out->p2d0_i1, &out->p2d1_i0};
  std::vector<float>* conf[2] = {&out->confidence0, &out->confidence1};
  // k: keypoints, q: predicted points (q[0] lives in image 1), both in pixels of the network input.
  std::vector<float> k[2], q[2];
  std::vector<char> valid[2];
  for (int i = 0; i < 2; ++i) {
    const int j = 1 - i;
    std::vector<float> pts(static_cast<size_t>(n[i]) * 2), refined;
    for (int a = 0; a < n[i]; ++a)
      for (int c = 0; c < 2; ++c) pts[2 * a + c] = xy[i][2 * a + c] / size[i][c] * 2 - 1;
    head(ops, t.layer, pts, cf[i], cf[j], &refined, conf[i], trace, std::to_string(i));
    pred[i]->resize(refined.size());
    k[i].resize(refined.size()), q[i].resize(refined.size());
    valid[i].assign(n[i], 1);
    for (int a = 0; a < n[i]; ++a)
      for (int c = 0; c < 2; ++c) {
        float p = (refined[2 * a + c] + 1) / 2 * size[j][c];
        (*pred[i])[2 * a + c] = p;
        k[i][2 * a + c] = xy[i][2 * a + c] * resized[i][c] / size[i][c];
        q[i][2 * a + c] = p * resized[j][c] / size[j][c];
        // Both the keypoint and its prediction must lie strictly inside their images.
        if (!(k[i][2 * a + c] > 0 && k[i][2 * a + c] < resized[i][c])) valid[i][a] = 0;
        if (!(q[i][2 * a + c] > 0 && q[i][2 * a + c] < resized[j][c])) valid[i][a] = 0;
      }
    if (t.covisibility < o.min_covisibility) valid[i].assign(n[i], 0);
  }

  // Mutual nearest neighbours on the sum of the two squared transfer errors:
  // |prediction of a in image 1 - b|^2 + |a - prediction of b in image 0|^2.
  auto dist2 = [&](int a, int b) {
    float dx = q[0][2 * a] - k[1][2 * b], dy = q[0][2 * a + 1] - k[1][2 * b + 1];
    float ex = k[0][2 * a] - q[1][2 * b], ey = k[0][2 * a + 1] - q[1][2 * b + 1];
    return dx * dx + dy * dy + ex * ex + ey * ey;
  };
  std::vector<int> best0(n0, -1), best1(n1, -1);
  std::vector<float> d1(n1, 0.0f);
  for (int a = 0; a < n0; ++a) {
    float best = 0;
    for (int b = 0; b < n1; ++b) {
      float d = dist2(a, b);
      if (best0[a] < 0 || d < best) best = d, best0[a] = b;
      if (best1[b] < 0 || d < d1[b]) d1[b] = d, best1[b] = a;
    }
  }
  out->matches0.assign(n0, -1);
  out->scores0.assign(n0, 0.0f);
  for (int a = 0; a < n0; ++a) {
    const int b = best0[a];
    if (b < 0 || best1[b] != a || !valid[0][a] || !valid[1][b]) continue;
    float score = std::sqrt((*conf[0])[a] * (*conf[1])[b]);
    if (!(score >= o.filter_threshold)) continue;
    float there = std::hypot(q[0][2 * a] - k[1][2 * b], q[0][2 * a + 1] - k[1][2 * b + 1]);
    float back = std::hypot(k[0][2 * a] - q[1][2 * b], k[0][2 * a + 1] - q[1][2 * b + 1]);
    if (o.max_distance_px > 0 && !(0.5f * (there + back) < o.max_distance_px)) continue;
    out->matches0[a] = b;
    out->scores0[a] = score;
  }
}

}  // namespace jmt
