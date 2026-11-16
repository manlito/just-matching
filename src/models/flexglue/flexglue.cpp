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

// act(linear(x)) + residual. The activation and the residual are passed to the backend with the product,
// so a backend can apply them while the result is still in registers.
TensorPtr FlexGlue::lin(Ops& ops, const std::string& name, const TensorPtr& x, Act act, const TensorPtr& residual) const {
  return ops.linear_fused(x, w_.get(name + ".weight"), w_.find(name + ".bias"), act, residual);
}

TensorPtr FlexGlue::norm(Ops& ops, const std::string& name, const TensorPtr& x, float eps, Act act) const {
  return ops.layer_norm_act(x, w_.get(name + ".weight"), w_.get(name + ".bias"), eps, act);
}

// Linear, LayerNorm, ReLU, Linear (entries 0, 1 and 3 of the upstream Sequential).
TensorPtr FlexGlue::mlp3(Ops& ops, const std::string& name, const TensorPtr& x) const {
  return lin(ops, name + ".3", norm(ops, name + ".1", lin(ops, name + ".0", x), 1e-5f, Act::Relu));
}

// ---------------------------------------------------------------- describe

TensorPtr FlexGlue::describe(Ops& ops, const float* rgb, int height, int width, int size, const Trace& trace) const {
  if (size <= 0 || size % 32 != 0) throw std::runtime_error("input size must be a positive multiple of 32");
  return backbone(ops, ops.preprocess(rgb, height, width, size, size, kMean, kStd), trace);
}

TensorPtr FlexGlue::describe_u8(Ops& ops, const unsigned char* data, int height, int width, int stride, int channels,
                                int size) const {
  if (size <= 0 || size % 32 != 0) throw std::runtime_error("input size must be a positive multiple of 32");
  return backbone(ops, ops.preprocess_u8(data, height, width, stride, channels, size, size, kMean, kStd), nullptr);
}

TensorPtr FlexGlue::backbone(Ops& ops, TensorPtr x, const Trace& trace) const {
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
      x = lin(ops, p + ".pwconv2", lin(ops, p + ".pwconv1", y, Act::Gelu), Act::None, x);  // layer scale is folded in
    }
    feats[s] = x;
    emit(trace, idx("stage", s), x);
  }
  // Neck: merge the four stages at stride 8.
  x = norm(ops, "neck.stem.0", feats[0]);
  x = ops.conv(x, w_.get("neck.stem.1.weight"), w_.get("neck.stem.1.bias"), 3, 1, 1);
  x = norm(ops, "neck.stem.2", x, 1e-5f, Act::Gelu);
  x = lin(ops, "neck.fine_proj", ops.pixel_unshuffle2(x));
  emit(trace, "neck_fine", x);
  for (int i = 0; i < 3; ++i) {
    TensorPtr c = lin(ops, idx("neck.projs.", i), norm(ops, idx("neck.norms.", i), feats[i + 1]));
    if (c->h != x->h || c->w != x->w) c = ops.resize(c, x->h, x->w);
    x = ops.add(x, c);
  }
  emit(trace, "neck_sum", x);
  x = lin(ops, "neck.mlp.3", lin(ops, "neck.mlp.1", norm(ops, "neck.mlp.0", x), Act::Gelu), Act::None, x);
  x = norm(ops, "neck.out_norm", lin(ops, "neck.down", x));
  emit(trace, "features", x);
  return x;
}

// ---------------------------------------------------------------- transformer

// cos and sin of the rotary angles of every token: (h * w, pairs). Positions are cell centres in [-1, 1],
// scaled by the longer side of the grid. Computed once per grid size and kept: backends may key a device
// copy on the address of the vectors.
const FlexGlue::Rotary& FlexGlue::rotary_encoding(int h, int w) const {
  std::lock_guard<std::mutex> lock(rotary_mutex_);
  std::unique_ptr<Rotary>& slot = rotary_[{h, w}];
  if (slot) return *slot;
  slot = std::make_unique<Rotary>();
  const Param& wr = w_.get("posenc.Wr.weight");  // (pairs, 2)
  const int pairs = wr.shape[0];
  const float scale = static_cast<float>(std::max(h, w));
  slot->cos.resize(static_cast<size_t>(h) * w * pairs);
  slot->sin.resize(slot->cos.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float cx = (static_cast<float>(x) + 0.5f) / scale * 2 - 1, cy = (static_cast<float>(y) + 0.5f) / scale * 2 - 1;
      for (int j = 0; j < pairs; ++j) {
        float angle = cx * wr.data[2 * j] + cy * wr.data[2 * j + 1];
        slot->cos[(static_cast<size_t>(y) * w + x) * pairs + j] = std::cos(angle);
        slot->sin[(static_cast<size_t>(y) * w + x) * pairs + j] = std::sin(angle);
      }
    }
  return *slot;
}

// x + Linear(GELU(LayerNorm(Linear([x, LayerNorm(message)])))).
TensorPtr FlexGlue::ffn(Ops& ops, const std::string& block, const TensorPtr& x, const TensorPtr& message) const {
  TensorPtr y = ops.concat(x, norm(ops, block + ".message_norm", message));
  y = norm(ops, block + ".ffn.1", lin(ops, block + ".ffn.0", y), 1e-5f, Act::Gelu);
  return lin(ops, block + ".ffn.3", y, Act::None, x);
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

// Covisibility: the mean over the tokens of both images of sigmoid(linear(token)). Layer 0 has no such
// head. The products are queued here and read by covisibility_read.
void FlexGlue::covisibility_queue(Ops& ops, int layer, const TensorPtr& x0, const TensorPtr& x1, TensorPtr logits[2]) const {
  logits[0] = logits[1] = nullptr;
  if (layer == 0) return;
  logits[0] = lin(ops, idx("heads.", layer) + ".covisibility", x0);
  logits[1] = lin(ops, idx("heads.", layer) + ".covisibility", x1);
}

float FlexGlue::covisibility_read(const TensorPtr logits[2]) const {
  if (!logits[0]) return 1.0f;
  double sum = 0;
  size_t count = 0;
  for (int i = 0; i < 2; ++i) {
    std::vector<float> v = read(logits[i]);
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
  const Rotary &r0 = rotary_encoding(f0->h, f0->w), &r1 = rotary_encoding(f1->h, f1->w);
  const std::vector<float>&cos0 = r0.cos, &sin0 = r0.sin, &cos1 = r1.cos, &sin1 = r1.sin;
  emit(trace, "tokens0_l0", x0), emit(trace, "tokens1_l0", x1);
  Tokens t{nullptr, nullptr, o.n_layers, -1.0f, {nullptr, nullptr}};
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
    TensorPtr logits[2];
    covisibility_queue(ops, k, x0, x1, logits);
    float score = covisibility_read(logits);  // waits for the backend: early stop costs a round trip per layer
    if (score < o.min_covisibility * std::pow(1.0f - o.early_stop_decay, static_cast<float>(n_layers_ - k))) {
      t.layer = k, t.covisibility = score;
      break;
    }
  }
  if (t.covisibility < 0) covisibility_queue(ops, t.layer, x0, x1, t.covis_logits);
  t.x0 = x0, t.x1 = x1;
  return t;
}

// ---------------------------------------------------------------- head

// Queues the head for n points of image i (pts, in [-1, 1]) against image j. With a backend that has the
// read-out on the device nothing is read back here.
FlexGlue::HeadWork FlexGlue::head_queue(Ops& ops, int layer, const std::vector<float>& pts, const TensorPtr& cf_i,
                                        const TensorPtr& cf_j, const Trace& trace, const std::string& tag) const {
  const std::string p = idx("heads.", layer);
  const int n = static_cast<int>(pts.size() / 2);
  if (n == 0) {
    HeadWork work;
    work.h = cf_j->h, work.w = cf_j->w, work.layer = layer;
    return work;
  }
  TensorPtr search = ops.l2_normalize(lin(ops, p + ".to_search", cf_j));
  TensorPtr f_query = ops.sample(cf_i, pts.data(), n);
  TensorPtr query = ops.l2_normalize(lin(ops, p + ".to_search", f_query));
  return head_tail(ops, layer, n, query, search, f_query, cf_j, "", trace, tag);
}

// Sampling commutes with input_proj (bilinear weights sum to 1), so the 512-channel tokens are needed
// only at the query points and at their best cells.
FlexGlue::HeadWork FlexGlue::head0_queue(Ops& ops, const std::vector<float>& pts, const TensorPtr& f_i,
                                         const TensorPtr& f_j, const TensorPtr& search_j) const {
  const int n = static_cast<int>(pts.size() / 2);
  if (n == 0) {
    HeadWork work;
    work.h = f_j->h, work.w = f_j->w;
    return work;
  }
  TensorPtr s = ops.sample(f_i, pts.data(), n);
  TensorPtr query = ops.l2_normalize(lin(ops, "desc0", s));
  return head_tail(ops, 0, n, query, search_j, lin(ops, "input_proj", s), f_j, "input_proj", nullptr, "");
}

FlexGlue::HeadWork FlexGlue::head_tail(Ops& ops, int layer, int n, const TensorPtr& query, const TensorPtr& search,
                                       const TensorPtr& f_query, const TensorPtr& src_j, const std::string& match_proj,
                                       const Trace& trace, const std::string& tag) const {
  const std::string p = idx("heads.", layer);
  HeadWork work;
  work.n = n, work.h = search->h, work.w = search->w, work.layer = layer;
  const float temperature = w_.get(p + ".temperature").data[0];
  TensorPtr f_match;
  if (ops.has_device_readout()) {
    ops.readout_device(query, search, temperature, &work.coarse_t, &work.mass_t);
    f_match = ops.sample_at(src_j, work.coarse_t);
  } else {
    ops.readout(query, search, temperature, &work.coarse, &work.mass);
    f_match = ops.sample(src_j, work.coarse.data(), n);
    if (trace) work.coarse_t = ops.upload(work.coarse.data(), 1, n, 2), work.mass_t = ops.upload(work.mass.data(), 1, n, 1);
  }
  if (!match_proj.empty()) f_match = lin(ops, match_proj, f_match);
  if (trace) trace("coarse" + tag, work.coarse_t), trace("mass" + tag, work.mass_t);
  // Sub-cell offset: a distribution over 17 bin edges spanning +-1 cell per axis; the 18th output is unused.
  TensorPtr x = ops.concat(mlp3(ops, p + ".offset_head.query_encoder", f_query),
                           mlp3(ops, p + ".offset_head.match_encoder", f_match));
  work.logits = mlp3(ops, p + ".offset_head.head", x);  // (n, 2 * 18)
  if (layer > 0) work.matchability = lin(ops, p + ".matchability", f_query);
  return work;
}

// Reads the head's results back -> predicted points in image j in [-1, 1] and their confidence.
void FlexGlue::head_read(HeadWork& work, std::vector<float>* refined, std::vector<float>* confidence) const {
  const int n = work.n, h = work.h, w = work.w;
  refined->assign(static_cast<size_t>(n) * 2, 0.0f);
  confidence->assign(n, 0.0f);
  if (n == 0) return;
  if (work.coarse.empty()) work.coarse = read(work.coarse_t), work.mass = read(work.mass_t);
  const std::vector<float>&coarse = work.coarse, &mass = work.mass;
  std::vector<float> logits = read(work.logits), matchability;
  if (work.matchability) matchability = read(work.matchability);
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
    float m = work.matchability ? 1.0f / (1.0f + std::exp(-matchability[i])) : 1.0f;
    (*confidence)[i] = m * mass[i];
  }
}

// ---------------------------------------------------------------- sparse matching

struct FlexGlue::Pair {
  MatchOptions options;
  int n[2];
  float size[2][2], resized[2][2];
  std::vector<float> xy[2];
  int layer = 0;
  float covisibility = -1;
  TensorPtr covis_logits[2];
  HeadWork head[2];
};

std::shared_ptr<FlexGlue::Pair> FlexGlue::match_queue(Ops& ops, const TensorPtr& f0, const float* xy0, int n0, float w0,
                                                      float h0, const TensorPtr& f1, const float* xy1, int n1, float w1,
                                                      float h1, const MatchOptions& o, const Trace& trace) const {
  auto pair = std::make_shared<Pair>();
  Pair& p = *pair;
  Tokens t = decode_pair(ops, f0, f1, o, trace);
  p.options = o;
  p.layer = t.layer, p.covisibility = t.covisibility;
  p.covis_logits[0] = t.covis_logits[0], p.covis_logits[1] = t.covis_logits[1];
  p.n[0] = n0, p.n[1] = n1;
  p.size[0][0] = w0, p.size[0][1] = h0, p.size[1][0] = w1, p.size[1][1] = h1;
  p.resized[0][0] = 8.0f * f0->w, p.resized[0][1] = 8.0f * f0->h, p.resized[1][0] = 8.0f * f1->w, p.resized[1][1] = 8.0f * f1->h;
  p.xy[0].assign(xy0, xy0 + static_cast<size_t>(n0) * 2), p.xy[1].assign(xy1, xy1 + static_cast<size_t>(n1) * 2);
  const TensorPtr cf[2] = {t.x0, t.x1};
  for (int i = 0; i < 2; ++i) {
    std::vector<float> pts(static_cast<size_t>(p.n[i]) * 2);
    for (int a = 0; a < p.n[i]; ++a)
      for (int c = 0; c < 2; ++c) pts[2 * a + c] = p.xy[i][2 * a + c] / p.size[i][c] * 2 - 1;
    p.head[i] = head_queue(ops, t.layer, pts, cf[i], cf[1 - i], trace, std::to_string(i));
  }
  return pair;
}

void FlexGlue::match_read(Pair& p, MatchResult* out) const {
  const MatchOptions& o = p.options;
  const int n0 = p.n[0], n1 = p.n[1];
  const int* n = p.n;
  if (p.covisibility < 0) p.covisibility = covisibility_read(p.covis_logits);
  out->covisibility = p.covisibility;
  out->stop_layer = p.layer;
  std::vector<float>* pred[2] = {&out->p2d0_i1, &out->p2d1_i0};
  std::vector<float>* conf[2] = {&out->confidence0, &out->confidence1};
  // k: keypoints, q: predicted points (q[0] lives in image 1), both in pixels of the network input.
  std::vector<float> k[2], q[2];
  std::vector<char> valid[2];
  for (int i = 0; i < 2; ++i) {
    const int j = 1 - i;
    std::vector<float> refined;
    head_read(p.head[i], &refined, conf[i]);
    pred[i]->resize(refined.size());
    k[i].resize(refined.size()), q[i].resize(refined.size());
    valid[i].assign(n[i], 1);
    for (int a = 0; a < n[i]; ++a)
      for (int c = 0; c < 2; ++c) {
        float pt = (refined[2 * a + c] + 1) / 2 * p.size[j][c];
        (*pred[i])[2 * a + c] = pt;
        k[i][2 * a + c] = p.xy[i][2 * a + c] * p.resized[i][c] / p.size[i][c];
        q[i][2 * a + c] = pt * p.resized[j][c] / p.size[j][c];
        // Both the keypoint and its prediction must lie strictly inside their images.
        if (!(k[i][2 * a + c] > 0 && k[i][2 * a + c] < p.resized[i][c])) valid[i][a] = 0;
        if (!(q[i][2 * a + c] > 0 && q[i][2 * a + c] < p.resized[j][c])) valid[i][a] = 0;
      }
    if (p.covisibility < o.min_covisibility) valid[i].assign(n[i], 0);
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

void FlexGlue::match(Ops& ops, const TensorPtr& f0, const float* xy0, int n0, float w0, float h0, const TensorPtr& f1,
                     const float* xy1, int n1, float w1, float h1, const MatchOptions& o, MatchResult* out,
                     const Trace& trace) const {
  match_read(*match_queue(ops, f0, xy0, n0, w0, h0, f1, xy1, n1, w1, h1, o, trace), out);
}

// ---------------------------------------------------------------- covisibility, levels 1 and 2

std::vector<float> FlexGlue::global_vector(const TensorPtr& features) const {
  const std::vector<float> f = read(features);
  const int c = features->c, n = features->n();
  std::vector<double> sum(c, 0.0);
  for (int t = 0; t < n; ++t)
    for (int k = 0; k < c; ++k) sum[k] += f[static_cast<size_t>(t) * c + k];
  double norm = 0;
  for (double& s : sum) s /= n, norm += s * s;
  norm = std::max(std::sqrt(norm), 1e-12);
  std::vector<float> out(c);
  for (int k = 0; k < c; ++k) out[k] = static_cast<float>(sum[k] / norm);
  return out;
}

TensorPtr FlexGlue::search0(Ops& ops, const TensorPtr& features) const {
  return ops.l2_normalize(lin(ops, "desc0", features));
}

std::vector<float> FlexGlue::grid_scores(Ops& ops, const std::vector<std::pair<Image0, Image0>>& pairs, int grid) const {
  std::vector<float> pts(static_cast<size_t>(grid) * grid * 2);
  for (int y = 0; y < grid; ++y)
    for (int x = 0; x < grid; ++x) {
      pts[2 * (y * grid + x)] = (x + 0.5f) / grid * 2 - 1;
      pts[2 * (y * grid + x) + 1] = (y + 0.5f) / grid * 2 - 1;
    }
  const int n = grid * grid;
  std::vector<float> scores(pairs.size(), 0.0f);
  // Pairs go in chunks, so a GPU backend has every head of a chunk queued before the first is read; the
  // way back needs the way there on the host (offsets are added there).
  const size_t chunk = 8;
  std::vector<HeadWork> work;
  std::vector<std::vector<float>> there;
  std::vector<float> back, conf;
  for (size_t first = 0; first < pairs.size(); first += chunk) {
    const size_t count = std::min(chunk, pairs.size() - first);
    // Direction d of pair p: from image i to image j.
    auto ends = [&](size_t k, const Image0** i, const Image0** j) {
      const std::pair<Image0, Image0>& pr = pairs[first + k / 2];
      *i = k % 2 ? &pr.second : &pr.first, *j = k % 2 ? &pr.first : &pr.second;
    };
    work.assign(2 * count, HeadWork());
    there.assign(2 * count, std::vector<float>());
    const Image0 *i, *j;
    for (size_t k = 0; k < work.size(); ++k) ends(k, &i, &j), work[k] = head0_queue(ops, pts, i->features, j->features, j->search);
    ops.flush();
    for (size_t k = 0; k < work.size(); ++k) head_read(work[k], &there[k], &conf);
    for (size_t k = 0; k < work.size(); ++k) ends(k, &i, &j), work[k] = head0_queue(ops, there[k], j->features, i->features, i->search);
    ops.flush();
    for (size_t k = 0; k < work.size(); ++k) {
      ends(k, &i, &j);
      head_read(work[k], &back, &conf);
      // Error in pixels of image i's network input; one cell is 8 of them.
      const float sx = 4.0f * i->features->w, sy = 4.0f * i->features->h;
      int returned = 0;
      for (int a = 0; a < n; ++a)
        if (std::hypot((back[2 * a] - pts[2 * a]) * sx, (back[2 * a + 1] - pts[2 * a + 1]) * sy) < 8.0f) ++returned;
      scores[first + k / 2] += 0.5f * static_cast<float>(returned) / n;
    }
  }
  return scores;
}

}  // namespace jmt
