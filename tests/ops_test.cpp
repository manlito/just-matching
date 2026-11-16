// Compares each operation of a GPU backend with the CPU backend on its own, on random tensors and
// weights, so that a wrong kernel is named directly (golden_test sees only whole stages). Needs no
// weights or golden files.
//   ops_test metal|cuda|webgpu[:fp16]
// Tolerance relative to the largest value of the CPU result: 1e-5 in FP32, 1e-2 in FP16.
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ops.h"

namespace {

int g_failed = 0;
double g_tol = 1e-5;

// Random values that stay alive for the whole run (Param points into them).
struct Random {
  std::vector<std::unique_ptr<std::vector<float>>> storage;
  std::vector<std::unique_ptr<jmt::Param>> params;
  uint32_t seed = 7;
  float next() {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>(seed >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
  }
  std::vector<float> values(size_t n, float scale) {
    std::vector<float> v(n);
    for (float& x : v) x = next() * scale;
    return v;
  }
  const jmt::Param& param(std::vector<int> shape, float scale, float offset = 0) {
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    storage.push_back(std::make_unique<std::vector<float>>(values(n, scale)));
    for (float& x : *storage.back()) x += offset;
    params.push_back(std::make_unique<jmt::Param>());
    params.back()->shape = shape;
    params.back()->data = storage.back()->data();
    return *params.back();
  }
};

std::vector<float> values(const jmt::TensorPtr& t) {
  std::vector<float> v(t->size());
  t->read(v.data());
  return v;
}

void check_vec(const std::string& name, const std::vector<float>& a, const std::vector<float>& b) {
  double d = a.size() == b.size() ? 0 : INFINITY, ref = 0;
  size_t at = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    ref = std::fmax(ref, std::fabs(a[i]));
    const double e = std::fabs(static_cast<double>(a[i]) - b[i]);
    if (e > d || std::isnan(e)) d = e, at = i;
  }
  const bool ok = d <= g_tol * std::fmax(ref, 1e-30);
  std::printf("%-40s %s  max|diff| %.2e  (max|ref| %.2e)", name.c_str(), ok ? "ok  " : "FAIL", d, ref);
  if (!ok && at < a.size() && at < b.size()) std::printf("  at %zu: cpu %g gpu %g", at, a[at], b[at]);
  std::printf("\n");
  g_failed += !ok;
}

void check(const std::string& name, const jmt::TensorPtr& a, const jmt::TensorPtr& b) {
  if (a->h != b->h || a->w != b->w || a->c != b->c) {
    std::printf("%-40s FAIL (shape: cpu %dx%dx%d, gpu %dx%dx%d)\n", name.c_str(), a->h, a->w, a->c, b->h, b->w, b->c);
    ++g_failed;
    return;
  }
  check_vec(name, values(a), values(b));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: ops_test metal|cuda|webgpu[:fp16]\n");
    return 2;
  }
  const std::string backend = argv[1];
  const bool fp16 = backend.find(":fp16") != std::string::npos;
  if (fp16) g_tol = 1e-2;
  try {
    std::unique_ptr<jmt::Ops> gpu;
    const std::string kind = backend.substr(0, backend.find(':'));
#ifdef JMT_HAVE_METAL
    if (kind == "metal") gpu = jmt::make_metal_ops(0, fp16);
#endif
#ifdef JMT_HAVE_CUDA
    if (kind == "cuda") gpu = jmt::make_cuda_ops(0, fp16);
#endif
#ifdef JMT_HAVE_WEBGPU
    if (kind == "webgpu") gpu = jmt::make_webgpu_ops(fp16);
#endif
    if (!gpu) throw std::runtime_error("this build has no " + backend + " backend");
    std::unique_ptr<jmt::Ops> cpu = jmt::make_cpu_ops(0);
    Random rnd;
    // The same tensor on both backends.
    auto both = [&](int h, int w, int c, float scale) {
      std::vector<float> v = rnd.values(static_cast<size_t>(h) * w * c, scale);
      jmt::TensorPtr a = cpu->upload(v.data(), h, w, c);
      jmt::TensorPtr b = gpu->upload(v.data(), h, w, c);
      if (fp16) {  // the GPU stores half: give the CPU the same rounded values
        std::vector<float> back = values(b);
        a = cpu->upload(back.data(), h, w, c);
      }
      return std::make_pair(a, b);
    };

    {
      auto [a, b] = both(37, 29, 3, 1.0f);
      const float mean[3] = {0.4f, 0.5f, 0.6f}, sd[3] = {0.2f, 0.25f, 0.3f};
      std::vector<unsigned char> img(53 * 41 * 4);
      for (auto& p : img) p = static_cast<unsigned char>((rnd.next() + 1) * 127.5f);
      for (int ch : {1, 3, 4})
        check("preprocess_u8 ch" + std::to_string(ch), cpu->preprocess_u8(img.data(), 41, 53, 53 * 4, ch, 32, 32, mean, sd),
              gpu->preprocess_u8(img.data(), 41, 53, 53 * 4, ch, 32, 32, mean, sd));
    }
    for (int n : {1, 37, 200}) {
      for (auto io : {std::make_pair(3, 7), std::make_pair(64, 64), std::make_pair(130, 70), std::make_pair(512, 1536)}) {
        auto [a, b] = both(1, n, io.first, 1.0f);
        const jmt::Param& wt = rnd.param({io.second, io.first}, 1.0f / std::sqrt(static_cast<float>(io.first)));
        const jmt::Param& bias = rnd.param({io.second}, 0.5f);
        const std::string tag = " n" + std::to_string(n) + " " + std::to_string(io.first) + "->" + std::to_string(io.second);
        check("linear" + tag, cpu->linear(a, wt, &bias), gpu->linear(b, wt, &bias));
        check("linear no bias" + tag, cpu->linear(a, wt, nullptr), gpu->linear(b, wt, nullptr));
        auto [ra, rb] = both(1, n, io.second, 1.0f);
        check("linear gelu + residual" + tag, cpu->linear_fused(a, wt, &bias, jmt::Act::Gelu, ra),
              gpu->linear_fused(b, wt, &bias, jmt::Act::Gelu, rb));
      }
    }
    {
      auto [a, b] = both(32, 24, 3, 1.0f);
      const jmt::Param& wt = rnd.param({128, 3, 4, 4}, 0.3f);
      const jmt::Param& bias = rnd.param({128}, 0.5f);
      check("conv 4x4 stride 4", cpu->conv(a, wt, bias, 4, 4, 0), gpu->conv(b, wt, bias, 4, 4, 0));
      auto [c, d] = both(12, 10, 64, 1.0f);
      const jmt::Param& w3 = rnd.param({96, 64, 3, 3}, 0.05f);
      const jmt::Param& b3 = rnd.param({96}, 0.5f);
      check("conv 3x3 pad 1", cpu->conv(c, w3, b3, 3, 1, 1), gpu->conv(d, w3, b3, 3, 1, 1));
      const jmt::Param& w2 = rnd.param({128, 64, 2, 2}, 0.05f);
      const jmt::Param& b2 = rnd.param({128}, 0.5f);
      check("conv 2x2 stride 2", cpu->conv(c, w2, b2, 2, 2, 0), gpu->conv(d, w2, b2, 2, 2, 0));
    }
    {
      auto [a, b] = both(13, 11, 128, 1.0f);
      const jmt::Param& wt = rnd.param({128, 1, 7, 7}, 0.2f);
      const jmt::Param& bias = rnd.param({128}, 0.5f);
      check("dwconv7", cpu->dwconv7(a, wt, bias), gpu->dwconv7(b, wt, bias));
    }
    for (int c : {128, 512, 1024}) {
      for (int g : {64, c}) {
        auto [a, b] = both(9, 7, c, 2.0f);
        const jmt::Param& gain = rnd.param({g}, 0.5f, 1.0f);
        const jmt::Param& bias = rnd.param({g}, 0.5f);
        const std::string tag = " c" + std::to_string(c) + " g" + std::to_string(g);
        check("layer_norm" + tag, cpu->layer_norm(a, gain, bias, 1e-6f), gpu->layer_norm(b, gain, bias, 1e-6f));
        check("layer_norm relu" + tag, cpu->layer_norm_act(a, gain, bias, 1e-5f, jmt::Act::Relu),
              gpu->layer_norm_act(b, gain, bias, 1e-5f, jmt::Act::Relu));
      }
    }
    {
      auto [a, b] = both(6, 10, 32, 3.0f);
      auto [c, d] = both(6, 10, 32, 3.0f);
      auto [e, f] = both(6, 10, 16, 3.0f);
      check("gelu", cpu->gelu(a), gpu->gelu(b));
      check("relu", cpu->relu(a), gpu->relu(b));
      check("add", cpu->add(a, c), gpu->add(b, d));
      check("concat", cpu->concat(a, e), gpu->concat(b, f));
      check("slice", cpu->slice(a, 8, 12), gpu->slice(b, 8, 12));
      check("pixel_unshuffle2", cpu->pixel_unshuffle2(a), gpu->pixel_unshuffle2(b));
      check("resize up", cpu->resize(a, 13, 21), gpu->resize(b, 13, 21));
      check("resize down", cpu->resize(a, 3, 4), gpu->resize(b, 3, 4));
      check("l2_normalize", cpu->l2_normalize(a), gpu->l2_normalize(b));
      std::vector<float> pts = rnd.values(2 * 50, 1.2f);
      check("sample", cpu->sample(a, pts.data(), 50), gpu->sample(b, pts.data(), 50));
    }
    {
      const int heads = 8, n = 60, pairs = 32;
      auto [a, b] = both(6, 10, heads * pairs * 2, 2.0f);
      std::vector<float> cs = rnd.values(static_cast<size_t>(n) * pairs, 1.0f), sn = rnd.values(cs.size(), 1.0f);
      check("rotary", cpu->rotary(a, cs, sn, heads), gpu->rotary(b, cs, sn, heads));
    }
    for (auto ps : {std::make_pair(1, 1), std::make_pair(37, 53), std::make_pair(200, 131), std::make_pair(1024, 1024)}) {
      const int heads = 8;
      auto [qa, qb] = both(1, ps.first, heads * 64, 3.0f);
      auto [ka, kb] = both(1, ps.second, heads * 64, 3.0f);
      auto [va, vb] = both(1, ps.second, heads * 64, 2.0f);
      check("attention P" + std::to_string(ps.first) + " S" + std::to_string(ps.second), cpu->attention(qa, ka, va, heads),
            gpu->attention(qb, kb, vb, heads));
    }
    for (int side : {8, 32, 64}) {
      const int N = 300, d = 64;
      auto [qa, qb] = both(1, N, d, 1.0f);
      auto [sa, sb] = both(side, side, d, 1.0f);
      qa = cpu->l2_normalize(qa), sa = cpu->l2_normalize(sa), qb = gpu->l2_normalize(qb), sb = gpu->l2_normalize(sb);
      if (fp16) {  // the same rounded inputs
        std::vector<float> q = values(qb), s = values(sb);
        qa = cpu->upload(q.data(), 1, N, d), sa = cpu->upload(s.data(), side, side, d);
      }
      std::vector<float> c0, m0;
      cpu->readout(qa, sa, 30.0f, &c0, &m0);
      const std::string tag = " " + std::to_string(side) + "x" + std::to_string(side);
      std::vector<float> c1, m1;
      gpu->readout(qb, sb, 30.0f, &c1, &m1);
      check_vec("readout coarse" + tag, c0, c1);
      check_vec("readout mass" + tag, m0, m1);
      if (gpu->has_device_readout()) {
        jmt::TensorPtr ct, mt;
        gpu->readout_device(qb, sb, 30.0f, &ct, &mt);
        check_vec("readout_device coarse" + tag, c0, values(ct));
        check_vec("readout_device mass" + tag, m0, values(mt));
        check("sample_at" + tag, cpu->sample(sa, c0.data(), N), gpu->sample_at(sb, ct));
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf(g_failed ? "FAILED (%d)\n" : "all ok\n", g_failed);
  return g_failed ? 1 : 0;
}
