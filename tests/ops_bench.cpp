// Times single operations of a GPU backend at the sizes of a pair at 512 px (4096 tokens of 512 channels).
//   ops_bench metal|cuda|webgpu[:fp16]
// Best of 20 runs of 5 calls each, waiting for the device at the end of each run.
#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <string>
#include <vector>
#include "core/ops.h"
using namespace jmt;
int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: ops_bench metal|cuda|webgpu[:fp16]\n");
    return 2;
  }
  const std::string b = argv[1], kind = b.substr(0, b.find(':'));
  const bool fp16 = b.find(":fp16") != std::string::npos;
  std::unique_ptr<Ops> ops;
#ifdef JMT_HAVE_METAL
  if (kind == "metal") ops = make_metal_ops(0, fp16);
#endif
#ifdef JMT_HAVE_CUDA
  if (kind == "cuda") ops = make_cuda_ops(0, fp16);
#endif
#ifdef JMT_HAVE_WEBGPU
  if (kind == "webgpu") ops = make_webgpu_ops(fp16);
#endif
  if (!ops) {
    std::fprintf(stderr, "this build has no %s backend\n", b.c_str());
    return 2;
  }
  std::vector<float> xv(4096 * 2048, 0.01f), wv(4096 * 2048, 0.02f);
  auto time = [&](const char* name, double gmacs, auto f) {
    f(); ops->finish();
    double best = 1e9;
    for (int r = 0; r < 20; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 5; ++i) f();
      ops->finish();
      best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5);
    }
    if (gmacs > 0) std::printf("%-34s %8.3f ms  %6.2f T multiply-adds/s\n", name, best, gmacs / best);
    else std::printf("%-34s %8.3f ms\n", name, best);
  };
  TensorPtr x512 = ops->upload(xv.data(), 64, 64, 512), x1024 = ops->upload(xv.data(), 64, 64, 1024);
  Param w1{{1536, 512}, wv.data()}, w2{{512, 512}, wv.data()}, w3{{512, 1024}, wv.data()}, bias{{1536}, wv.data()};
  time("linear 4096 x 512 -> 1536", 4096.0 * 512 * 1536 / 1e9, [&] { ops->linear(x512, w1, &bias); });
  time("linear 4096 x 512 -> 512", 4096.0 * 512 * 512 / 1e9, [&] { ops->linear(x512, w2, &bias); });
  time("linear 4096 x 1024 -> 512", 4096.0 * 1024 * 512 / 1e9, [&] { ops->linear(x1024, w3, &bias); });
  time("attention 4096 x 4096 x 8 heads", 2.0 * 4096 * 4096 * 512 / 1e9, [&] { ops->attention(x512, x512, x512, 8); });
  Param bias2{{2048}, wv.data()};
  Param g{{512}, wv.data()}, g64{{64}, wv.data()};
  time("layer_norm 4096 x 512", 0, [&] { ops->layer_norm(x512, g, g, 1e-5f); });
  time("layer_norm 4096 x 512 g64", 0, [&] { ops->layer_norm(x512, g64, g64, 1e-5f); });
  time("slice 4096 x 512", 0, [&] { ops->slice(x1024, 512, 512); });
  time("concat 4096 x (512 + 512)", 0, [&] { ops->concat(x512, x512); });
  std::vector<float> cs(4096 * 32, 0.5f);
  time("rotary 4096 x 512", 0, [&] { ops->rotary(x512, cs, cs, 8); });
  time("add 4096 x 512", 0, [&] { ops->add(x512, x512); });
  // The products of describe at 512 px: stage 2 (32 x 32 tokens, 27 blocks) and stage 3 (16 x 16, 3 blocks).
  TensorPtr s2 = ops->upload(xv.data(), 32, 32, 512), s2w = ops->upload(xv.data(), 32, 32, 2048);
  Param p1{{2048, 512}, wv.data()}, p2{{512, 2048}, wv.data()}, dw{{512, 1, 7, 7}, wv.data()};
  time("linear 1024 x 512 -> 2048", 1024.0 * 512 * 2048 / 1e9, [&] { ops->linear(s2, p1, &bias2); });
  time("linear 1024 x 2048 -> 512", 1024.0 * 512 * 2048 / 1e9, [&] { ops->linear(s2w, p2, &bias2); });
  time("dwconv7 32 x 32 x 512", 1024.0 * 512 * 49 / 1e9, [&] { ops->dwconv7(s2, dw, bias2); });
  time("layer_norm 1024 x 512", 0, [&] { ops->layer_norm(s2, g, g, 1e-6f); });
  return 0;
}
