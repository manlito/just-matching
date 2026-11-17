// FlexGlue's checkpoint read by the library gives the tensors of the exported weights file, bit for bit.
//   checkpoint_test flexglue.pth flexglue.jmtw
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/weights.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: checkpoint_test flexglue.pth flexglue.jmtw\n");
    return 2;
  }
  try {
    auto t0 = std::chrono::steady_clock::now();
    jmt::Weights pth = jmt::Weights::from_file(argv[1]);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    jmt::Weights ref = jmt::Weights::from_file(argv[2]);
    int bad = 0;
    size_t values = 0;
    const auto names = pth.names(), ref_names = ref.names();
    if (names != ref_names) {
      std::fprintf(stderr, "tensor names differ: %zu read, %zu in the weights file\n", names.size(), ref_names.size());
      for (const auto& n : ref_names)
        if (!pth.find(n)) std::fprintf(stderr, "  missing %s\n", n.c_str());
      for (const auto& n : names)
        if (!ref.find(n)) std::fprintf(stderr, "  extra %s\n", n.c_str());
      return 1;
    }
    for (const auto& n : names) {
      const jmt::Param &a = pth.get(n), &b = ref.get(n);
      if (a.shape != b.shape || std::memcmp(a.data, b.data, a.size() * sizeof(float)) != 0) {
        if (bad++ < 10) std::fprintf(stderr, "differs: %s\n", n.c_str());
      }
      values += a.size();
    }
    for (const char* k : {"model", "n_layers", "dim", "heads", "feature_dim", "search_dim", "stride", "depths", "widths"})
      if (pth.meta(k) != ref.meta(k)) {
        std::fprintf(stderr, "metadata %s: %s, file %s\n", k, pth.meta(k).c_str(), ref.meta(k).c_str());
        ++bad;
      }
    std::printf("%zu tensors, %zu values, read in %.0f ms: %s\n", names.size(), values, ms,
                bad ? "DIFFERENT" : "identical");
    return bad ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
