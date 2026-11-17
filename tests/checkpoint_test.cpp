// A checkpoint read by the library gives the tensors of the exported weights file, bit for bit, and its
// metadata but for where it came from.
//   checkpoint_test flexglue.pth flexglue.jmtw
//   checkpoint_test aliked_lightglue.pth lightglue-aliked.jmtw
//   checkpoint_test embedded:lightglue-aliked lightglue-aliked.jmtw   (the copy built into the library)
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/weights.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: checkpoint_test CHECKPOINT.pth WEIGHTS.jmtw\n");
    return 2;
  }
  try {
    auto t0 = std::chrono::steady_clock::now();
    const std::string source = argv[1], kEmbedded = "embedded:";
    jmt::Weights pth;
    if (source.compare(0, kEmbedded.size(), kEmbedded) == 0) {
      const void* blob = nullptr;
      size_t size = 0;
      if (!jmt::embedded_weights(source.substr(kEmbedded.size()), &blob, &size))
        throw std::runtime_error(source.substr(kEmbedded.size()) + " is not built into this library");
      pth = jmt::Weights::from_memory(blob, size);
    } else {
      pth = jmt::Weights::from_file(source);
    }
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
    for (const auto& kv : ref.metadata()) {
      // Where the file came from is known to the exporter only.
      if (kv.first == "source" || kv.first == "source_commit" || kv.first == "variant") continue;
      const auto it = pth.metadata().find(kv.first);
      const std::string got = it == pth.metadata().end() ? "(missing)" : it->second;
      if (got != kv.second) {
        std::fprintf(stderr, "metadata %s: %s, file %s\n", kv.first.c_str(), got.c_str(), kv.second.c_str());
        ++bad;
      }
    }
    std::printf("%zu tensors, %zu values, read in %.0f ms: %s\n", names.size(), values, ms,
                bad ? "DIFFERENT" : "identical");
    return bad ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
