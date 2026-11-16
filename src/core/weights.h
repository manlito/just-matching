// Reader/writer for the tensor container (.jmtw weights, .jmtt tensors; the just-keypoints format). See tools/jmtfile.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace jmt {

// A float32 tensor that lives inside a loaded container (float16 tensors are widened to float32 on load).
struct Param {
  std::vector<int> shape;
  const float* data = nullptr;
  size_t size() const {
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    return n;
  }
};

class Weights {
 public:
  // Both throw std::runtime_error on malformed input. The blob is copied.
  static Weights from_memory(const void* data, size_t size);
  static Weights from_file(const std::string& path);

  const Param& get(const std::string& name) const;       // throws if missing
  const Param* find(const std::string& name) const;      // nullptr if missing
  const std::string& meta(const std::string& key) const; // throws if missing
  int meta_int(const std::string& key) const;

 private:
  std::vector<uint32_t> storage_;  // 4-byte aligned copy of the file
  std::deque<std::vector<float>> widened_;  // float16 tensors as float32
  std::map<std::string, Param> tensors_;
  std::map<std::string, std::string> meta_;
};

// Tensor to be written by write_container (float32 only).
struct NamedTensor {
  std::string name;
  std::vector<int> shape;
  const float* data;
};
void write_container(const std::string& path, const std::map<std::string, std::string>& meta,
                     const std::vector<NamedTensor>& tensors);


}  // namespace jmt
