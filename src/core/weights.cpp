#include "core/weights.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace jmt {
namespace {

const char kMagic[4] = {'J', 'K', 'P', 'W'};
const uint32_t kVersion = 1;
const size_t kAlign = 64;

struct Reader {
  const uint8_t* p;
  size_t size, pos = 0;

  void need(size_t n) const {
    if (pos + n > size) throw std::runtime_error("weights: truncated file");
  }
  // The format is little-endian; assemble byte by byte so this works on any host.
  uint64_t uint(int bytes) {
    need(bytes);
    uint64_t v = 0;
    for (int i = 0; i < bytes; ++i) v |= static_cast<uint64_t>(p[pos + i]) << (8 * i);
    pos += bytes;
    return v;
  }
  std::string str() {
    size_t n = uint(2);
    need(n);
    std::string s(reinterpret_cast<const char*>(p + pos), n);
    pos += n;
    return s;
  }
};

void put(std::string& out, uint64_t v, int bytes) {
  for (int i = 0; i < bytes; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}
void put_str(std::string& out, const std::string& s) {
  put(out, s.size(), 2);
  out += s;
}
size_t align_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

}  // namespace

Weights Weights::from_memory(const void* data, size_t size) {
  Weights w;
  w.storage_.resize((size + 3) / 4);
  std::memcpy(w.storage_.data(), data, size);
  Reader r{reinterpret_cast<const uint8_t*>(w.storage_.data()), size};

  r.need(4);
  if (std::memcmp(r.p, kMagic, 4) != 0) throw std::runtime_error("weights: not a weights container");
  r.pos = 4;
  if (r.uint(4) != kVersion) throw std::runtime_error("weights: unsupported file version");
  size_t n_meta = r.uint(4), n_tensors = r.uint(4);
  for (size_t i = 0; i < n_meta; ++i) {
    std::string k = r.str();
    w.meta_[k] = r.str();
  }
  for (size_t i = 0; i < n_tensors; ++i) {
    std::string name = r.str();
    int dtype = static_cast<int>(r.uint(1)), ndim = static_cast<int>(r.uint(1));
    Param p;
    for (int d = 0; d < ndim; ++d) p.shape.push_back(static_cast<int>(r.uint(4)));
    uint64_t offset = r.uint(8), nbytes = r.uint(8);
    if (dtype != 0) continue;  // only float32 tensors are used by the library
    if (offset % 4 != 0 || offset > size || nbytes > size - offset || nbytes != p.size() * 4)
      throw std::runtime_error("weights: bad tensor entry '" + name + "'");
    p.data = reinterpret_cast<const float*>(w.storage_.data()) + offset / 4;
    w.tensors_[name] = p;
  }
  return w;
}

Weights Weights::from_file(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("weights: cannot open '" + path + "'");
  std::vector<char> buf;
  char chunk[1 << 16];
  size_t n;
  while ((n = std::fread(chunk, 1, sizeof(chunk), f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
  std::fclose(f);
  return from_memory(buf.data(), buf.size());
}

const Param* Weights::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const Param& Weights::get(const std::string& name) const {
  const Param* p = find(name);
  if (!p) throw std::runtime_error("weights: missing tensor '" + name + "'");
  return *p;
}

const std::string& Weights::meta(const std::string& key) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) throw std::runtime_error("weights: missing metadata '" + key + "'");
  return it->second;
}

int Weights::meta_int(const std::string& key) const { return std::stoi(meta(key)); }

void write_container(const std::string& path, const std::map<std::string, std::string>& meta,
                     const std::vector<NamedTensor>& tensors) {
  std::string head(kMagic, 4);
  put(head, kVersion, 4);
  put(head, meta.size(), 4);
  put(head, tensors.size(), 4);
  for (const auto& kv : meta) {
    put_str(head, kv.first);
    put_str(head, kv.second);
  }
  size_t table = 0;
  for (const auto& t : tensors) table += 2 + t.name.size() + 2 + 4 * t.shape.size() + 16;
  size_t offset = align_up(head.size() + table);
  std::vector<size_t> offsets, sizes;
  for (const auto& t : tensors) {
    size_t count = 1;
    for (int d : t.shape) count *= static_cast<size_t>(d);
    put_str(head, t.name);
    put(head, 0, 1);  // float32
    put(head, t.shape.size(), 1);
    for (int d : t.shape) put(head, static_cast<uint64_t>(d), 4);
    put(head, offset, 8);
    put(head, count * 4, 8);
    offsets.push_back(offset);
    sizes.push_back(count);
    offset = align_up(offset + count * 4);
  }
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write '" + path + "'");
  std::string out = head;
  for (size_t i = 0; i < tensors.size(); ++i) {
    out.resize(offsets[i], '\0');
    for (size_t j = 0; j < sizes[i]; ++j) {
      uint32_t bits;
      std::memcpy(&bits, tensors[i].data + j, 4);
      put(out, bits, 4);
    }
  }
  bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
  ok = std::fclose(f) == 0 && ok;
  if (!ok) throw std::runtime_error("cannot write '" + path + "'");
}

}  // namespace jmt
