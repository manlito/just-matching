#include "core/checkpoint.h"

#include <cstdint>
#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/half.h"

namespace jmt {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("checkpoint: " + what); }

uint64_t le(const uint8_t* p, int bytes) {
  uint64_t v = 0;
  for (int i = 0; i < bytes; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}

// ---- The zip archive of torch.save: entries are stored, not compressed.

struct Entry {
  const uint8_t* data;
  size_t size;
};

std::map<std::string, Entry> read_zip(const uint8_t* p, size_t size) {
  if (size < 22) fail("not a zip archive");
  size_t eocd = std::string::npos;
  for (size_t i = size - 22 + 1; i-- > 0 && size - i <= 22 + 65535;)
    if (le(p + i, 4) == 0x06054b50) {
      eocd = i;
      break;
    }
  if (eocd == std::string::npos) fail("no end of central directory");
  uint64_t n_entries = le(p + eocd + 10, 2), cd_offset = le(p + eocd + 16, 4);
  if (n_entries == 0xffff || cd_offset == 0xffffffff) {  // zip64
    if (eocd < 20 || le(p + eocd - 20, 4) != 0x07064b50) fail("no zip64 locator");
    uint64_t z = le(p + eocd - 20 + 8, 8);
    if (z > size - 56 || le(p + z, 4) != 0x06064b50) fail("bad zip64 end of central directory");
    n_entries = le(p + z + 32, 8);
    cd_offset = le(p + z + 48, 8);
  }
  std::map<std::string, Entry> entries;
  size_t pos = cd_offset;
  for (uint64_t e = 0; e < n_entries; ++e) {
    if (pos > size - 46 || le(p + pos, 4) != 0x02014b50) fail("bad central directory");
    const int method = static_cast<int>(le(p + pos + 10, 2));
    uint64_t csize = le(p + pos + 20, 4), usize = le(p + pos + 24, 4);
    const size_t name_len = le(p + pos + 28, 2), extra_len = le(p + pos + 30, 2), comment_len = le(p + pos + 32, 2);
    uint64_t local = le(p + pos + 42, 4);
    if (pos + 46 + name_len + extra_len + comment_len > size) fail("bad central directory");
    std::string name(reinterpret_cast<const char*>(p + pos + 46), name_len);
    for (size_t x = pos + 46 + name_len, end = x + extra_len; x + 4 <= end;) {  // zip64 sizes and offset
      const uint64_t id = le(p + x, 2), len = le(p + x + 2, 2);
      size_t f = x + 4;
      if (id == 1) {
        if (usize == 0xffffffff && f + 8 <= end) usize = le(p + f, 8), f += 8;
        if (csize == 0xffffffff && f + 8 <= end) csize = le(p + f, 8), f += 8;
        if (local == 0xffffffff && f + 8 <= end) local = le(p + f, 8), f += 8;
      }
      x += 4 + len;
    }
    if (method != 0 || csize != usize) fail("compressed entry '" + name + "'");
    if (local > size - 30 || le(p + local, 4) != 0x04034b50) fail("bad local header of '" + name + "'");
    const size_t start = local + 30 + le(p + local + 26, 2) + le(p + local + 28, 2);
    if (start > size || csize > size - start) fail("truncated entry '" + name + "'");
    entries[name] = {p + start, static_cast<size_t>(csize)};
    pos += 46 + name_len + extra_len + comment_len;
  }
  return entries;
}

// ---- The pickle of the state dict: the opcodes torch.save writes for a dict of tensors.

struct Value;
using Ref = std::shared_ptr<Value>;
struct Value {
  enum Kind { None, Bool, Int, Float, Str, Tuple, List, Dict, Global, Storage, Tensor, Mark } kind = None;
  int64_t i = 0;
  double f = 0;
  std::string s, s2;                        // Str; Global: module, name; Storage: dtype name, key
  std::vector<Ref> items;                   // Tuple, List
  std::vector<std::pair<Ref, Ref>> pairs;   // Dict
  Ref storage;                              // Tensor
  int64_t offset = 0;
  std::vector<int64_t> shape, stride;
};

Ref make(Value::Kind k) {
  auto v = std::make_shared<Value>();
  v->kind = k;
  return v;
}

int64_t as_int(const Ref& v) {
  if (v->kind != Value::Int && v->kind != Value::Bool) fail("expected an integer in the pickle");
  return v->i;
}

std::vector<int64_t> as_ints(const Ref& v) {
  if (v->kind != Value::Tuple) fail("expected a tuple in the pickle");
  std::vector<int64_t> out;
  for (const Ref& x : v->items) out.push_back(as_int(x));
  return out;
}

Ref unpickle(const uint8_t* p, size_t size) {
  std::vector<Ref> stack;
  std::map<uint64_t, Ref> memo;
  size_t pos = 0;
  auto need = [&](size_t n) {
    if (pos + n > size) fail("truncated pickle");
  };
  auto pop = [&]() {
    if (stack.empty()) fail("pickle stack underflow");
    Ref v = stack.back();
    stack.pop_back();
    return v;
  };
  auto pop_mark = [&]() {  // the items above the last mark
    size_t m = stack.size();
    while (m > 0 && stack[m - 1]->kind != Value::Mark) --m;
    if (m == 0) fail("pickle mark missing");
    std::vector<Ref> items(stack.begin() + m, stack.end());
    stack.resize(m - 1);
    return items;
  };
  auto line = [&]() {
    size_t end = pos;
    while (end < size && p[end] != '\n') ++end;
    if (end == size) fail("truncated pickle");
    std::string s(reinterpret_cast<const char*>(p + pos), end - pos);
    pos = end + 1;
    return s;
  };
  auto tuple = [&](std::vector<Ref> items) {
    Ref t = make(Value::Tuple);
    t->items = std::move(items);
    return t;
  };
  auto integer = [&](int64_t i) {
    Ref v = make(Value::Int);
    v->i = i;
    return v;
  };

  for (;;) {
    need(1);
    const uint8_t op = p[pos++];
    switch (op) {
      case 0x80: need(1); pos += 1; break;                                   // PROTO
      case '.': if (stack.size() != 1) fail("bad pickle end"); return stack[0];  // STOP
      case '(': stack.push_back(make(Value::Mark)); break;                   // MARK
      case 'N': stack.push_back(make(Value::None)); break;                   // NONE
      case 0x88: case 0x89: {                                                // NEWTRUE, NEWFALSE
        Ref b = make(Value::Bool);
        b->i = op == 0x88;
        stack.push_back(b);
        break;
      }
      case 'K': need(1); stack.push_back(integer(p[pos])); pos += 1; break;  // BININT1
      case 'M': need(2); stack.push_back(integer(static_cast<int64_t>(le(p + pos, 2)))); pos += 2; break;  // BININT2
      case 'J': need(4); stack.push_back(integer(static_cast<int32_t>(le(p + pos, 4)))); pos += 4; break;  // BININT
      case 0x8a: {                                                           // LONG1
        need(1);
        const size_t n = p[pos++];
        need(n);
        if (n > 8) fail("integer too large in the pickle");
        int64_t v = 0;
        for (size_t i = 0; i < n; ++i) v |= static_cast<int64_t>(p[pos + i]) << (8 * i);
        if (n > 0 && n < 8 && (p[pos + n - 1] & 0x80)) v -= int64_t(1) << (8 * n);
        stack.push_back(integer(v));
        pos += n;
        break;
      }
      case 'G': {                                                            // BINFLOAT, big-endian
        need(8);
        uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) bits = bits << 8 | p[pos + i];
        Ref v = make(Value::Float);
        std::memcpy(&v->f, &bits, 8);
        stack.push_back(v);
        pos += 8;
        break;
      }
      case 'X': case 0x8c: {                                                 // BINUNICODE, SHORT_BINUNICODE
        const int w = op == 'X' ? 4 : 1;
        need(w);
        const size_t n = le(p + pos, w);
        pos += w;
        need(n);
        Ref v = make(Value::Str);
        v->s.assign(reinterpret_cast<const char*>(p + pos), n);
        stack.push_back(v);
        pos += n;
        break;
      }
      case 'c': {                                                            // GLOBAL
        Ref g = make(Value::Global);
        g->s = line();
        g->s2 = line();
        stack.push_back(g);
        break;
      }
      case ')': stack.push_back(tuple({})); break;                           // EMPTY_TUPLE
      case 0x85: case 0x86: case 0x87: {                                     // TUPLE1..3
        const size_t n = op - 0x84;
        if (stack.size() < n) fail("pickle stack underflow");
        std::vector<Ref> items(stack.end() - n, stack.end());
        stack.resize(stack.size() - n);
        stack.push_back(tuple(std::move(items)));
        break;
      }
      case 't': stack.push_back(tuple(pop_mark())); break;                   // TUPLE
      case '}': stack.push_back(make(Value::Dict)); break;                   // EMPTY_DICT
      case ']': stack.push_back(make(Value::List)); break;                   // EMPTY_LIST
      case 'a': {                                                            // APPEND
        Ref v = pop();
        if (stack.empty() || stack.back()->kind != Value::List) fail("APPEND without a list");
        stack.back()->items.push_back(v);
        break;
      }
      case 'e': {                                                            // APPENDS
        std::vector<Ref> items = pop_mark();
        if (stack.empty() || stack.back()->kind != Value::List) fail("APPENDS without a list");
        for (Ref& v : items) stack.back()->items.push_back(v);
        break;
      }
      case 's': {                                                            // SETITEM
        Ref v = pop(), k = pop();
        if (stack.empty() || stack.back()->kind != Value::Dict) fail("SETITEM without a dict");
        stack.back()->pairs.emplace_back(k, v);
        break;
      }
      case 'u': {                                                            // SETITEMS
        std::vector<Ref> items = pop_mark();
        if (items.size() % 2 || stack.empty() || stack.back()->kind != Value::Dict) fail("bad SETITEMS");
        for (size_t i = 0; i < items.size(); i += 2) stack.back()->pairs.emplace_back(items[i], items[i + 1]);
        break;
      }
      case 'q': need(1); if (stack.empty()) fail("BINPUT on an empty stack"); memo[p[pos]] = stack.back(); pos += 1; break;
      case 'r': need(4); if (stack.empty()) fail("BINPUT on an empty stack"); memo[le(p + pos, 4)] = stack.back(); pos += 4; break;
      case 'h': case 'j': {                                                  // BINGET, LONG_BINGET
        const int w = op == 'h' ? 1 : 4;
        need(w);
        auto it = memo.find(le(p + pos, w));
        if (it == memo.end()) fail("pickle memo miss");
        stack.push_back(it->second);
        pos += w;
        break;
      }
      case 'Q': {                                                            // BINPERSID: a storage
        Ref pid = pop();
        if (pid->kind != Value::Tuple || pid->items.size() < 5 || pid->items[0]->kind != Value::Str ||
            pid->items[0]->s != "storage" || pid->items[1]->kind != Value::Global || pid->items[2]->kind != Value::Str)
          fail("unsupported persistent id");
        Ref st = make(Value::Storage);
        st->s = pid->items[1]->s2;  // FloatStorage, HalfStorage, BFloat16Storage
        st->s2 = pid->items[2]->s;  // key: the file data/<key>
        st->i = as_int(pid->items[4]);  // number of elements
        stack.push_back(st);
        break;
      }
      case 'R': {                                                            // REDUCE
        Ref args = pop(), fn = pop();
        if (fn->kind != Value::Global || args->kind != Value::Tuple) fail("unsupported REDUCE");
        if (fn->s == "collections" && fn->s2 == "OrderedDict") {
          stack.push_back(make(Value::Dict));
        } else if (fn->s == "torch._utils" && fn->s2 == "_rebuild_tensor_v2") {
          if (args->items.size() < 4 || args->items[0]->kind != Value::Storage) fail("bad _rebuild_tensor_v2");
          Ref t = make(Value::Tensor);
          t->storage = args->items[0];
          t->offset = as_int(args->items[1]);
          t->shape = as_ints(args->items[2]);
          t->stride = as_ints(args->items[3]);
          stack.push_back(t);
        } else {
          fail("unsupported object " + fn->s + "." + fn->s2);
        }
        break;
      }
      case 'b': pop(); break;                                                // BUILD: state of an OrderedDict
      default: fail("unsupported pickle opcode " + std::to_string(op));
    }
  }
}

// ---- The state dict as float32 tensors.

std::vector<OwnedTensor> read_state_dict(const uint8_t* p, size_t size) {
  std::map<std::string, Entry> zip = read_zip(p, size);
  std::string prefix;
  for (const auto& kv : zip) {
    const std::string& n = kv.first;
    if (n.size() >= 8 && n.compare(n.size() - 8, 8, "data.pkl") == 0) {
      prefix = n.substr(0, n.size() - 8);
      break;
    }
  }
  auto pkl = zip.find(prefix + "data.pkl");
  if (pkl == zip.end()) fail("no data.pkl: not a torch.save file");
  auto order = zip.find(prefix + "byteorder");
  if (order != zip.end() && std::string(reinterpret_cast<const char*>(order->second.data), order->second.size) != "little")
    fail("big-endian checkpoints are not supported");
  Ref root = unpickle(pkl->second.data, pkl->second.size);
  if (root->kind != Value::Dict) fail("the checkpoint is not a state dict");

  std::vector<OwnedTensor> out;
  for (const auto& kv : root->pairs) {
    if (kv.first->kind != Value::Str || kv.second->kind != Value::Tensor) fail("the checkpoint is not a state dict");
    const Value& t = *kv.second;
    const Value& st = *t.storage;
    const int elem = st.s == "FloatStorage" ? 4 : (st.s == "HalfStorage" || st.s == "BFloat16Storage") ? 2 : 0;
    if (!elem) fail("tensor '" + kv.first->s + "' is a " + st.s);
    auto data = zip.find(prefix + "data/" + st.s2);
    if (data == zip.end() || data->second.size != static_cast<size_t>(st.i) * elem) fail("bad storage " + st.s2);
    OwnedTensor o;
    o.name = kv.first->s;
    size_t count = 1;
    for (int64_t d : t.shape) {
      if (d < 0) fail("bad shape of '" + o.name + "'");
      o.shape.push_back(static_cast<int>(d));
      count *= static_cast<size_t>(d);
    }
    if (t.stride.size() != t.shape.size()) fail("bad strides of '" + o.name + "'");
    o.data.resize(count);
    const uint8_t* src = data->second.data;
    std::vector<int64_t> index(t.shape.size(), 0);
    for (size_t i = 0; i < count; ++i) {  // any strides; the released file is contiguous
      int64_t at = t.offset;
      for (size_t d = 0; d < index.size(); ++d) at += index[d] * t.stride[d];
      if (at < 0 || at >= st.i) fail("tensor '" + o.name + "' outside its storage");
      const uint8_t* e = src + static_cast<size_t>(at) * elem;
      if (elem == 4) {
        uint32_t bits = static_cast<uint32_t>(le(e, 4));
        std::memcpy(&o.data[i], &bits, 4);
      } else if (st.s == "HalfStorage") {
        o.data[i] = half_to_float(static_cast<uint16_t>(le(e, 2)));
      } else {
        uint32_t bits = static_cast<uint32_t>(le(e, 2)) << 16;
        std::memcpy(&o.data[i], &bits, 4);
      }
      for (size_t d = index.size(); d-- > 0;) {
        if (++index[d] < t.shape[d]) break;
        index[d] = 0;
      }
    }
    out.push_back(std::move(o));
  }
  return out;
}

bool ends_with(const std::string& s, const std::string& end) {
  return s.size() >= end.size() && s.compare(s.size() - end.size(), end.size(), end) == 0;
}

// Rows of self_attn.Wqkv from (head, dim, q|k|v) to (q|k|v, head, dim), as in both originals' forward.
void reorder_wqkv(OwnedTensor& t, size_t heads) {
  const size_t rows = t.shape.empty() ? 0 : t.shape[0];
  if (!rows || rows % (3 * heads)) fail("shape of '" + t.name + "'");
  const size_t row = t.data.size() / rows, head_dim = rows / (3 * heads);
  std::vector<float> v(t.data.size());
  for (size_t s = 0; s < 3; ++s)
    for (size_t h = 0; h < heads; ++h)
      for (size_t d = 0; d < head_dim; ++d) {
        const size_t to = (s * heads + h) * head_dim + d, from = (h * head_dim + d) * 3 + s;
        std::memcpy(&v[to * row], &t.data[from * row], row * sizeof(float));
      }
  t.data = std::move(v);
}

}  // namespace

bool is_torch_checkpoint(const void* data, size_t size) {
  return size >= 4 && le(static_cast<const uint8_t*>(data), 4) == 0x04034b50;
}

namespace {

Weights flexglue_from_state_dict(std::vector<OwnedTensor> sd) {
  const size_t kHeads = 8;
  std::map<std::string, const OwnedTensor*> gammas;  // block prefix -> layer scale
  for (const OwnedTensor& t : sd)
    if (ends_with(t.name, ".gamma")) gammas[t.name.substr(0, t.name.size() - 5)] = &t;
  if (gammas.empty()) fail("no ConvNeXt layer scales: not FlexGlue's checkpoint");

  std::vector<OwnedTensor> out;
  for (OwnedTensor& t : sd) {
    if (ends_with(t.name, ".gamma")) continue;
    // ConvNeXt layer scale folded into pwconv2 (rows of the weight, and the bias).
    for (const auto& g : gammas) {
      const bool w = t.name == g.first + "pwconv2.weight", b = t.name == g.first + "pwconv2.bias";
      if (!w && !b) continue;
      const std::vector<float>& s = g.second->data;
      if (t.shape.empty() || static_cast<size_t>(t.shape[0]) != s.size()) fail("layer scale of '" + t.name + "'");
      const size_t row = t.data.size() / s.size();
      for (size_t r = 0; r < s.size(); ++r)
        for (size_t c = 0; c < row; ++c) t.data[r * row + c] = t.data[r * row + c] * s[r];
    }
    if (ends_with(t.name, "self_attn.Wqkv.weight") || ends_with(t.name, "self_attn.Wqkv.bias"))
      reorder_wqkv(t, kHeads);
    // 1x1 convolutions as (out, in) matrices; scalars (the readout temperatures) as one value, as in the file.
    if (t.shape.size() == 4 && t.shape[2] == 1 && t.shape[3] == 1) t.shape.resize(2);
    if (t.shape.empty()) t.shape = {1};
    // Names.
    std::string name = t.name;
    for (size_t at; (at = name.find(".disambiguate.")) != std::string::npos;)
      name.replace(at, 14, ".covisibility.");
    static const std::pair<const char*, const char*> kPrefixes[] = {{"extractor.backbone.", "backbone."},
                                                                     {"extractor.head.", "neck."},
                                                                     {"extractor.descriptor_head.", "desc0."},
                                                                     {"matcher.", ""}};
    for (const auto& pr : kPrefixes)
      if (name.compare(0, std::strlen(pr.first), pr.first) == 0) {
        name = pr.second + name.substr(std::strlen(pr.first));
        break;
      }
    t.name = name;
    out.push_back(std::move(t));
  }
  std::map<std::string, std::string> meta = {
      {"kind", "weights"},       {"model", "flexglue"},          {"n_layers", "9"},     {"dim", "512"},
      {"heads", "8"},            {"feature_dim", "256"},         {"search_dim", "64"},  {"stride", "8"},
      {"depths", "3,3,27,3"},    {"widths", "128,256,512,1024"}, {"source", "flexglue.pth"},
      {"license", "backbone.*: DINOv3 License; everything else: MIT"}};
  return Weights::from_tensors(std::move(meta), std::move(out));
}

Weights lightglue_from_state_dict(std::vector<OwnedTensor> sd) {
  const size_t kHeads = 4;
  std::vector<OwnedTensor> out;
  for (OwnedTensor& t : sd) {
    if (t.name == "confidence_thresholds") continue;  // a buffer the graph computes
    if (ends_with(t.name, "self_attn.Wqkv.weight") || ends_with(t.name, "self_attn.Wqkv.bias"))
      reorder_wqkv(t, kHeads);
    out.push_back(std::move(t));
  }
  const auto input = std::find_if(out.begin(), out.end(), [](const OwnedTensor& t) { return t.name == "input_proj.weight"; });
  if (input == out.end() || input->shape != std::vector<int>{256, 128})
    fail("not a LightGlue checkpoint for 128-dim descriptors (input_proj.weight)");
  std::map<std::string, std::string> meta = {{"kind", "weights"},  {"model", "lightglue"}, {"n_layers", "9"},
                                             {"dim", "256"},       {"heads", "4"},         {"input_dim", "128"},
                                             {"license", "Apache-2.0"}, {"source", "LightGlue checkpoint"}};
  return Weights::from_tensors(std::move(meta), std::move(out));
}

bool has_layer_scales(const std::vector<OwnedTensor>& sd) {
  return std::any_of(sd.begin(), sd.end(), [](const OwnedTensor& t) { return ends_with(t.name, ".gamma"); });
}

}  // namespace

Weights flexglue_from_checkpoint(const void* data, size_t size) {
  return flexglue_from_state_dict(read_state_dict(static_cast<const uint8_t*>(data), size));
}

Weights lightglue_from_checkpoint(const void* data, size_t size) {
  return lightglue_from_state_dict(read_state_dict(static_cast<const uint8_t*>(data), size));
}

Weights weights_from_checkpoint(const void* data, size_t size) {
  // FlexGlue's backbone has ConvNeXt layer scales; LightGlue has none.
  std::vector<OwnedTensor> sd = read_state_dict(static_cast<const uint8_t*>(data), size);
  return has_layer_scales(sd) ? flexglue_from_state_dict(std::move(sd)) : lightglue_from_state_dict(std::move(sd));
}

}  // namespace jmt
