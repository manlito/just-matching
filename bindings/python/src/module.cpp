// Python bindings over the C API (include/jmt/jmt.h), as thin as just-ba's: a Matcher owns a context, Features
// hold one image's features and keep their Matcher alive, results come back as NumPy arrays (copies). Arrays go
// in through numpy.ascontiguousarray, so lists and arrays of other types and layouts are accepted.
//
// A context is not thread-safe: each Matcher has a mutex, and the GIL is released while the library works, so
// several Matchers run in parallel from several Python threads.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "core/api_internal.h"
#include "jmt/jmt.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
using In = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;
template <typename T>
using Out = nb::ndarray<nb::numpy, T>;

template <typename T>
Out<T> to_numpy(const T* src, std::size_t rows, std::size_t cols) {
  const std::size_t count = rows * cols;
  T* data = new T[std::max<std::size_t>(count, 1)];
  if (count) std::copy(src, src + count, data);
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  if (cols == 1) {
    std::size_t shape[1] = {rows};
    return Out<T>(data, 1, shape, owner);
  }
  std::size_t shape[2] = {rows, cols};
  return Out<T>(data, 2, shape, owner);
}

template <typename T>
Out<T> to_numpy(const std::vector<T>& v, std::size_t cols) {
  return to_numpy(v.data(), cols ? v.size() / cols : 0, cols);
}

// Anything NumPy can turn into a C-contiguous array of T.
template <typename T>
In<T> as_array(nb::handle h, const char* name) {
  const char* dtype = std::is_same_v<T, float> ? "float32" : "uint8";
  nb::object arr;
  try {
    arr = nb::module_::import_("numpy").attr("ascontiguousarray")(h, "dtype"_a = dtype);
  } catch (nb::python_error&) {
    throw nb::type_error((std::string(name) + ": expected an array of numbers").c_str());
  }
  return nb::cast<In<T>>(arr);
}

// Points (n, 2), copied: the library reads them while the GIL is released.
std::vector<float> points(nb::handle h, const char* name) {
  In<float> a = as_array<float>(h, name);
  if (a.ndim() != 2 || a.shape(1) != 2) throw nb::value_error((std::string(name) + " must have shape (n, 2)").c_str());
  return std::vector<float>(a.data(), a.data() + a.size());
}

[[noreturn]] void fail(const char* what) {
  const char* e = jmt_last_error();
  throw std::runtime_error(std::string(what) + ": " + (e && *e ? e : "unknown error"));
}

const char* const kBackendNames[] = {"cpu", "cuda", "metal", "webgpu"};

jmt_backend backend_of(const std::string& name) {
  for (int i = 0; i < 4; ++i)
    if (name == kBackendNames[i]) return static_cast<jmt_backend>(i);
  throw nb::value_error(("unknown backend '" + name + "' (cpu, cuda, metal, webgpu)").c_str());
}

// ---------------------------------------------------------------------------

struct Matcher {
  jmt_ctx* ctx = nullptr;
  std::string model;    // "flexglue" or "lightglue"
  std::string backend;
  std::mutex mutex;
  ~Matcher() {
    if (ctx) jmt_destroy(ctx);
  }
};

struct Features {
  jmt_features* f = nullptr;
  nb::object owner;  // the Python Matcher: GPU features must be freed before their context
  Matcher* matcher = nullptr;
  int width = 0, height = 0;
  std::optional<std::vector<float>> xy;  // LightGlue: the keypoints given to describe_keypoints
  ~Features() {
    if (f) jmt_features_free(f);
  }
};

struct Matches {
  Out<int32_t> idx;
  Out<float> score, a_in_b, b_in_a, conf_a, conf_b;
  float covisibility = 0;
  int stop_layer = 0;
};

// One result, copied out of the context's arrays while the GIL is released.
struct MatchData {
  std::vector<int32_t> idx;
  std::vector<float> score, a_in_b, b_in_a, conf_a, conf_b;
  float covisibility = 0;
  int stop_layer = 0;
  explicit MatchData(const jmt_matches& m)
      : idx(m.idx, m.idx + 2 * static_cast<std::size_t>(m.n)),
        score(m.score, m.score + m.n),
        a_in_b(m.a_in_b, m.a_in_b + 2 * static_cast<std::size_t>(m.n_a)),
        b_in_a(m.b_in_a, m.b_in_a + 2 * static_cast<std::size_t>(m.n_b)),
        conf_a(m.conf_a, m.conf_a + m.n_a),
        conf_b(m.conf_b, m.conf_b + m.n_b),
        covisibility(m.covisibility),
        stop_layer(m.stop_layer) {}
  Matches to_python() const {
    return Matches{to_numpy(idx, 2),    to_numpy(score, 1),  to_numpy(a_in_b, 2), to_numpy(b_in_a, 2),
                   to_numpy(conf_a, 1), to_numpy(conf_b, 1), covisibility,         stop_layer};
  }
};

struct MatchOptions {
  std::optional<int> n_layers;
  std::optional<float> filter_threshold, max_distance_px, min_covisibility, early_stop_decay, depth_confidence,
      width_confidence;
  jmt_match_options get() const {
    jmt_match_options o;
    jmt_match_options_default(&o);
    if (n_layers) o.n_layers = *n_layers;
    if (filter_threshold) o.filter_threshold = *filter_threshold;
    if (max_distance_px) o.max_distance_px = *max_distance_px;
    if (min_covisibility) o.min_covisibility = *min_covisibility;
    if (early_stop_decay) o.early_stop_decay = *early_stop_decay;
    if (depth_confidence) o.depth_confidence = *depth_confidence;
    if (width_confidence) o.width_confidence = *width_confidence;
    return o;
  }
};

Features& features_of(Matcher& m, nb::handle h, const char* name) {
  Features* f = nullptr;
  if (!nb::try_cast(h, f) || !f) throw nb::type_error((std::string(name) + ": expected Features").c_str());
  if (f->matcher != &m) throw nb::value_error((std::string(name) + ": features made by another Matcher").c_str());
  return *f;
}

// One pair's keypoints: given, or those of describe_keypoints.
std::vector<float> keypoints_of(const Features& f, nb::handle xy, const char* name) {
  if (!xy.is_none()) return points(xy, name);
  if (f.xy) return *f.xy;
  throw nb::value_error((std::string(name) + " is needed: FlexGlue matches the keypoints it is given").c_str());
}

std::vector<Matches> match_pairs(Matcher& m, const std::vector<std::tuple<Features*, Features*>>& fs,
                                 std::vector<std::vector<float>>& xys, const MatchOptions& options) {
  const jmt_match_options o = options.get();
  std::vector<MatchData> data;
  {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(m.mutex);
    std::vector<jmt_pair> pairs(fs.size());
    for (std::size_t i = 0; i < fs.size(); ++i) {
      const auto& xa = xys[2 * i];
      const auto& xb = xys[2 * i + 1];
      pairs[i] = {std::get<0>(fs[i])->f, xa.data(), static_cast<int>(xa.size() / 2),
                  std::get<1>(fs[i])->f, xb.data(), static_cast<int>(xb.size() / 2)};
    }
    std::vector<jmt_matches> out(fs.size());
    if (jmt_match_batch(m.ctx, pairs.data(), static_cast<int>(pairs.size()), &o, out.data()) != 0) fail("match");
    data.reserve(out.size());
    for (const auto& r : out) data.emplace_back(r);
  }
  std::vector<Matches> result;
  result.reserve(data.size());
  for (const auto& d : data) result.push_back(d.to_python());
  return result;
}

#define JMT_OPTION_ARGS                                                                                         \
  nb::kw_only(), "n_layers"_a = nb::none(), "filter_threshold"_a = nb::none(), "max_distance_px"_a = nb::none(), \
      "min_covisibility"_a = nb::none(), "early_stop_decay"_a = nb::none(), "depth_confidence"_a = nb::none(),   \
      "width_confidence"_a = nb::none()

#define JMT_OPTION_PARAMS                                                                                        \
  std::optional<int> n_layers, std::optional<float> filter_threshold, std::optional<float> max_distance_px,      \
      std::optional<float> min_covisibility, std::optional<float> early_stop_decay,                              \
      std::optional<float> depth_confidence, std::optional<float> width_confidence

#define JMT_OPTIONS                                                                                              \
  MatchOptions {                                                                                                 \
    n_layers, filter_threshold, max_distance_px, min_covisibility, early_stop_decay, depth_confidence,           \
        width_confidence                                                                                         \
  }

const char* kOptionsDoc = R"(
Options, each left at the library's default when not given:
  n_layers          transformer layers to run (FlexGlue 0..9, LightGlue 1..9; default 9)
  filter_threshold  minimum match score (default 0.1)
  max_distance_px   FlexGlue: largest distance between a keypoint and the point predicted for its partner,
                    in pixels of the network input (default 3; <= 0 off)
  min_covisibility  FlexGlue: pairs below it get no matches (default 0, off)
  early_stop_decay  FlexGlue: stop a pair early when its covisibility falls behind (default -1, off)
  depth_confidence  LightGlue: stop when this share of keypoints is confident (default 0.95; <= 0 off)
  width_confidence  LightGlue: drop confident unmatchable keypoints (default 0.99; <= 0 off))";

}  // namespace

NB_MODULE(_core, mod) {
  mod.doc() = "just-matching: FlexGlue and LightGlue image matching (compiled core).";

  mod.def(
      "backend_available", [](const std::string& name) { return jmt_backend_available(backend_of(name)) == 1; },
      "name"_a, "Whether the backend ('cpu', 'cuda', 'metal', 'webgpu') is compiled into this build.");
  mod.def(
      "compiled_backends",
      [] {
        std::vector<std::string> out;
        for (int i = 0; i < 4; ++i)
          if (jmt_backend_available(static_cast<jmt_backend>(i))) out.emplace_back(kBackendNames[i]);
        return out;
      },
      "The backends compiled into this build. Whether a device is there is checked when a Matcher is made.");

  nb::class_<Matches>(mod, "Matches",
                      "The matches of one pair. Keypoint coordinates are pixels of the original images, x right, "
                      "y down, (0, 0) the centre of the top-left pixel.")
      .def_prop_ro("idx", [](const Matches& m) { return m.idx; }, nb::rv_policy::automatic, "(n, 2) int32: index of the keypoint in a, index of its partner in b.")
      .def_prop_ro("score", [](const Matches& m) { return m.score; }, nb::rv_policy::automatic, "(n,) float32 match scores in (0, 1].")
      .def_prop_ro("a_in_b", [](const Matches& m) { return m.a_in_b; }, nb::rv_policy::automatic,
              "(n_a, 2): the point predicted in b for every keypoint of a (LightGlue: the matched keypoint, NaN "
              "when unmatched).")
      .def_prop_ro("b_in_a", [](const Matches& m) { return m.b_in_a; }, nb::rv_policy::automatic, "(n_b, 2): the same for b in a.")
      .def_prop_ro("conf_a", [](const Matches& m) { return m.conf_a; }, nb::rv_policy::automatic, "(n_a,): confidence of a_in_b (LightGlue: matchability).")
      .def_prop_ro("conf_b", [](const Matches& m) { return m.conf_b; }, nb::rv_policy::automatic, "(n_b,): the same for b.")
      .def_ro("covisibility", &Matches::covisibility,
              "FlexGlue: probability that the images show the same place. LightGlue: -1.")
      .def_ro("stop_layer", &Matches::stop_layer, "Transformer layers actually run.")
      .def("__len__", [](const Matches& m) { return m.idx.shape(0); })
      .def("__repr__", [](const Matches& m) {
        return "<Matches " + std::to_string(m.idx.shape(0)) + " of " + std::to_string(m.conf_a.shape(0)) + " and " +
               std::to_string(m.conf_b.shape(0)) + " keypoints, " + std::to_string(m.stop_layer) + " layers>";
      });

  nb::class_<Features>(mod, "Features",
                       "One image's features, made by Matcher.describe (FlexGlue) or Matcher.describe_keypoints "
                       "(LightGlue), for that Matcher only.")
      .def_ro("width", &Features::width)
      .def_ro("height", &Features::height)
      .def_prop_ro(
          "keypoints",
          [](const Features& f) -> std::optional<Out<float>> {
            if (!f.xy) return std::nullopt;
            return to_numpy(*f.xy, 2);
          },
          nb::rv_policy::automatic, "LightGlue: the (n, 2) keypoints given to describe_keypoints. FlexGlue: None.")
      .def("__repr__", [](const Features& f) {
        std::string s = "<Features " + std::to_string(f.width) + "x" + std::to_string(f.height);
        if (f.xy) s += ", " + std::to_string(f.xy->size() / 2) + " keypoints";
        return s + ">";
      });

  nb::class_<Matcher>(mod, "Matcher",
                      R"(A matching context: one model's weights on one backend.

weights: a path to a checkpoint as released (flexglue.pth, aliked_lightglue.pth, raco_aliked_lightglue.pth;
  read without PyTorch) or a .jmtw file, or its contents as bytes. The model is the one of the weights.
  No weights are in the package: see the README for where to get them and their licences.
model: "flexglue", "lightglue-aliked" or "lightglue-raco-aliked"; checked against the weights. A LightGlue model
  with no weights uses the built-in ones of a library built with JMT_EMBED_WEIGHTS (not the PyPI packages).
backend: "cpu", "cuda", "metal" or "webgpu" (compiled_backends() lists this build's).
input_size: FlexGlue's network input side in pixels, a multiple of 32 (default 512).
threads: CPU worker threads, 0 = all. device: GPU index. precision: "fp32" or "fp16" (GPU backends).)")
      .def(
          "__init__",
          [](Matcher* self, nb::handle weights, std::optional<std::string> model, const std::string& backend,
             int input_size, int threads, int device, const std::string& precision) {
            jmt_config c;
            jmt_config_default(&c);
            c.backend = backend_of(backend);
            if (precision == "fp32") c.precision = JMT_PRECISION_FP32;
            else if (precision == "fp16") c.precision = JMT_PRECISION_FP16;
            else throw nb::value_error("precision must be 'fp32' or 'fp16'");
            c.input_size = input_size, c.num_threads = threads, c.device = device;
            if (model) c.model = model->c_str();
            std::string path, blob;
            if (nb::isinstance<nb::bytes>(weights)) {
              nb::bytes b = nb::borrow<nb::bytes>(weights);
              blob.assign(b.c_str(), b.size());
              c.weights_data = blob.data(), c.weights_size = blob.size();
            } else if (!weights.is_none()) {
              path = nb::cast<std::string>(nb::module_::import_("os").attr("fspath")(weights));
              c.weights_path = path.c_str();
            }
            jmt_ctx* ctx;
            {
              nb::gil_scoped_release release;
              ctx = jmt_create(&c);
            }
            if (!ctx) fail("Matcher");
            new (self) Matcher();
            self->ctx = ctx;
            self->model = jmt::is_lightglue(ctx) ? "lightglue" : "flexglue";
            self->backend = backend;
          },
          "weights"_a = nb::none(), nb::kw_only(), "model"_a = nb::none(), "backend"_a = "cpu", "input_size"_a = 512,
          "threads"_a = 0, "device"_a = 0, "precision"_a = "fp32")
      .def_prop_ro("model", [](const Matcher& m) { return m.model; }, "'flexglue' or 'lightglue'.")
      .def_prop_ro("backend", [](const Matcher& m) { return m.backend; })
      .def(
          "describe",
          [](nb::handle self_h, nb::handle image) {
            Matcher& m = nb::cast<Matcher&>(self_h);
            In<uint8_t> a = as_array<uint8_t>(image, "image");
            int ch;
            if (a.ndim() == 2) ch = 1;
            else if (a.ndim() == 3 && (a.shape(2) == 1 || a.shape(2) == 3 || a.shape(2) == 4)) ch = static_cast<int>(a.shape(2));
            else throw nb::value_error("image must have shape (h, w), (h, w, 1), (h, w, 3) or (h, w, 4), uint8");
            jmt_image img;
            img.data = a.data();
            img.height = static_cast<int>(a.shape(0)), img.width = static_cast<int>(a.shape(1));
            img.stride = img.width * ch;
            img.format = ch == 1 ? JMT_FORMAT_GRAY8 : ch == 3 ? JMT_FORMAT_RGB8 : JMT_FORMAT_RGBA8;
            jmt_features* f;
            {
              nb::gil_scoped_release release;
              std::lock_guard<std::mutex> lock(m.mutex);
              f = jmt_describe(m.ctx, &img);
            }
            if (!f) fail("describe");
            auto* out = new Features();
            out->f = f, out->owner = nb::borrow(self_h), out->matcher = &m;
            out->width = img.width, out->height = img.height;
            return out;
          },
          "image"_a, nb::rv_policy::take_ownership,
          "FlexGlue: the features of an 8-bit image, (h, w) gray, (h, w, 3) RGB or (h, w, 4) RGBA.")
      .def(
          "describe_keypoints",
          [](nb::handle self_h, int width, int height, nb::handle xy_h, nb::handle desc_h) {
            Matcher& m = nb::cast<Matcher&>(self_h);
            std::vector<float> xy = points(xy_h, "xy");
            In<float> d = as_array<float>(desc_h, "desc");
            const std::size_t n = xy.size() / 2;
            if (d.ndim() != 2 || d.shape(0) != n)
              throw nb::value_error("desc must have shape (n, dim), one row per keypoint of xy");
            std::vector<float> desc(d.data(), d.data() + d.size());
            const int dim = static_cast<int>(d.shape(1));
            jmt_features* f;
            {
              nb::gil_scoped_release release;
              std::lock_guard<std::mutex> lock(m.mutex);
              f = jmt_describe_keypoints(m.ctx, width, height, xy.data(), desc.data(), static_cast<int>(n), dim);
            }
            if (!f) fail("describe_keypoints");
            auto* out = new Features();
            out->f = f, out->owner = nb::borrow(self_h), out->matcher = &m;
            out->width = width, out->height = height, out->xy = std::move(xy);
            return out;
          },
          "width"_a, "height"_a, "xy"_a, "desc"_a, nb::rv_policy::take_ownership,
          "LightGlue: keypoints xy (n, 2) of an image of width x height pixels and their descriptors (n, 128), "
          "as just-keypoints' aliked-n16 gives them. match() then uses these keypoints.")
      .def(
          "match",
          [](Matcher& m, nb::handle a, nb::handle b, nb::handle xy_a, nb::handle xy_b, JMT_OPTION_PARAMS) {
            std::vector<std::tuple<Features*, Features*>> fs = {{&features_of(m, a, "a"), &features_of(m, b, "b")}};
            std::vector<std::vector<float>> xys = {keypoints_of(*std::get<0>(fs[0]), xy_a, "xy_a"),
                                                   keypoints_of(*std::get<1>(fs[0]), xy_b, "xy_b")};
            return match_pairs(m, fs, xys, JMT_OPTIONS)[0];
          },
          "a"_a, "b"_a, "xy_a"_a = nb::none(), "xy_b"_a = nb::none(), JMT_OPTION_ARGS,
          (std::string("Matches keypoints xy_a (n_a, 2) of image a with xy_b (n_b, 2) of image b. LightGlue: the "
                       "keypoints of describe_keypoints, so xy_a and xy_b can be left out.\n") +
           kOptionsDoc)
              .c_str())
      .def(
          "match_batch",
          [](Matcher& m, nb::handle pairs, JMT_OPTION_PARAMS) {
            std::vector<std::tuple<Features*, Features*>> fs;
            std::vector<std::vector<float>> xys;
            for (nb::handle p : pairs) {
              nb::sequence t = nb::borrow<nb::sequence>(p);
              const std::size_t len = nb::len(t);
              if (len != 2 && len != 4) throw nb::value_error("each pair is (a, b) or (a, b, xy_a, xy_b)");
              Features& fa = features_of(m, t[0], "a");
              Features& fb = features_of(m, t[1], "b");
              fs.emplace_back(&fa, &fb);
              xys.push_back(keypoints_of(fa, len == 4 ? nb::handle(t[2]) : nb::none(), "xy_a"));
              xys.push_back(keypoints_of(fb, len == 4 ? nb::handle(t[3]) : nb::none(), "xy_b"));
            }
            return match_pairs(m, fs, xys, JMT_OPTIONS);
          },
          "pairs"_a, JMT_OPTION_ARGS,
          (std::string("Matches many pairs, each (a, b) or (a, b, xy_a, xy_b), as match() would; on a GPU backend "
                       "the pairs are pipelined.\n") +
           kOptionsDoc)
              .c_str())
      .def(
          "covisibility",
          [](Matcher& m, nb::handle a, nb::handle b, int level) {
            Features& fa = features_of(m, a, "a");
            Features& fb = features_of(m, b, "b");
            float score = 0;
            int rc;
            {
              nb::gil_scoped_release release;
              std::lock_guard<std::mutex> lock(m.mutex);
              rc = jmt_covisibility(m.ctx, fa.f, fb.f, level, &score);
            }
            if (rc != 0) fail("covisibility");
            return score;
          },
          "a"_a, "b"_a, "level"_a = 1,
          "FlexGlue: how likely two images overlap, from their features alone. Level 1: cosine similarity of "
          "the mean feature vectors (cheap, in practice 0.8 to 1). Level 2: grid points through the first layer "
          "and back, the share that return, in [0, 1].")
      .def(
          "shortlist",
          [](Matcher& m, nb::handle images, int k, int neighbours, bool rerank, float min_score) {
            std::vector<const jmt_features*> fs;
            for (nb::handle h : images) fs.push_back(features_of(m, h, "images").f);
            jmt_shortlist_options o;
            jmt_shortlist_options_default(&o);
            o.k = k, o.neighbours = neighbours, o.rerank = rerank ? 1 : 0, o.min_score = min_score;
            std::vector<int32_t> idx;
            std::vector<float> score;
            {
              nb::gil_scoped_release release;
              std::lock_guard<std::mutex> lock(m.mutex);
              jmt_pair_list out;
              if (jmt_shortlist(m.ctx, fs.data(), static_cast<int>(fs.size()), &o, &out) != 0) fail("shortlist");
              idx.assign(out.idx, out.idx + 2 * static_cast<std::size_t>(out.n));
              score.assign(out.score, out.score + out.n);
            }
            return std::make_tuple(to_numpy(idx, 2), to_numpy(score, 1));
          },
          "images"_a, nb::kw_only(), "k"_a = 10, "neighbours"_a = 0, "rerank"_a = false, "min_score"_a = 0.0f,
          "FlexGlue: the pairs of images worth matching, best first, as (idx (n, 2) int32, score (n,)). k partners "
          "per image by level 1; neighbours adds (i, i+1) .. (i, i+neighbours) for images in capture order; "
          "rerank scores the kept pairs with level 2 and drops those below min_score.");
}
