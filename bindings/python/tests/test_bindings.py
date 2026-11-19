"""Tests of the Python bindings. Without weights they check the API and its errors; with weights they match.

Weights are looked for in the environment, then in the repository (as tools/fetch_lightglue.sh and the README
put them); the wheel builds have none, so there only the first part runs:

    JMT_LIGHTGLUE_WEIGHTS   aliked_lightglue.pth or lightglue-aliked.jmtw
    JMT_FLEXGLUE_WEIGHTS    flexglue.pth or flexglue.jmtw

GPU backends run when compiled in and a device is there; otherwise those tests skip.
"""

import os
import struct
import threading
from pathlib import Path

import numpy as np
import pytest

import just_matching as jmt

REPO = Path(__file__).resolve().parents[3]
GPU = [b for b in ("cuda", "metal") if jmt.backend_available(b)]


def find(env, *candidates):
    if os.environ.get(env):
        return os.environ[env]
    for c in candidates:
        if (REPO / c).is_file():
            return str(REPO / c)
    return None


LIGHTGLUE = find("JMT_LIGHTGLUE_WEIGHTS", "third_party/lightglue-weights/aliked_lightglue.pth",
                 "weights/lightglue-aliked.jmtw")
FLEXGLUE = find("JMT_FLEXGLUE_WEIGHTS", "third_party/flexglue.pth", "weights/flexglue.jmtw")
needs_lightglue = pytest.mark.skipif(not LIGHTGLUE, reason="no LightGlue weights (JMT_LIGHTGLUE_WEIGHTS)")
needs_flexglue = pytest.mark.skipif(not FLEXGLUE, reason="no FlexGlue weights (JMT_FLEXGLUE_WEIGHTS)")


def matcher_or_skip(weights, backend, **kw):
    try:
        return jmt.Matcher(weights, backend=backend, **kw)
    except RuntimeError as e:
        if backend != "cpu":
            pytest.skip(f"{backend}: {e}")
        raise


# ---- Without weights.

def test_build_info():
    assert jmt.VARIANT in ("cpu", "cu12", "cu13", "source")
    assert jmt.__version__.count(".") == 2
    assert "cpu" in jmt.compiled_backends()
    assert jmt.backend_available("cpu")
    assert bool(jmt.CUDA_VERSION) == jmt.backend_available("cuda")
    if jmt.VARIANT.startswith("cu"):
        assert jmt.backend_available("cuda") and jmt.CUDA_VERSION.startswith(jmt.VARIANT[2:])


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        jmt.backend_available("tpu")
    with pytest.raises(ValueError, match="unknown backend"):
        jmt.Matcher("x.pth", backend="tpu")


def test_no_weights():
    with pytest.raises(RuntimeError, match="no weights given"):
        jmt.Matcher()
    with pytest.raises(RuntimeError, match="not built into this library"):
        jmt.Matcher(model="lightglue-aliked")


def test_bad_weights(tmp_path):
    with pytest.raises(RuntimeError):
        jmt.Matcher(tmp_path / "missing.pth")
    with pytest.raises(RuntimeError, match="not a weights container"):
        jmt.Matcher(b"not weights at all, just bytes")
    with pytest.raises(RuntimeError, match="unknown model"):
        jmt.Matcher(b"x", model="superglue")
    with pytest.raises(ValueError, match="precision"):
        jmt.Matcher(b"x", precision="int8")


# ---- LightGlue.

def random_keypoints(rng, n, w=640, h=480, dim=128):
    xy = rng.uniform([0, 0], [w - 1, h - 1], size=(n, 2)).astype(np.float32)
    desc = rng.normal(size=(n, dim)).astype(np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    return xy, desc


@needs_lightglue
def test_lightglue_api():
    m = jmt.Matcher(LIGHTGLUE)
    assert m.model == "lightglue" and m.backend == "cpu"
    rng = np.random.default_rng(0)
    xy, desc = random_keypoints(rng, 300)
    a = m.describe_keypoints(640, 480, xy, desc)
    assert (a.width, a.height) == (640, 480)
    np.testing.assert_array_equal(a.keypoints, xy)
    with pytest.raises(ValueError, match="shape"):
        m.describe_keypoints(640, 480, xy[:, :1], desc)
    with pytest.raises(ValueError, match="one row per keypoint"):
        m.describe_keypoints(640, 480, xy, desc[:10])
    with pytest.raises(RuntimeError, match="128 values"):
        m.describe_keypoints(640, 480, xy, desc[:, :64])
    with pytest.raises(RuntimeError, match="use jmt_describe_keypoints"):
        m.describe(np.zeros((64, 64), np.uint8))
    with pytest.raises(RuntimeError, match="needs FlexGlue"):
        m.covisibility(a, a)
    other = jmt.Matcher(LIGHTGLUE)
    with pytest.raises(ValueError, match="another Matcher"):
        m.match(a, other.describe_keypoints(640, 480, xy, desc))
    with pytest.raises(TypeError, match="expected Features"):
        m.match(a, "b")
    empty = m.describe_keypoints(640, 480, np.zeros((0, 2)), np.zeros((0, 128)))
    r = m.match(a, empty)
    assert len(r) == 0 and r.conf_a.shape == (300,) and r.conf_b.shape == (0,)


@needs_lightglue
@pytest.mark.parametrize("backend", ["cpu"] + GPU)
def test_lightglue_finds_a_permutation(backend):
    # Image b is image a with its keypoints shuffled, moved by a fraction of a pixel: every match must be the
    # keypoint's own copy.
    m = matcher_or_skip(LIGHTGLUE, backend)
    rng = np.random.default_rng(1)
    xy, desc = random_keypoints(rng, 500)
    perm = rng.permutation(500)
    xy_b = xy[perm] + rng.normal(0, 0.2, size=(500, 2)).astype(np.float32)
    a = m.describe_keypoints(640, 480, xy, desc)
    b = m.describe_keypoints(640, 480, xy_b, desc[perm])
    r = m.match(a, b)
    assert r.idx.dtype == np.int32 and r.idx.shape == (len(r), 2) and r.score.shape == (len(r),)
    assert len(r) > 400
    inverse = np.argsort(perm)
    assert np.mean(inverse[r.idx[:, 0]] == r.idx[:, 1]) > 0.99
    assert r.covisibility == -1 and 1 <= r.stop_layer <= 9
    np.testing.assert_allclose(r.a_in_b[r.idx[:, 0]], xy_b[r.idx[:, 1]])
    # The same through match_batch, with the keypoints given again, and with options.
    batch = m.match_batch([(a, b), (b, a, xy_b, xy)])
    np.testing.assert_array_equal(batch[0].idx, r.idx)
    assert len(batch[1]) == len(r)
    few = m.match(a, b, n_layers=2, depth_confidence=-1)
    assert few.stop_layer == 2


def read_jkpt(path):
    """keypoints, descriptors and image size of a just-keypoints .jkpt file (tools/jmtfile.py, cut down)."""
    buf = open(path, "rb").read()
    pos, meta, tensors = 16, {}, {}
    n_meta, n_tensors = struct.unpack_from("<II", buf, 8)

    def s():
        nonlocal pos
        (n,) = struct.unpack_from("<H", buf, pos)
        pos += 2 + n
        return buf[pos - n:pos].decode()

    for _ in range(n_meta):
        k = s()
        meta[k] = s()
    for _ in range(n_tensors):
        name = s()
        dtype, ndim = struct.unpack_from("<BB", buf, pos)
        dims = struct.unpack_from(f"<{ndim}I", buf, pos + 2)
        offset, nbytes = struct.unpack_from("<QQ", buf, pos + 2 + 4 * ndim)
        pos += 2 + 4 * ndim + 16
        if dtype == 0:
            tensors[name] = np.frombuffer(buf, "<f4", count=nbytes // 4, offset=offset).reshape(dims)
    return int(meta["width"]), int(meta["height"]), tensors["keypoints"], tensors["descriptors"]


BERLIN = [REPO / f"build/keypoints/berlin{i}.jpg.aliked.2048.jkpt" for i in (1, 2)]


@needs_lightglue
@pytest.mark.skipif(not all(p.is_file() for p in BERLIN), reason="no Berlin keypoints (tools/dump_golden_lightglue.py)")
def test_lightglue_berlin():
    # The C API's end-to-end check (CMakeLists.txt, lightglue_cli_builtin): the original's 219 matches after 7
    # layers with upstream's adaptive defaults.
    m = jmt.Matcher(LIGHTGLUE)
    a, b = (m.describe_keypoints(*read_jkpt(p)) for p in BERLIN)
    r = m.match(a, b)
    assert (len(r), r.stop_layer) == (219, 7)


@needs_lightglue
def test_threads():
    # Two matchers in two threads, and one matcher shared by two threads (its mutex), give the same results.
    rng = np.random.default_rng(2)
    xy, desc = random_keypoints(rng, 400)
    m = jmt.Matcher(LIGHTGLUE, threads=2)
    a = m.describe_keypoints(640, 480, xy, desc)
    b = m.describe_keypoints(640, 480, xy[::-1].copy(), desc[::-1].copy())
    want = m.match(a, b).idx
    got = []
    ts = [threading.Thread(target=lambda: got.append(m.match(a, b).idx)) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(got) == 4 and all(np.array_equal(g, want) for g in got)


# ---- FlexGlue.

def scene(rng, w=320, h=240):
    """Two views of one textured plane: b is a shifted by a few pixels."""
    big = (rng.uniform(0, 255, size=(h // 8 + 4, w // 8 + 4, 3))).astype(np.uint8)
    big = np.kron(big, np.ones((8, 8, 1), np.uint8))
    return np.ascontiguousarray(big[:h, :w]), np.ascontiguousarray(big[12:12 + h, 16:16 + w])


@needs_flexglue
@pytest.mark.parametrize("backend", ["cpu"] + GPU)
def test_flexglue(backend):
    m = matcher_or_skip(FLEXGLUE, backend, input_size=256)
    assert m.model == "flexglue"
    rng = np.random.default_rng(3)
    img_a, img_b = scene(rng)
    a, b = m.describe(img_a), m.describe(img_b[..., 0])  # RGB and gray
    assert a.keypoints is None and (a.width, a.height) == (320, 240)
    with pytest.raises(ValueError, match="FlexGlue matches the keypoints"):
        m.match(a, b)
    with pytest.raises(ValueError, match="image must have shape"):
        m.describe(np.zeros((4, 4, 2), np.uint8))
    xy = rng.uniform([20, 20], [300, 220], size=(200, 2)).astype(np.float32)
    r = m.match(a, b, xy, xy - [16, 12])
    assert r.a_in_b.shape == (200, 2) and r.conf_a.shape == (200,) and 0 <= r.covisibility <= 1
    assert 0 < m.covisibility(a, b, level=1) <= 1.0001
    assert 0 <= m.covisibility(a, b, level=2) <= 1
    with pytest.raises(RuntimeError, match="level must be 1 or 2"):
        m.covisibility(a, b, level=3)
    c = m.describe(rng.uniform(0, 255, size=(240, 320, 4)).astype(np.uint8))
    idx, score = m.shortlist([a, b, c], k=1)
    assert idx.dtype == np.int32 and idx.shape[1] == 2 and score.shape == (len(idx),)
    assert [0, 1] in idx.tolist()
    with pytest.raises(RuntimeError, match="use jmt_describe"):
        m.describe_keypoints(320, 240, xy, np.zeros((200, 128), np.float32))
