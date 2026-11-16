"""Reader/writer for the just-keypoints tensor container (.jkpw weights, .jkpt test tensors).

Layout (all integers little-endian):

    char[4]  magic        "JKPW"
    u32      version      1
    u32      n_meta
    u32      n_tensors
    n_meta    x { u16 key_len, key bytes, u16 value_len, value bytes }      (UTF-8 strings)
    n_tensors x { u16 name_len, name bytes, u8 dtype, u8 ndim, u32 dims[ndim], u64 offset, u64 nbytes }
    padding, then raw tensor data; each tensor starts at `offset` (from file start, 64-byte aligned)

dtype: 0 = float32, 1 = float16, 2 = int32, 3 = uint8. Tensor data is row-major (C order).

Metadata keys used by weights files: kind, model, c1, c2, c3, c4, dim, K, M, source, source_commit, license.
"""
import struct

import numpy as np

MAGIC = b"JKPW"
VERSION = 1
ALIGN = 64
DTYPES = {0: np.dtype("<f4"), 1: np.dtype("<f2"), 2: np.dtype("<i4"), 3: np.dtype("u1")}
DTYPE_IDS = {v: k for k, v in DTYPES.items()}


def _align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def write(path, meta, tensors):
    """meta: dict[str, str-able]; tensors: dict[str, np.ndarray] (insertion order is kept)."""
    arrays = {}
    for name, a in tensors.items():
        a = np.asarray(a)
        if a.dtype == np.int64:
            a = a.astype(np.int32)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<")))
        if a.dtype not in DTYPE_IDS:
            raise TypeError(f"{name}: unsupported dtype {a.dtype}")
        arrays[name] = a

    def s(x):
        b = str(x).encode("utf-8")
        return struct.pack("<H", len(b)) + b

    head = MAGIC + struct.pack("<III", VERSION, len(meta), len(arrays))
    for k, v in meta.items():
        head += s(k) + s(v)
    table_size = sum(2 + len(n.encode()) + 2 + 4 * a.ndim + 16 for n, a in arrays.items())
    offset = _align(len(head) + table_size)
    table, layout = b"", []
    for name, a in arrays.items():
        table += s(name) + struct.pack("<BB", DTYPE_IDS[a.dtype], a.ndim)
        table += struct.pack(f"<{a.ndim}I", *a.shape) + struct.pack("<QQ", offset, a.nbytes)
        layout.append((offset, a))
        offset = _align(offset + a.nbytes)
    with open(path, "wb") as f:
        f.write(head + table)
        for off, a in layout:
            f.write(b"\0" * (off - f.tell()))
            f.write(a.tobytes())


def read(path):
    """Returns (meta: dict[str, str], tensors: dict[str, np.ndarray])."""
    buf = open(path, "rb").read()
    if buf[:4] != MAGIC:
        raise ValueError(f"{path}: bad magic")
    version, n_meta, n_tensors = struct.unpack_from("<III", buf, 4)
    if version != VERSION:
        raise ValueError(f"{path}: unsupported version {version}")
    pos = 16

    def s():
        nonlocal pos
        (n,) = struct.unpack_from("<H", buf, pos)
        pos += 2 + n
        return buf[pos - n:pos].decode("utf-8")

    meta = {}
    for _ in range(n_meta):
        k = s()
        meta[k] = s()
    tensors = {}
    for _ in range(n_tensors):
        name = s()
        dtype, ndim = struct.unpack_from("<BB", buf, pos)
        pos += 2
        dims = struct.unpack_from(f"<{ndim}I", buf, pos)
        pos += 4 * ndim
        offset, nbytes = struct.unpack_from("<QQ", buf, pos)
        pos += 16
        tensors[name] = np.frombuffer(buf, DTYPES[dtype], count=nbytes // DTYPES[dtype].itemsize,
                                      offset=offset).reshape(dims)
    return meta, tensors
