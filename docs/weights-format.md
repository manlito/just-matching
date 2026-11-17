# Weights format

The library reads FlexGlue's released checkpoint, `flexglue.pth`, directly: a zip archive of a pickled state dict,
which it reads without PyTorch and rearranges as below. It also reads its own container, `.jmtw`: one flat binary
file with a header, a table of tensors and the raw data. It needs no parsing
library and loads equally with `fread`, `mmap` or `fetch()` into an `ArrayBuffer`. It is the container of
just-keypoints' `.jkpw` files, with FlexGlue's tensors.

The same container, with other tensors, holds the command-line tool's output (`.jmtt`): `matches`, `scores`,
`a_in_b`, `b_in_a`, `conf_a` and `conf_b` from `jmt match`, and `pairs` and `scores` from `jmt shortlist`.

## Layout

All integers are little-endian.

```text
char[4]  magic        "JKPW"
u32      version      1
u32      n_meta
u32      n_tensors
n_meta    x { u16 key_len, key bytes, u16 value_len, value bytes }      (UTF-8 strings)
n_tensors x { u16 name_len, name bytes, u8 dtype, u8 ndim, u32 dims[ndim], u64 offset, u64 nbytes }
padding, then the raw tensor data; each tensor starts at `offset` (from the start of the file, 64-byte aligned)
```

`dtype`: 0 = float32, 1 = float16, 2 = int32, 3 = uint8. Tensor data is row-major (C order).

Metadata keys of the weights files: `kind`, `model`, `n_layers`, `dim`, `heads`, `feature_dim`, `search_dim`,
`stride`, `depths`, `widths`, `source`, `source_commit`, `license`.

## The weights

| File | Tensors | Size |
|---|---|---|
| `flexglue.pth` | FlexGlue's v0.1 checkpoint, as released | 567 MB |
| `flexglue.jmtw` | 870 tensors, 141,741,444 values, float32 | 567 MB |
| `flexglue-fp16.jmtw` | the same tensors as float16 | 284 MB |

The repository's `tools/export_weights.py` makes the `.jmtw` files from the checkpoint; reading the checkpoint
gives the float32 file's tensors bit for bit. 89.6 million of the values are the DINOv3 ConvNeXt-B backbone and
its neck, under the DINOv3 License; the 52.2 million of the matcher are under the MIT licence
([Licences](licences.md)).

Changes made to the checkpoint, when it is read or exported, none of which alters the function:

- ConvNeXt's layer scale is folded into the second pointwise convolution of each block.
- The rows of each attention layer's combined query-key-value projection are reordered so that queries, keys and
  values are contiguous.
- 1x1 convolution weights are stored as matrices.
