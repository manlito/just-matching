"""Turns pyproject.toml into one of the CUDA packages, in place, as just-ba's script:

    python packaging/variant.py cu12 --root /opt/jmt-cuda --platform linux
    python packaging/variant.py cu13 --root C:/jmt-cuda --platform windows

Run by the release workflow only, before cibuildwheel; never commit the result. ``--root`` is where
packaging/fetch_cuda.py put the CUDA toolkit (``cuda/``), as the build sees it (inside the manylinux container
on Linux). ``--root system`` uses the toolkit CMake finds itself (a machine with CUDA installed).

The CUDA runtime is linked statically and the backend uses no other CUDA library, so the CUDA packages depend
on nothing from NVIDIA beyond the driver.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# GPU architectures per package: real code for each, plus PTX for the newest so later GPUs can still run it.
# The kernels use ldmatrix and mma.sync m16n8k8 (PLAN.md), which need sm_75 (Turing): both packages start
# there, and differ in the toolkit, so in the drivers they need (CUDA 12: R525 or later; CUDA 13: R580).
ARCHITECTURES = {
    "cu12": "75-real;80-real;86-real;89-real;90-real;100-real;120",
    "cu13": "75-real;80-real;86-real;89-real;90-real;100-real;120",
}


def sub_once(text: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"variant.py: pattern not found in pyproject.toml: {pattern}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(ARCHITECTURES))
    ap.add_argument("--root", required=True)
    ap.add_argument("--platform", choices=["linux", "windows"], required=True)
    ap.add_argument("--architectures", help="override the list, e.g. 75 for a local check")
    args = ap.parse_args()
    v, root = args.variant, args.root.rstrip("/\\")

    text = PYPROJECT.read_text()
    text = sub_once(text, r'^name = "just-matching"$', f'name = "just-matching-{v}"')
    text = sub_once(text, r'^(description = ".*)"$', rf'\1 (CUDA {v[2:]} package)"')
    defines = [
        f'JMT_PACKAGE_VARIANT = "{v}"',
        'JMT_METAL = "OFF"',
        f'CMAKE_CUDA_ARCHITECTURES = "{args.architectures or ARCHITECTURES[v]}"',
    ]
    if root != "system":
        nvcc = f"{root}/cuda/bin/nvcc" + (".exe" if args.platform == "windows" else "")
        defines += [f'CMAKE_CUDA_COMPILER = "{nvcc}"', f'CUDAToolkit_ROOT = "{root}/cuda"']
    text = sub_once(text, r'^JMT_CUDA = "OFF"$', 'JMT_CUDA = "ON"')
    text = sub_once(text, r'^JMT_PACKAGE_VARIANT = "cpu"$', "\n".join(defines))
    PYPROJECT.write_text(text)
    print(f"pyproject.toml is now just-matching-{v} ({args.platform}, CUDA under {root})")


if __name__ == "__main__":
    main()
