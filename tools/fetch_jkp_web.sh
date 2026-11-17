#!/usr/bin/env bash
# Fetch just-keypoints' browser module (the demo's keypoints: RaCo) from just-keypoints' documentation site into
# third_party/jkp-web (not committed), checked against pinned hashes:
#
#   tools/fetch_jkp_web.sh
#
# The files are what just-keypoints' main branch builds with Emscripten (same bytes). When just-keypoints deploys
# a new module the check fails: run the script with JKP_WEB_UPDATE=1 to print the new hashes, check the demo with
# them, then paste them below. CMakeLists.txt takes the module from here by default (JMT_JKP_WEB_DIR).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/jkp-web"
BASE="https://just-keypoints.pages.dev/lib"
FILES=(
  "jkp.js d459bd7cbb03187b0662a289e8235ce4b70d83969b93b3a750f71e89972d0262"
  "jkp_wasm.js 54d4ddebaa5187427297c2c1ef401ca8ba4016367ad763f96a4bc58b7e36ae9a"
  "jkp_wasm.wasm 3fc11898ad410815e8a399c7e110bc9cb218362088c91bba6f84ec8a12711d20"
  "raco.jkpw a206d87e639b697d478d7bc7ec86d1e7f5a00f51d972721e69dbc784efa82abc"
)

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

mkdir -p "$DEST"
fail=0
for entry in "${FILES[@]}"; do
  read -r name want <<<"$entry"
  if [ -f "$DEST/$name" ] && [ "$(sha256 "$DEST/$name")" = "$want" ]; then continue; fi
  curl -fsSL -o "$DEST/$name.part" "$BASE/$name"
  got="$(sha256 "$DEST/$name.part")"
  if [ "$got" != "$want" ]; then
    if [ "${JKP_WEB_UPDATE:-0}" = 1 ]; then
      echo "  \"$name $got\""
      mv "$DEST/$name.part" "$DEST/$name"
    else
      echo "$name: sha256 $got, expected $want (just-keypoints deployed a new module? see the top of $0)" >&2
      rm -f "$DEST/$name.part"
      fail=1
    fi
    continue
  fi
  mv "$DEST/$name.part" "$DEST/$name"
done
[ "$fail" = 0 ] || exit 1
echo "just-keypoints' module: $DEST"
