#!/usr/bin/env bash
# Fetch the original LightGlue at its pinned commit into third_party/LightGlue, and its two checkpoints for
# 128-dim ALIKED descriptors into third_party/lightglue-weights/. Code and weights are Apache-2.0
# (third_party/LightGlue/LICENSE); nothing in them comes from DINO.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

dest="$ROOT/third_party/LightGlue"
commit=eb42fee2d71449efb0aa5c10549752b5d75384d8
if [ ! -d "$dest/.git" ]; then
  git clone -q https://github.com/cvg/LightGlue.git "$dest"
fi
git -C "$dest" fetch -q origin 2>/dev/null || true
git -C "$dest" checkout -q "$commit"
echo "LightGlue at $(git -C "$dest" rev-parse HEAD) in $dest"

mkdir -p "$ROOT/third_party/lightglue-weights"
release=https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv
while read -r name sha; do
  pth="$ROOT/third_party/lightglue-weights/$name"
  if [ ! -f "$pth" ]; then
    curl -fsSL -o "$pth.part" "$release/$name"
    mv "$pth.part" "$pth"
  fi
  if [ "$sha" = PENDING ]; then
    shasum -a 256 "$pth"
  else
    echo "$sha  $pth" | shasum -a 256 -c -
  fi
done <<'EOF'
aliked_lightglue.pth d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503
raco_aliked_lightglue.pth a804c3fe2851cc3c765cda6a9f412845893c8d8e2a3048f848d7f6ba172e6312
EOF
