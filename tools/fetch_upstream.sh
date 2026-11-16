#!/usr/bin/env bash
# Fetch the original FlexGlue at its pinned commit into third_party/FlexGlue, and the released checkpoint
# (backbone + matcher, 567 MB) as third_party/flexglue.pth. The backbone weights in that file are under the
# DINOv3 License (third_party/FlexGlue/DINOv3_LICENSE.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

fetch() {  # name url commit
  local dest="$ROOT/third_party/$1"
  if [ ! -d "$dest/.git" ]; then
    mkdir -p "$(dirname "$dest")"
    git clone "$2" "$dest"
  fi
  git -C "$dest" fetch -q origin "$3" 2>/dev/null || git -C "$dest" fetch -q origin
  git -C "$dest" checkout -q "$3"
  echo "$1 at $(git -C "$dest" rev-parse HEAD) in $dest"
}

fetch FlexGlue https://github.com/cvg/FlexGlue.git 99920296d082c9405ebfd612d674efe288e74881

PTH="$ROOT/third_party/flexglue.pth"
SHA256=d7371f743b8cdc4adf723c2aa05b181587fe4bc177fced139697975dff790068
if [ ! -f "$PTH" ]; then
  curl -fsSL -o "$PTH.part" https://github.com/cvg/FlexGlue/releases/download/v0.1/flexglue.pth
  mv "$PTH.part" "$PTH"
fi
if [ "$SHA256" = PENDING ]; then
  shasum -a 256 "$PTH"
else
  echo "$SHA256  $PTH" | shasum -a 256 -c -
fi
