#!/bin/bash
# Runs a command on another machine, for testing a backend this machine does not have:
#   tools/remote.sh USER@HOST REMOTE_DIR COMMAND...
# COMMAND runs in REMOTE_DIR. Arguments that are absolute paths of local files are copied over first, and
# the file named after --out is copied back. With tools/compare.py:
#   --b "cmd:tools/remote.sh user@host /tmp/just-matching build/release/jmt match --backend cuda"
set -e
host=$1; dir=$2; shift 2
tmp=$(ssh "$host" mktemp -d /tmp/jmt-remote.XXXXXX)
args=(); files=(); out=; prev=
for a in "$@"; do
  if [ "$prev" = "--out" ]; then out=$a; args+=("$tmp/out.jmtt")
  elif [[ "$a" = /* ]] && [ -f "$a" ]; then files+=("$a"); args+=("$tmp/$(basename "$a")")
  else args+=("$a"); fi
  prev=$a
done
[ ${#files[@]} -eq 0 ] || scp -q "${files[@]}" "$host:$tmp/"
ssh "$host" "cd $dir && ${args[*]}"
[ -z "$out" ] || scp -q "$host:$tmp/out.jmtt" "$out"
ssh "$host" "rm -r $tmp"
