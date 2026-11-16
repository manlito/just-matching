#!/bin/bash
# Compares a backend of the jmt CLI with the original on the 6 test pairs at two settings:
#   tools/compare_all.sh metal:fp16 [extra compare.py options, e.g. --save build/repro]
# JMT is the command that runs the CLI (default build/release/jmt); for a backend on another machine:
#   JMT="tools/remote.sh user@host /path/to/just-matching build/release/jmt" tools/compare_all.sh cuda
# Needs third_party/FlexGlue (tools/fetch_upstream.sh) and the synthetic pairs of tools/presets.py in build/synth.
B=${1:-metal}; shift
EXTRA=("$@")
JMT=${JMT:-build/release/jmt}
PY=${PYTHON:-python}
A=third_party/FlexGlue/assets; S=build/synth
PAIRS="$A/berlin1.jpg $A/berlin2.jpg $A/lamar_cab_1446.jpg $A/lamar_cab_1450.jpg $A/berlin1.jpg $A/lamar_cab_1446.jpg
       $S/berlin1_h1.png $S/berlin1_h2.png $S/lamar_cab_1446_h1.png $S/lamar_cab_1446_h2.png $S/berlin2_h1.png $S/lamar_cab_1450_h2.png"
for setting in "512 9" "384 5"; do
  set -- $setting
  echo "=== $B, $1 px, $2 layers"
  $PY tools/compare.py $PAIRS --b "cmd:$JMT match --backend $B" --size $1 --layers $2 "${EXTRA[@]}"
done
