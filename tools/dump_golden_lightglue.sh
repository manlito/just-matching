#!/bin/bash
# Golden files of the original LightGlue on the 6 test pairs of compare_all.sh, for both ALIKED models and
# LighterGlue on XFeat.
#   tools/dump_golden_lightglue.sh   ->  build/golden/lightglue-{aliked,raco-aliked,xfeat}/<A>-<B>.jmtt
# Needs tools/fetch_lightglue.sh, the synthetic pairs of tools/presets.py in build/synth, and the jkp CLI (JKP).
set -euo pipefail
PY=${PYTHON:-python}
A=third_party/FlexGlue/assets; S=build/synth
PAIRS=($A/berlin1.jpg $A/berlin2.jpg $A/lamar_cab_1446.jpg $A/lamar_cab_1450.jpg $A/berlin1.jpg $A/lamar_cab_1446.jpg
       $S/berlin1_h1.png $S/berlin1_h2.png $S/lamar_cab_1446_h1.png $S/lamar_cab_1446_h2.png $S/berlin2_h1.png $S/lamar_cab_1450_h2.png)
for m in aliked raco-aliked xfeat; do
  for ((i = 0; i < ${#PAIRS[@]}; i += 2)); do
    $PY tools/dump_golden_lightglue.py "${PAIRS[i]}" "${PAIRS[i+1]}" --model $m "$@"
  done
done
