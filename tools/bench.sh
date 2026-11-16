#!/bin/bash
# usage: tools/bench.sh [backend] [extra jmt options]  -> wall time (best of 4) and GPU ms per kernel
# backend: metal, metal:fp16, cuda, cuda:fp16
A=third_party/FlexGlue/assets; K=build/keypoints
B=${1:-metal}; shift
ARGS="$A/berlin1.jpg $A/berlin2.jpg --kpts-a $K/berlin1.jpg.raco.2048.jkpt --kpts-b $K/berlin2.jpg.raco.2048.jkpt --backend $B $@"
build/release/jmt match $ARGS --runs 4 | tr '\n' ' '; echo
JMT_METAL_PROFILE=1 JMT_CUDA_PROFILE=1 build/release/jmt match $ARGS 2>&1 | grep -A7 "^kernel\|^kind" | tail -7 | awk '{printf "  %-22s %8s ms\n", $1, $2}'
