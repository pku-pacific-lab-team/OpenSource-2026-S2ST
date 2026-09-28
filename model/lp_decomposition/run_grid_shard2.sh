#!/usr/bin/env bash
# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

# Grid over LP block size x prediction order on test1k utterances 668-999 (abs 0.02, open loop).
# One background process per block size; each runs its orders sequentially (whisper loaded once).
# Usage: bash <this repo>/model/lp_decomposition/run_grid_shard2.sh
SCR="${SCRATCH_DIR:-res}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCR/lpfeat/grid_shard2"
run_block() {  # $1=block  $2...=orders
  local b=$1; shift
  for o in "$@"; do
    PYTHONIOENCODING=utf-8 python "$HERE/replay_eval.py" \
      --rec "$SCR/rec-fp-s2st" --start 668 --n 332 \
      --out "$OUT/b${b}_o${o}" --thresh 0.02 --mode abs --block "$b" --order "$o"
    echo "[$(date +%H:%M:%S)] done b=$b o=$o"
  done
}
mkdir -p "$OUT"
run_block 32  4 8 16  > "$SCR/lpfeat_grid_b32.log"  2>&1 &
run_block 64  8 16    > "$SCR/lpfeat_grid_b64.log"  2>&1 &
run_block 128 4 8 16  > "$SCR/lpfeat_grid_b128.log" 2>&1 &
wait
echo "[$(date +%H:%M:%S)] grid done"
