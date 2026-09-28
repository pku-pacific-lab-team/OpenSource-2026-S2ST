#!/usr/bin/env bash
# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

# Stepwise thresholded beam search (product of new-token probs, stop at first drop).
# Usage: bash <this repo>/model/speculative_generation/run_stepwise.sh <subset> [tag] [thresholds...]
set -u
ROOT="${STREAMSPEECH_ROOT:-$(pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB=${1:-test100}; TAG=${2:-$SUB}; shift 2 || true
THRS=${@:-"0.1 0.3 0.5"}
DATA=datasets/cvss-eval/$SUB; OUT=res/conf-trigger/$TAG; CHUNK=320
run_conf() {
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] START $name"
  PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT/fairseq" simuleval \
    --data-bin "$ROOT/configs/fr-en" \
    --user-dir "$ROOT/researches/ctc_unity" --agent-dir "$ROOT/agent" \
    --source $DATA/wav_list.txt --target $DATA/target.txt \
    --model-path "$ROOT/pretrain_models/streamspeech.simultaneous.fr-en.pt" \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --source-segment-size $CHUNK \
    --quality-metrics BLEU --latency-metrics AL LAAL RTF --device gpu \
    --agent "$HERE/speech_to_text.s2tt.confidence.agent.py" \
    --output "$OUT/$name" --trace-file "$OUT/$name/trace.jsonl" "$@" 2>&1 | tail -2
  echo "[$(date +%H:%M:%S)] DONE $name"
}
for thr in $THRS; do
  run_conf stepwise-prod-$thr --beam-size 5 --conf-threshold $thr --conf-type prod --commit-mode stepwise
done
echo "[$(date +%H:%M:%S)] ALL DONE"
