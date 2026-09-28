#!/usr/bin/env bash
# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

# Sweep of the confidence-triggered S2TT policy (ASR-new-word -> MT beam search ->
# commit if top-1 confidence >= thr).  TTS is never run; quality is text BLEU.
# Usage: bash <this repo>/model/speculative_generation/run_sweep.sh <subset: test100|test1k> [tag]
set -u
ROOT="${STREAMSPEECH_ROOT:-$(pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB=${1:-test100}
TAG=${2:-$SUB}
DATA=datasets/cvss-eval/$SUB
OUT=res/conf-trigger/$TAG
CHUNK=320
mkdir -p "$OUT"

common() {
  PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT/fairseq" simuleval \
    --data-bin "$ROOT/configs/fr-en" \
    --user-dir "$ROOT/researches/ctc_unity" --agent-dir "$ROOT/agent" \
    --source $DATA/wav_list.txt --target $DATA/target.txt \
    --model-path "$ROOT/pretrain_models/streamspeech.simultaneous.fr-en.pt" \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --source-segment-size $CHUNK \
    --quality-metrics BLEU --latency-metrics AL LAAL RTF --device gpu "$@"
}

run_conf() {  # $1=name, rest = agent args
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] START $name"
  common --agent "$HERE/speech_to_text.s2tt.confidence.agent.py" \
    --output "$OUT/$name" --trace-file "$OUT/$name/trace.jsonl" "$@" 2>&1 | tail -2
  echo "[$(date +%H:%M:%S)] DONE $name"
}

# reference: the released CTC-policy S2TT agent
echo "[$(date +%H:%M:%S)] START baseline-ctc"
common --agent "$ROOT/agent/speech_to_text.s2tt.streamspeech.agent.py" \
  --output "$OUT/baseline-ctc" 2>&1 | tail -2
echo "[$(date +%H:%M:%S)] DONE baseline-ctc"

# "offline" upper bound of the same beam-5 MT decoder: never commit before source end
run_conf offline-b5      --beam-size 5 --conf-threshold 1.01

# user's scheme: commit the whole top-1 continuation when its mean token prob >= thr
for thr in 0.0 0.3 0.5 0.7 0.9; do
  run_conf all-mean-$thr --beam-size 5 --conf-threshold $thr --conf-type mean --commit-mode all
done
# variant: gate on the weakest new token instead of the mean
for thr in 0.5 0.7; do
  run_conf all-min-$thr  --beam-size 5 --conf-threshold $thr --conf-type min --commit-mode all
done
# variant: commit only the leading run of tokens that are individually confident
for thr in 0.5 0.7 0.9; do
  run_conf prefix-$thr   --beam-size 5 --conf-threshold $thr --commit-mode prefix
done
# greedy (beam 1) for comparison with beam 5
run_conf all-mean-0.7-b1 --beam-size 1 --conf-threshold 0.7 --conf-type mean --commit-mode all
echo "[$(date +%H:%M:%S)] ALL DONE"
