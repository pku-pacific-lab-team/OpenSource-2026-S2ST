#!/usr/bin/env bash
# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

# Per-language-pair runs of the chosen policy on a 100-utt subset:
#   baseline-ctc (released CTC policy, BLEU reference)
#   stepwise-prod-0.5-tts        (stepwise prod>=0.5, no speculation, TTS shapes)
#   stepwise-prod-0.5-earlytok-0.1 (+ position-free speculation tau=0.1, min 5 steps)
# Usage: bash <this repo>/model/speculative_generation/run_lang.sh es-en es-test100
set -u
ROOT="${STREAMSPEECH_ROOT:-$(pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOC="$ROOT/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en"
PAIR=$1; SUB=$2
DATA=datasets/cvss-eval/$SUB
OUT=res/conf-trigger/$SUB
mkdir -p "$OUT"
common() {
  PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT/fairseq" simuleval \
    --data-bin "$ROOT/configs/$PAIR" \
    --user-dir "$ROOT/researches/ctc_unity" --agent-dir "$ROOT/agent" \
    --source $DATA/wav_list.txt --target $DATA/target.txt \
    --model-path "$ROOT/pretrain_models/streamspeech.simultaneous.$PAIR.pt" \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --source-segment-size 320 \
    --quality-metrics BLEU --latency-metrics AL LAAL RTF --device gpu "$@" 2>&1 | grep -v STAGE | tail -2
}
run() {  # $1=name, rest=agent args
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] START $SUB/$name"; rm -rf "$OUT/$name"
  common --output "$OUT/$name" "$@"
  echo "[$(date +%H:%M:%S)] DONE $SUB/$name"
}
run baseline-ctc --agent "$ROOT/agent/speech_to_text.s2tt.streamspeech.agent.py"
CONF="--agent $HERE/speech_to_text.s2tt.confidence.agent.py --beam-size 5 --conf-threshold 0.5 --conf-type prod --commit-mode stepwise --tts-profile --vocoder $VOC/g_00500000 --vocoder-cfg $VOC/config.json"
run stepwise-prod-0.5-tts $CONF --trace-file "$OUT/stepwise-prod-0.5-tts/trace.jsonl"
run stepwise-prod-0.5-earlytok-0.1 $CONF --early-mass-thr 0.1 --early-mode token --min-steps 5 \
    --trace-file "$OUT/stepwise-prod-0.5-earlytok-0.1/trace.jsonl"
echo "[$(date +%H:%M:%S)] ALL DONE $SUB"
