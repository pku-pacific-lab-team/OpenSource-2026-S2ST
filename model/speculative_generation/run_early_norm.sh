#!/usr/bin/env bash
# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

# stepwise prod>=0.5 + position-free speculative pre-synthesis (token mode, min 5 search steps)
set -u
ROOT="${STREAMSPEECH_ROOT:-$(pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOC="$ROOT/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en"
SUB=${1:-test100}; TAG=${2:-$SUB}; shift 2 || true
TAUS=${@:-"0.5 0.3 0.1"}
MINSTEPS=${MINSTEPS:-5}
DATA=datasets/cvss-eval/$SUB; OUT=res/conf-trigger/$TAG
for tau in $TAUS; do
  name=stepwise-prod-0.5-earlynorm-$tau
  echo "[$(date +%H:%M:%S)] START $name"
  rm -rf $OUT/$name
  PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT/fairseq" simuleval \
    --data-bin "$ROOT/configs/fr-en" \
    --user-dir "$ROOT/researches/ctc_unity" --agent-dir "$ROOT/agent" \
    --source $DATA/wav_list.txt --target $DATA/target.txt \
    --model-path "$ROOT/pretrain_models/streamspeech.simultaneous.fr-en.pt" \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --source-segment-size 320 \
    --quality-metrics BLEU --latency-metrics AL LAAL RTF --device gpu \
    --agent "$HERE/speech_to_text.s2tt.confidence.agent.py" \
    --beam-size 5 --conf-threshold 0.5 --conf-type prod --commit-mode stepwise \
    --tts-profile --vocoder "$VOC/g_00500000" --vocoder-cfg "$VOC/config.json" \
    --early-mass-thr $tau --early-mode token --min-steps $MINSTEPS --early-norm \
    --output "$OUT/$name" --trace-file "$OUT/$name/trace.jsonl" 2>&1 | grep -v STAGE | tail -2
  echo "[$(date +%H:%M:%S)] DONE $name"
done
echo "[$(date +%H:%M:%S)] ALL DONE"
