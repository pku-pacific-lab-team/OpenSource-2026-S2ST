#!/usr/bin/env bash
# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

# 1000-utterance CVSS-C fr-en test-subset S2ST eval, scored with Whisper ASR-BLEU
# (whisper-small, lowercase + no punctuation, matching the normalisation of target.txt).
# Usage: bash <this repo>/model/quantization/run_test1k_s2st.sh <ckpt basename> <outname> [w4a4 0|1] [block]
set -u
ROOT="${STREAMSPEECH_ROOT:-$(pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB=datasets/cvss-eval/test1k
CHUNK=320
VOC="$ROOT/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en"

CKPT=$1; OUT=$2; W4A4=${3:-0}; BLOCK=${4:-32}
pp="$ROOT/fairseq"
if [ "$W4A4" = "1" ]; then pp="$ROOT/fairseq:$HERE"; export MXINT4_W4A4=1 MXINT4_BLOCK=$BLOCK; fi

RUN_DIR="res/fulltest/$OUT"
rm -rf "$RUN_DIR"

echo "[$(date +%H:%M:%S)] START $OUT"
PYTHONIOENCODING=utf-8 PYTHONPATH="$pp" simuleval \
  --data-bin "$ROOT/configs/fr-en" \
  --user-dir "$ROOT/researches/ctc_unity" --agent-dir "$ROOT/agent" \
  --source $SUB/wav_list.txt --target $SUB/target.txt \
  --model-path "$ROOT/pretrain_models/$CKPT" \
  --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
  --agent "$ROOT/agent/speech_to_speech.streamspeech.agent.py" \
  --vocoder "$VOC/g_00500000" --vocoder-cfg "$VOC/config.json" --dur-prediction \
  --output "$RUN_DIR" --source-segment-size $CHUNK \
  --quality-metrics WHISPER_ASR_BLEU --whisper-model-size small \
  --transcript-lowercase --transcript-non-punctuation \
  --latency-metrics AL LAAL RTF --device gpu
echo "[$(date +%H:%M:%S)] DONE $OUT -> $RUN_DIR"
