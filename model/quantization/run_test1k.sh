#!/usr/bin/env bash
# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

# Sequential 1000-utterance CVSS-C fr-en test-subset evals: FP baseline + MXINT4 configs.
# Priority order: FP, then g32 (standard MX), then g8 (sweet spot), then g16/g4.
set -u
ROOT="${STREAMSPEECH_ROOT:-$(pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB=datasets/cvss-eval/test1k
CHUNK=320

run_one() {  # $1=ckpt basename  $2=task(asr|s2tt)  $3=outname  $4=w4a4(0|1)  $5=block
  local agent target
  if [ "$2" = "asr" ]; then
    agent=speech_to_text.asr.streamspeech.agent.py; target=$SUB/src.txt
  else
    agent=speech_to_text.s2tt.streamspeech.agent.py; target=$SUB/target.txt
  fi
  local pp="$ROOT/fairseq"
  local w4a4_env=""
  if [ "$4" = "1" ]; then pp="$ROOT/fairseq:$HERE"; fi
  echo "[$(date +%H:%M:%S)] START $3"
  if [ "$4" = "1" ]; then
    MXINT4_W4A4=1 MXINT4_BLOCK=$5 PYTHONIOENCODING=utf-8 PYTHONPATH="$pp" simuleval \
      --data-bin "$ROOT/configs/fr-en" \
      --user-dir "$ROOT/researches/ctc_unity" --agent-dir "$ROOT/agent" \
      --source $SUB/wav_list.txt --target "$target" \
      --model-path "$ROOT/pretrain_models/$1" \
      --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
      --agent "$ROOT/agent/$agent" \
      --output "res/fulltest/$3" --source-segment-size $CHUNK \
      --quality-metrics BLEU --latency-metrics AL LAAL RTF --device gpu 2>&1 | tail -3
  else
    PYTHONIOENCODING=utf-8 PYTHONPATH="$pp" simuleval \
      --data-bin "$ROOT/configs/fr-en" \
      --user-dir "$ROOT/researches/ctc_unity" --agent-dir "$ROOT/agent" \
      --source $SUB/wav_list.txt --target "$target" \
      --model-path "$ROOT/pretrain_models/$1" \
      --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
      --agent "$ROOT/agent/$agent" \
      --output "res/fulltest/$3" --source-segment-size $CHUNK \
      --quality-metrics BLEU --latency-metrics AL LAAL RTF --device gpu 2>&1 | tail -3
  fi
  echo "[$(date +%H:%M:%S)] DONE $3"
}

FP=streamspeech.simultaneous.fr-en.pt
G32=streamspeech.simultaneous.fr-en.mxint4-all.pt
G16=streamspeech.simultaneous.fr-en.mxint4-all-g16.pt
G8=streamspeech.simultaneous.fr-en.mxint4-all-g8.pt
G4=streamspeech.simultaneous.fr-en.mxint4-all-g4.pt

run_one $FP  s2tt fp-s2tt      0 32
run_one $FP  asr  fp-asr       0 32
run_one $G32 s2tt w4-g32-s2tt  0 32
run_one $G32 asr  w4-g32-asr   0 32
run_one $G32 s2tt w4a4-g32-s2tt 1 32
run_one $G32 asr  w4a4-g32-asr  1 32
run_one $G8  s2tt w4-g8-s2tt   0 8
run_one $G8  asr  w4-g8-asr    0 8
run_one $G8  s2tt w4a4-g8-s2tt 1 8
run_one $G8  asr  w4a4-g8-asr  1 8
run_one $G16 s2tt w4-g16-s2tt  0 16
run_one $G16 asr  w4-g16-asr   0 16
run_one $G16 s2tt w4a4-g16-s2tt 1 16
run_one $G16 asr  w4a4-g16-asr  1 16
run_one $G4  s2tt w4-g4-s2tt   0 4
run_one $G4  asr  w4-g4-asr    0 4
run_one $G4  s2tt w4a4-g4-s2tt 1 4
run_one $G4  asr  w4a4-g4-asr  1 4
echo "[$(date +%H:%M:%S)] ALL DONE"
