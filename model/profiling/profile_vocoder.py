# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Count parameters and FLOPs of the CodeHiFiGAN vocoder (unit -> 16kHz waveform).

Baseline input: 50 units (no duration prediction) -> exactly 50*320 = 16000
samples = 1.0 s of audio, so reported FLOPs are per second of generated speech.
FLOPs count multiply-add as 2 ops, matching profile_flops.py.

Usage (from the StreamSpeech root): PYTHONPATH=fairseq python <this repo>/model/profiling/profile_vocoder.py
"""
import json
import os
from collections import OrderedDict

import torch
import torch.nn as nn

ROOT = os.environ.get("STREAMSPEECH_ROOT", os.getcwd()).replace("\\", "/")
VOC = f"{ROOT}/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en"

from agent.tts.vocoder import CodeHiFiGANVocoderWithDur

with open(f"{VOC}/config.json") as f:
    cfg = json.load(f)
voc = CodeHiFiGANVocoderWithDur(f"{VOC}/g_00500000", cfg)
voc.eval()
gen = voc.model

# ---- grouping: map full module names to report rows ----
UP_CH = [cfg["upsample_initial_channel"] // (2 ** (i + 1)) for i in range(5)]


def group_of(name):
    if name.startswith("dict"):
        return "unit embedding (dict)"
    if name.startswith("dur_predictor"):
        return "duration predictor"
    if name.startswith("conv_pre"):
        return "conv_pre (128->512, k7)"
    if name.startswith("ups."):
        i = int(name.split(".")[1])
        return f"up{i} ConvT (r={cfg['upsample_rates'][i]}, {UP_CH[i]*2}->{UP_CH[i]})"
    if name.startswith("resblocks."):
        i = int(name.split(".")[1])
        s = i // 3
        return f"resblocks stage{s} (3 blk @ {UP_CH[s]}ch)"
    if name.startswith("conv_post"):
        return "conv_post (16->1, k7)"
    return "other"


# ---- parameters ----
params = OrderedDict()
for name, p in gen.named_parameters():
    g = group_of(name)
    params[g] = params.get(g, 0) + p.numel()

# ---- FLOPs via hooks (analytic; torch profiler reports 0 for conv1d) ----
flops = {}


def make_hook(name):
    def hook(mod, inputs, output):
        x = inputs[0]
        if isinstance(mod, nn.ConvTranspose1d):
            f = 2 * x.shape[-1] * (mod.in_channels // mod.groups) * mod.out_channels * mod.kernel_size[0]
            f *= x.shape[0]
        elif isinstance(mod, nn.Conv1d):
            f = 2 * output.shape[-1] * (mod.in_channels // mod.groups) * mod.out_channels * mod.kernel_size[0]
            f *= x.shape[0]
        elif isinstance(mod, nn.Linear):
            f = 2 * mod.in_features * mod.out_features * (x.numel() // x.shape[-1])
        else:  # Embedding: table lookup, no MACs
            f = 0
        g = group_of(name)
        flops[g] = flops.get(g, 0) + f

    return hook


for name, m in gen.named_modules():
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear, nn.Embedding)):
        m.register_forward_hook(make_hook(name))

N_UNITS = 50
torch.manual_seed(0)
code = torch.randint(0, cfg["num_embeddings"], (1, N_UNITS))

# run WITH dur_prediction once, only to capture the dur-predictor cost on N_UNITS
with torch.no_grad():
    wav_dp, dur = voc({"code": code.clone()}, dur_prediction=True)
dur_pred_flops = flops.get("duration predictor", 0)
expanded = int(dur.sum())

# clean baseline run: no dur expansion -> output is exactly N_UNITS*320 samples
flops.clear()
with torch.no_grad():
    wav, _ = voc({"code": code.clone()}, dur_prediction=False)
assert wav.numel() == N_UNITS * 320, wav.numel()
flops["duration predictor"] = dur_pred_flops  # measured separately above

sec = wav.numel() / cfg["sampling_rate"]
total_p = sum(params.values())
total_f = sum(flops.values())

print("=" * 78)
print(f"CodeHiFiGAN vocoder  |  input: {N_UNITS} units -> {wav.numel()} samples = {sec:.2f}s @16kHz")
print(f"(with --dur-prediction the same {N_UNITS} units expanded to {expanded} frames)")
print("=" * 78)
print(f"{'component':<38}{'params':>12}{'p-share':>9}{'GFLOPs':>10}{'f-share':>9}")
order = sorted(params, key=lambda g: -flops.get(g, 0))
for g in order:
    p, fl = params.get(g, 0), flops.get(g, 0)
    print(f"{g:<38}{p:>12,}{p/total_p*100:>8.2f}%{fl/1e9:>10.3f}{fl/total_f*100:>8.2f}%")
print("-" * 78)
print(f"{'TOTAL':<38}{total_p:>12,}{'':>9}{total_f/1e9:>10.3f}")
print(f"\nper second of generated audio : {total_f/sec/1e9:.2f} GFLOPs  ({total_f/sec/2e9:.2f} GMACs)")
print(f"per unit (20 ms of audio)     : {total_f/N_UNITS/1e6:.1f} MFLOPs")
print(f"total params: {total_p/1e6:.2f} M  ({total_p*4/1e6:.1f} MB fp32)")
