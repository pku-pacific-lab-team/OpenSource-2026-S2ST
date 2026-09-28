# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Profile per-component FLOPs of StreamSpeech during real SimulEval inference.

Usage: python profile_flops.py <source_segment_size_ms> <output_tag>
"""
import os
import sys
import functools
import time

import torch
from torch.profiler import profile, ProfilerActivity, record_function

SEG_SIZE = sys.argv[1] if len(sys.argv) > 1 else "320"
TAG = sys.argv[2] if len(sys.argv) > 2 else "streaming"

ROOT = os.environ.get("STREAMSPEECH_ROOT", os.getcwd()).replace("\\", "/")
VOC = f"{ROOT}/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en"

sys.argv = [
    "simuleval",
    "--data-bin", f"{ROOT}/configs/fr-en",
    "--user-dir", f"{ROOT}/researches/ctc_unity",
    "--agent-dir", f"{ROOT}/agent",
    "--source", "example/wav_list.txt",
    "--target", "example/target.txt",
    "--model-path", f"{ROOT}/pretrain_models/streamspeech.simultaneous.fr-en.pt",
    "--config-yaml", "config_gcmvn.yaml",
    "--multitask-config-yaml", "config_mtl_asr_st_ctcst.yaml",
    "--agent", f"{ROOT}/agent/speech_to_speech.streamspeech.agent.py",
    "--vocoder", f"{VOC}/g_00500000",
    "--vocoder-cfg", f"{VOC}/config.json",
    "--dur-prediction",
    "--output", f"res/profile-{TAG}",
    "--source-segment-size", SEG_SIZE,
    "--end-index", "1",
    "--latency-metrics", "AL",
    "--device", "gpu",
]

from simuleval.utils.agent import build_system_args
from simuleval.evaluator import build_evaluator

system, args = build_system_args()
evaluator = build_evaluator(args)

model = system.generator.model.single_model
components = {
    "encoder (Conformer)": model.encoder,
    "ASR CTC head": model.source_unigram_decoder,
    "ST CTC head (policy)": model.ctc_target_unigram_decoder,
    "MT decoder (text)": getattr(model, f"{model.mt_task_name}_decoder"),
    "T2U encoder": model.synthesizer_encoder,
    "unit CTC decoder": model.decoder,
    "vocoder (HiFi-GAN)": system.vocoder,
}

MARK = "COMP::"
CURRENT_SCOPE = [None]
conv_flops = {}
wall_time = {}


def patch(module, name):
    orig = module.forward

    @functools.wraps(orig)
    def wrapped(*a, **k):
        CURRENT_SCOPE[0] = name
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            with record_function(MARK + name):
                return orig(*a, **k)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_time[name] = wall_time.get(name, 0.0) + time.perf_counter() - t0
            CURRENT_SCOPE[0] = None

    module.forward = wrapped


for name, mod in components.items():
    patch(mod, name)


# torch profiler assigns zero FLOPs to conv1d/conv_transpose1d ops, which
# dominate the vocoder and appear in the conformer/subsampler; count those
# analytically via hooks using real input shapes
def conv_hook(mod, inputs, output):
    x = inputs[0]
    k = mod.kernel_size[0]
    if isinstance(mod, torch.nn.ConvTranspose1d):
        flops = 2 * x.shape[-1] * (mod.in_channels // mod.groups) * mod.out_channels * k
    else:
        flops = 2 * output.shape[-1] * (mod.in_channels // mod.groups) * mod.out_channels * k
    flops *= x.shape[0]
    scope = CURRENT_SCOPE[0] or "other"
    conv_flops[scope] = conv_flops.get(scope, 0) + flops


for root in list(components.values()):
    for m in root.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            m.register_forward_hook(conv_hook)

with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
    evaluator(system)


def subtree_flops(evt):
    total = evt.flops or 0
    for c in evt.cpu_children:
        total += subtree_flops(c)
    return total


scope_flops, scope_calls = {}, {}
grand_total = 0
for evt in prof.profiler.function_events:
    if evt.cpu_parent is None:
        grand_total += subtree_flops(evt)
    if evt.name.startswith(MARK):
        n = evt.name[len(MARK):]
        scope_flops[n] = scope_flops.get(n, 0) + subtree_flops(evt)
        scope_calls[n] = scope_calls.get(n, 0) + 1

import soundfile as sf

wav_path = open("example/wav_list.txt").readline().strip()
data, sr = sf.read(wav_path)
dur = len(data) / sr

total = grand_total + sum(conv_flops.values())
print("\n" + "=" * 84)
print(f"FLOPs breakdown  (segment={SEG_SIZE}ms, utterance #0: {dur:.2f}s French audio)")
print("=" * 84)
print(f"{'component':<28}{'calls':>6}{'matmul-G':>10}{'conv1d-G':>10}{'total-G':>10}{'share':>8}")
attributed = 0
for n in components:
    f_mm = scope_flops.get(n, 0)
    f_cv = conv_flops.get(n, 0)
    f = f_mm + f_cv
    attributed += f
    print(f"{n:<28}{scope_calls.get(n, 0):>6}{f_mm / 1e9:>10.2f}{f_cv / 1e9:>10.2f}{f / 1e9:>10.2f}{f / total * 100:>7.1f}%")
print("-" * 84)
print(f"{'attributed':<34}{'':>10}{'':>10}{attributed / 1e9:>10.2f}{attributed / total * 100:>7.1f}%")
print(f"{'TOTAL':<34}{'':>10}{'':>10}{total / 1e9:>10.2f}")
print(f"per second of input audio: {total / dur / 1e9:.2f} GFLOPs/s")

tw = sum(wall_time.values())
print(f"\n{'component':<28}{'wall-time s':>12}{'time share':>11}")
for n in components:
    t = wall_time.get(n, 0.0)
    print(f"{n:<28}{t:>12.3f}{t / tw * 100:>10.1f}%")
print(f"{'sum (component wall time)':<28}{tw:>12.3f}")
