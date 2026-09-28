# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Calibrate per-component FLOPs (2 ops per MAC) as functions of sequence length,
for the critical-path latency model of the stepwise-commit policy.

Output: flops_calib.json next to this script

Method: torch.profiler counts matmul FLOPs; conv1d / conv_transpose1d are not
counted by the profiler and are added analytically via forward hooks (same
approach as profile_flops.py).  Everything is measured on the real modules
with random inputs of the given lengths.

Usage (from the StreamSpeech root): python <this repo>/model/speculative_generation/calibrate_flops.py
"""
import json
import sys

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

import os

ROOT = os.environ.get("STREAMSPEECH_ROOT", os.getcwd()).replace("\\", "/")
VOC = f"{ROOT}/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en"
PAIR = os.environ.get("LANG_PAIR", "fr-en")  # fr-en | es-en | de-en
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "flops_calib.json" if PAIR == "fr-en" else f"flops_calib_{PAIR}.json")

sys.argv = [
    "simuleval",
    "--data-bin", f"{ROOT}/configs/{PAIR}",
    "--user-dir", f"{ROOT}/researches/ctc_unity",
    "--agent-dir", f"{ROOT}/agent",
    "--source", "example/wav_list.txt",
    "--target", "example/target.txt",
    "--model-path", f"{ROOT}/pretrain_models/streamspeech.simultaneous.{PAIR}.pt",
    "--config-yaml", "config_gcmvn.yaml",
    "--multitask-config-yaml", "config_mtl_asr_st_ctcst.yaml",
    "--agent", f"{ROOT}/agent/speech_to_speech.streamspeech.agent.py",
    "--vocoder", f"{VOC}/g_00500000",
    "--vocoder-cfg", f"{VOC}/config.json",
    "--dur-prediction",
    "--output", "res/conf-trigger/calib-tmp",
    "--source-segment-size", "320",
    "--end-index", "1",
    "--latency-metrics", "AL",
    "--device", "gpu",
]
from simuleval.utils.agent import build_system_args  # noqa: E402

system, args = build_system_args()
model = system.generator.model.single_model
voc = system.vocoder
mt_decoder = getattr(model, f"{model.mt_task_name}_decoder")
dev = "cuda"
torch.manual_seed(0)

# ---------------------------------------------------------------- FLOPs meter
conv_flops = [0]


def conv_hook(mod, inputs, output):
    x = inputs[0]
    k = mod.kernel_size[0]
    if isinstance(mod, torch.nn.ConvTranspose1d):
        f = 2 * x.shape[-1] * (mod.in_channels // mod.groups) * mod.out_channels * k
    else:
        f = 2 * output.shape[-1] * (mod.in_channels // mod.groups) * mod.out_channels * k
    conv_flops[0] += f * x.shape[0]


for m in list(model.modules()) + list(voc.modules()):
    if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        m.register_forward_hook(conv_hook)


def measure(fn):
    conv_flops[0] = 0
    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU], with_flops=True
    ) as prof:
        fn()
    mm = sum((e.flops or 0) for e in prof.events())
    return float(mm + conv_flops[0])


def enc_out_of(n_fbank):
    src = torch.randn(1, n_fbank, 80, device=dev)
    lens = torch.tensor([n_fbank], device=dev)
    return model.encoder(src, src_lengths=lens)


calib = {}

# ---------------------------------------------------------------- 1. encoder
# marginal cost of the newest 320 ms chunk (32 fbank frames) at total length L
lens = list(range(32, 32 * 41, 32))  # up to 12.8 s
tot = []
for L in lens:
    tot.append(measure(lambda: enc_out_of(L)))
marg = [tot[0]] + [tot[i] - tot[i - 1] for i in range(1, len(tot))]
calib["encoder_marginal"] = {"fbank_len": lens, "flops": marg}
print("encoder marginal per chunk (GFLOPs) at 1s / 5s / 10s:",
      [round(marg[lens.index(L)] / 1e9, 3) for L in (96, 480, 960)])

# ---------------------------------------------------------------- 2. ASR CTC head
eo = enc_out_of(320)
T = eo["encoder_out"][0].size(0)
f_asr = measure(lambda: model.source_unigram_decoder(eo["encoder_out"][0]))
calib["asr_head_per_enc_frame"] = f_asr / T
print("ASR head per encoder frame (MFLOPs):", round(f_asr / T / 1e6, 3))

# ---------------------------------------------------------------- 3. MT decoder, one token with KV cache
# cost(P, T) for one token forward of one beam, prefix P cached, encoder T frames
V = len(system.generator_mt.tgt_dict)
rows = []
for n_fbank in (64, 128, 256, 400, 640, 960):
    eo = enc_out_of(n_fbank)
    T = eo["encoder_out"][0].size(0)
    for P in (0, 4, 8, 16, 32, 64):
        toks = torch.randint(4, V - 1, (1, P + 2), device=dev)
        toks[0, 0] = mt_decoder.dictionary.eos()
        state = {}
        with torch.inference_mode():
            # fill the KV cache token by token (with incremental_state the decoder
            # only processes the last token of the sequence it is given)
            for j in range(P + 1):
                mt_decoder(toks[:, : j + 1], encoder_out=eo, incremental_state=state)
            # one-token step on top of a cache of P+1 positions
            f = measure(lambda: mt_decoder(toks[:, : P + 2], encoder_out=eo, incremental_state=state))
        rows.append((P, T, f))
        print(f"  mt token fwd: P={P:3d} T={T:4d}  {f/1e6:8.2f} MFLOPs")
A = np.array([[1.0, p, t] for p, t, _ in rows])
y = np.array([f for _, _, f in rows])
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
calib["mt_dec_token"] = {"a": float(coef[0]), "b_prefix": float(coef[1]), "c_enc": float(coef[2])}
resid = float(np.abs(A @ coef - y).max() / y.mean())
print("MT decoder per-token: a=%.3g b_prefix=%.3g c_enc=%.3g (max rel resid %.3f)" % (*coef, resid))
print("  decoder dims: embed", mt_decoder.embed_dim if hasattr(mt_decoder, "embed_dim") else "?",
      "layers", len(mt_decoder.layers), "vocab", V)

# ---------------------------------------------------------------- 4. T2U encoder & 5. unit CTC decoder
# total cost as a function of text length P; marginal for new tokens = f(P+n) - f(P)
plen = list(range(1, 81))
t2u_tot, unit_tot = [], []
C = mt_decoder.embed_dim if hasattr(mt_decoder, "embed_dim") else mt_decoder.output_embed_dim
for P in plen:
    x = torch.randn(P, 1, C, device=dev)
    if getattr(model, "proj", None) is not None:
        with torch.inference_mode():
            x = model.proj(x)
    t2u_tot.append(measure(lambda: model.synthesizer_encoder(x, None)))
    with torch.inference_mode():
        t2u = model.synthesizer_encoder(x, None)
    unit_tot.append(measure(lambda: model.decoder(None, encoder_out=t2u)))
calib["t2u_total"] = {"len": plen, "flops": t2u_tot}
calib["unit_ctc_total"] = {"len": plen, "flops": unit_tot}
calib["ctc_upsample_rate"] = int(getattr(model.decoder, "ctc_upsample_rate", 0))
print("T2U total at P=10/40 (MFLOPs):", round(t2u_tot[9] / 1e6, 2), round(t2u_tot[39] / 1e6, 2))
print("unit CTC total at P=10/40 (MFLOPs):", round(unit_tot[9] / 1e6, 2), round(unit_tot[39] / 1e6, 2),
      "upsample", calib["ctc_upsample_rate"])

# ---------------------------------------------------------------- 6. vocoder (with duration prediction)
# cost ~ a + b*U (input units) + c*E (duration-expanded frames, 320 samples each)
rows = []
for U in (1, 2, 4, 8, 16, 32, 64):
    for rep in range(3):
        code = torch.randint(0, 1000, (1, U), device=dev)
        holder = {}

        def run():
            _, dur = voc({"code": code.clone()}, True)
            holder["E"] = int(dur.sum())

        f = measure(run)
        rows.append((U, holder["E"], f))
A = np.array([[1.0, u, e] for u, e, _ in rows])
y = np.array([f for _, _, f in rows])
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
calib["vocoder"] = {"a": float(coef[0]), "b_units": float(coef[1]), "c_expanded": float(coef[2])}
resid = float(np.abs(A @ coef - y).max() / y.mean())
print("vocoder: a=%.3g b_units=%.3g c_expanded=%.3g (max rel resid %.3f)" % (*coef, resid))
print("vocoder per expanded frame (MFLOPs):", round(coef[2] / 1e6, 2))

json.dump(calib, open(OUT, "w"), indent=1)
print("saved", OUT)
