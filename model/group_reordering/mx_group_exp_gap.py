# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""MX group-scale gap statistics for the ASR path (encoder Transformer part + CTC head).

For every Linear / 1x1-Conv matmul  Y = X W^T  in the encoder Conformer layers
(attention q/k/v/out/pos projections, FFN w_1/w_2, conv-module pointwise convs)
and the ASR CTC head, the reduction dim K is split into K/g MX groups of g
elements. Each group has a shared E8M0 exponent for the activation block
(e_x[token, group]) and for the weight block (e_w[out, group]); the partial
dot product of that group carries the combined exponent e_x + e_w.

We look at every *adjacent* pair of groups (j, j+1) that must be accumulated
into the same output element and record |Δe| = |(e_x+e_w)[j+1] - (e_x+e_w)[j]|,
i.e. how many bit positions the two partial sums have to be shifted relative
to each other before adding. Reported as a histogram + exceedance counts.

Groups whose activation or weight block is all-zero contribute nothing to the
sum, so pairs touching such a group are excluded (counted separately).

The exponent only depends on the block amax, so the result is the same for
MXINT4 / MXINT8 / MXFP* elements — only g matters.

Usage (from repo root, venv active):
  PYTHONPATH=fairseq python <this repo>/model/group_reordering/mx_group_exp_gap.py \
      --wav-list example/wav_list.txt --n-utts 10 --group 8
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantization"))
from compare_logit_drift import (load_task_and_model, features, encode, greedy_mt,  # noqa: E402
                                 CKPT_A, DATA_BIN)

MAXGAP = 64  # histogram bins 0..MAXGAP (values above are clipped into the last bin)


def block_exp(x2d: torch.Tensor, g: int) -> torch.Tensor:
    """x2d: (R, K) -> (R, K//g) E8M0 exponents; -inf-like marker (-1000) for all-zero blocks."""
    R, K = x2d.shape
    pad = (g - K % g) % g
    if pad:
        x2d = torch.nn.functional.pad(x2d, (0, pad))
    xb = x2d.reshape(R, -1, g)
    amax = xb.abs().amax(-1)
    exp = torch.floor(torch.log2(amax.clamp(min=2.0 ** -126)))
    exp = torch.where(amax > 0, exp, torch.full_like(exp, -1000.0))
    return exp


class GapStats:
    def __init__(self, g):
        self.g = g
        self.hist = defaultdict(lambda: torch.zeros(MAXGAP + 1, dtype=torch.long))
        self.hist_mod = defaultdict(lambda: torch.zeros(MAXGAP + 1, dtype=torch.long))
        self.hist_x = defaultdict(lambda: torch.zeros(MAXGAP + 1, dtype=torch.long))
        self.hist_w = defaultdict(lambda: torch.zeros(MAXGAP + 1, dtype=torch.long))
        self.zero_pairs = defaultdict(int)
        self.calls = defaultdict(int)
        self.w_exp_cache = {}
        self.w_done = set()
        self.enabled = True

    def _acc(self, hist, d):
        hist += torch.bincount(d.clamp(max=MAXGAP).flatten(), minlength=MAXGAP + 1).cpu()

    def record(self, kind, x2d, w2d, name):
        g = self.g
        ex = block_exp(x2d, g)  # (N, G)
        if name not in self.w_exp_cache:
            self.w_exp_cache[name] = block_exp(w2d, g)  # (O, G)
        ew = self.w_exp_cache[name]
        e = ex[:, None, :] + ew[None, :, :]  # (N, O, G)
        valid = (ex[:, None, :] > -500) & (ew[None, :, :] > -500)
        pair_valid = valid[..., 1:] & valid[..., :-1]
        d = (e[..., 1:] - e[..., :-1]).abs().to(torch.int64)
        dv = d[pair_valid]
        self._acc(self.hist[kind], dv)
        self._acc(self.hist_mod[name], dv)
        self.zero_pairs[kind] += int((~pair_valid).sum())
        # activation-only and weight-only gaps for reference
        vx = (ex[:, 1:] > -500) & (ex[:, :-1] > -500)
        self._acc(self.hist_x[kind], (ex[:, 1:] - ex[:, :-1]).abs().to(torch.int64)[vx])
        if name not in self.w_done:
            vw = (ew[:, 1:] > -500) & (ew[:, :-1] > -500)
            self._acc(self.hist_w[kind], (ew[:, 1:] - ew[:, :-1]).abs().to(torch.int64)[vw])
            self.w_done.add(name)
        self.calls[kind] += 1


def kind_of(name):
    if name.startswith("target_unigram_decoder"):
        if "self_attn" in name:
            return "MT self-attn proj (q/k/v/out)"
        if "encoder_attn" in name:
            return "MT cross-attn proj (q/k/v/out)"
        if "fc1" in name or "fc2" in name:
            return "MT ffn (fc1/fc2)"
        if "output_projection" in name:
            return "MT output projection (vocab 6000)"
        return "MT other"
    if "self_attn" in name:
        return "attn proj (q/k/v/out/pos)"
    if "ffn" in name:
        return "ffn (w_1/w_2)"
    if "pointwise_conv" in name:
        return "conv-module pointwise (1x1)"
    if "source_unigram_decoder" in name:
        return "ASR CTC head"
    return "other"


PARTS = {
    "asr": ("encoder.conformer_layers", "source_unigram_decoder"),
    "mt": ("target_unigram_decoder",),
}


def install(model, stats, prefixes):
    n = 0
    for name, mod in model.named_modules():
        if not name.startswith(prefixes):
            continue
        if isinstance(mod, nn.Linear):
            def hook(m, inp, name=name):
                if not stats.enabled:
                    return
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                stats.record(kind_of(name), x, m.weight.float(), name)
            mod.register_forward_pre_hook(hook)
            n += 1
        elif isinstance(mod, nn.Conv1d) and mod.kernel_size == (1,):
            def hook(m, inp, name=name):
                if not stats.enabled:
                    return
                x = inp[0].transpose(1, 2).reshape(-1, inp[0].shape[1]).float()  # (B*T, C)
                stats.record(kind_of(name), x, m.weight.reshape(m.weight.shape[0], -1).float(), name)
            mod.register_forward_pre_hook(hook)
            n += 1
    return n


def report(title, hists, zero_pairs=None, thresholds=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16)):
    print(f"\n### {title}")
    kinds = list(hists.keys())
    total = sum(hists.values())
    hists = dict(hists)
    hists["ALL"] = total
    for kind in kinds + ["ALL"]:
        h = hists[kind].double()
        n = h.sum().item()
        if n == 0:
            continue
        idx = torch.arange(len(h), dtype=torch.double)
        mean = (h * idx).sum().item() / n
        nz = torch.nonzero(h).flatten()
        mx = nz[-1].item() if len(nz) else 0
        print(f"\n[{kind}]  adjacent-group pairs = {int(n):,}  mean|Δe| = {mean:.2f}  max = {mx}"
              + (f"  (excluded pairs touching an all-zero block: {zero_pairs[kind]:,})"
                 if zero_pairs and kind in zero_pairs else ""))
        print("  histogram |Δe| = 0..12:  " + "  ".join(f"{k}:{100*h[k].item()/n:.2f}%" for k in range(13)))
        line = []
        for t in thresholds:
            c = h[t + 1:].sum().item()
            line.append(f">{t}: {int(c):,} ({100*c/n:.3f}%)")
        print("  count |Δe| > T :  " + " | ".join(line))


def report_per_module(hist_mod, target=0.25):
    """Per matmul: threshold T* = smallest T with P(|Δe| > T) <= target (i.e. the
    (1-target)-quantile of |Δe|), plus the exceedance just above and below it."""
    print(f"\n### per-matmul threshold with exceedance ~{target*100:.0f}%  "
          f"(T* = smallest T with P(|Δe|>T) <= {target:.2f})")
    print(f"{'module':52s} {'mean':>5s} {'T*':>3s} {'P(>T*-1)':>9s} {'P(>T*)':>8s} {'P(>T*+1)':>9s} {'max':>4s}")
    rows = []
    for name in sorted(hist_mod, key=lambda s: (len(s.split(".")), s)):
        h = hist_mod[name].double()
        n = h.sum().item()
        idx = torch.arange(len(h), dtype=torch.double)
        mean = (h * idx).sum().item() / n
        exceed = lambda t: h[t + 1:].sum().item() / n if t >= 0 else 1.0
        t_star = next(t for t in range(MAXGAP + 1) if exceed(t) <= target)
        mx = torch.nonzero(h).flatten()[-1].item()
        rows.append((name, mean, t_star, exceed(t_star - 1), exceed(t_star), exceed(t_star + 1), mx))
    for name, mean, t, lo, mid, hi, mx in rows:
        short = name.replace("encoder.conformer_layers.", "L")
        print(f"{short:52s} {mean:5.2f} {t:3d} {lo*100:8.1f}% {mid*100:7.1f}% {hi*100:8.1f}% {mx:4d}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-list", default="example/wav_list.txt")
    ap.add_argument("--n-utts", type=int, default=10)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--ckpt", default=CKPT_A)
    ap.add_argument("--save-hist", default=None, help="save the combined-gap histograms (per kind) to this .pt")
    ap.add_argument("--part", choices=list(PARTS), default="asr",
                    help="asr: encoder conformer layers + ASR CTC head; mt: target_unigram_decoder")
    ap.add_argument("--target", type=float, default=0.25,
                    help="target exceedance fraction for the per-matmul threshold report")
    ap.add_argument("--w4a4", action="store_true",
                    help="fake-quantize activations to MXINT4 (same group) before measuring, "
                         "so later layers see quantized-model activations; pair with a "
                         "mxint4-*.pt checkpoint for full W4A4")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    with open(f"{DATA_BIN}/config_gcmvn.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.BaseLoader)
    gcmvn = np.load(cfg["global_cmvn"]["stats_npz_path"])
    task, model = load_task_and_model(args.ckpt)

    stats = GapStats(args.group)
    if args.w4a4:
        # registered first, so the measuring hooks below see the quantized input
        from mx_act_quant import install_hooks
        print(f"activation MXINT4 hooks: {install_hooks(model, block=args.group)}")
    n = install(model, stats, PARTS[args.part])
    wavs = [l.strip() for l in open(args.wav_list) if l.strip()][: args.n_utts]
    print(f"part={args.part}, group={args.group}, hooked {n} matmuls, {len(wavs)} utterances")
    frames, tgt_tokens = 0, 0
    mt = model.target_unigram_decoder
    eos = task.multitask_tasks["target_unigram"].tgt_dict.eos()
    for w in wavs:
        feat = features(w, gcmvn)
        enc = encode(model, feat)
        frames += enc["encoder_out"][0].shape[0]
        if args.part == "asr":
            model.source_unigram_decoder(enc["encoder_out"][0])
        else:
            # greedy decode without recording (each prefix would be re-counted),
            # then one teacher-forced pass over the greedy output with recording on
            stats.enabled = False
            toks = greedy_mt(mt, enc, eos)
            stats.enabled = True
            prev = torch.LongTensor([[eos] + toks]).cuda()
            mt(prev, encoder_out=enc)
            tgt_tokens += prev.shape[1]
    print(f"total encoder frames: {frames}, MT decoder positions: {tgt_tokens}")

    if args.save_hist:
        torch.save({k: v.clone() for k, v in stats.hist.items()}, args.save_hist)
    report(f"COMBINED exponent gap |Δ(e_x + e_w)| between adjacent groups, g={args.group}",
           stats.hist, stats.zero_pairs)
    report_per_module(stats.hist_mod, target=args.target)
    report("activation-only gap |Δe_x| (per token, per adjacent group pair)", stats.hist_x)
    report("weight-only gap |Δe_w| (per output row, per adjacent group pair; each weight counted once)",
           stats.hist_w)


if __name__ == "__main__":
    main()
