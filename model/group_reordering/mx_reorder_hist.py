# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Histogram of the within-tile adjacent-cycle |Δe| under a feeding order
(natural vs. Top-2 + shift x16 anchor reordering), g=8, 8x8 tiles.

Pairs touching an all-zero activation block (MT fc2) need no shift and are
counted as |Δe| = 0, consistent with the >1 shares reported by
mx_reorder_variants.py.

Usage: PYTHONPATH=fairseq python <this repo>/model/group_reordering/mx_reorder_hist.py --part asr --n-utts 10
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantization"))
from compare_logit_drift import load_task_and_model, features, encode, greedy_mt, CKPT_A, DATA_BIN  # noqa: E402
from mx_group_exp_gap import block_exp, kind_of, PARTS  # noqa: E402
from mx_tile_order import TILE, hamming_table  # noqa: E402
from mx_reorder_variants import reversal_engine, install  # noqa: E402

MAXGAP = 32


class HistStats:
    def __init__(self, n_iter, gre_thr):
        self.n_iter, self.gre_thr = n_iter, gre_thr
        self.hist = defaultdict(lambda: {"natural": torch.zeros(MAXGAP + 1, dtype=torch.long),
                                         "shift": torch.zeros(MAXGAP + 1, dtype=torch.long)})
        self.w_exp_cache = {}
        self.enabled = True

    def record(self, kind, x2d, w2d, name):
        ex = block_exp(x2d, 8)
        if name not in self.w_exp_cache:
            self.w_exp_cache[name] = block_exp(w2d, 8)
        ew = self.w_exp_cache[name]
        N, G = ex.shape
        O = ew.shape[0]
        vx, vw = ex > -500, ew > -500
        ex = torch.where(vx, ex, torch.zeros_like(ex))
        ew = torch.where(vw, ew, torch.zeros_like(ew))
        ot, gt = O // TILE, G // TILE
        acc = {"natural": 0, "shift": 0}
        for s in range(0, N, 32):
            exs = ex[s:s + 32]
            n = exs.shape[0]
            e = exs[:, None, :] + ew[None, :, :]
            valid = vx[s:s + 32, None, :] & vw[None, :, :]
            tiles = lambda a: a.reshape(n, ot, TILE, gt, TILE).permute(0, 1, 3, 2, 4).reshape(-1, TILE, TILE)
            e_t, v_t = tiles(e), tiles(valid)  # (B, 8 cols, 8 steps)
            hd = hamming_table(e_t.transpose(1, 2), v_t.transpose(1, 2))
            order = reversal_engine(hd, "top2-shift", self.gre_thr, max_iter=self.n_iter)
            B = e_t.shape[0]
            nat = torch.arange(TILE, device=e_t.device)[None].expand(B, -1)
            for key, o in (("natural", nat), ("shift", order)):
                idx = o[:, None, :].expand(-1, TILE, -1)
                eo, vo = e_t.gather(2, idx), v_t.gather(2, idx)
                d = (eo[:, :, 1:] - eo[:, :, :-1]).abs()
                d = torch.where(vo[:, :, 1:] & vo[:, :, :-1], d, torch.zeros_like(d))
                acc[key] = acc[key] + torch.bincount(d.flatten().long().clamp(max=MAXGAP), minlength=MAXGAP + 1).cpu()
        for k in (kind, "ALL"):
            for key in acc:
                self.hist[k][key] += acc[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-list", default="datasets/cvss-eval/test100/wav_list.txt")
    ap.add_argument("--n-utts", type=int, default=10)
    ap.add_argument("--part", choices=list(PARTS), default="asr")
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--gre-thr", type=int, default=13)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    with open(f"{DATA_BIN}/config_gcmvn.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.BaseLoader)
    gcmvn = np.load(cfg["global_cmvn"]["stats_npz_path"])
    task, model = load_task_and_model(CKPT_A)
    stats = HistStats(args.iters, args.gre_thr)
    install(model, stats, PARTS[args.part])
    wavs = [l.strip() for l in open(args.wav_list) if l.strip()][: args.n_utts]
    mt = model.target_unigram_decoder
    eos = task.multitask_tasks["target_unigram"].tgt_dict.eos()
    for w in wavs:
        feat = features(w, gcmvn)
        enc = encode(model, feat)
        if args.part == "asr":
            model.source_unigram_decoder(enc["encoder_out"][0])
        else:
            stats.enabled = False
            toks = greedy_mt(mt, enc, eos)
            stats.enabled = True
            mt(torch.LongTensor([[eos] + toks]).cuda(), encoder_out=enc)
    out = args.out or f"res/quant-mxint4/reorder_hist_top2shift{args.iters}_g8_{args.part}.pt"
    torch.save({k: {kk: v.clone() for kk, v in d.items()} for k, d in stats.hist.items()}, out)
    thr = [1, 2, 3, 4, 6, 8, 12, 16]
    for k, d in stats.hist.items():
        for key, h in d.items():
            n = h.sum().item()
            ex = " | ".join(f">{t}: {100*h[t+1:].sum().item()/n:.4f}%" for t in thr)
            print(f"[{k}] {key:8s} pairs={n:,} max={torch.nonzero(h).flatten()[-1].item()}  {ex}")
    print("saved", out)


if __name__ == "__main__":
    main()
