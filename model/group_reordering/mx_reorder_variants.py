# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Variants of the anchor-based Group Reordering Engine, evaluated on real tiles.

All variants only use the 8x8 Hamming table (2-bit bins, column-mismatch HD)
like the original GRE; the true |Δe|>thr pair count is used for evaluation only.

  gre            : original (anchor filter + sub-sequence reversal), fixed threshold
  gre-sched      : same engine, but the anchor threshold decreases every iteration
                   (14,13,...,9) so the anchor set / candidate runs change each round
  nn             : greedy nearest-neighbour chain on the HD table (best of 8 starts)
  nn+gre         : nn as initial order, then gre polish
  anchor-move    : anchor filter picks the worst anchor (largest HD sum), the local
                   engine pulls that row out and re-inserts it at the best of the 8
                   gap positions (accepted if the HD path sum drops); iterate
  nn+anchor-move : nn init, then anchor-move
  gre+anchor-move: gre, then anchor-move

Usage:
  PYTHONPATH=fairseq python <this repo>/model/group_reordering/mx_reorder_variants.py --n-utts 1 --gre-thr 13
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
from compare_logit_drift import load_task_and_model, features, encode, greedy_mt, CKPT_A, DATA_BIN  # noqa: E402
from mx_group_exp_gap import block_exp, kind_of, PARTS  # noqa: E402
from mx_tile_order import (TILE, pair_costs, natural_cost, order_cost, min_path_cost,  # noqa: E402
                           hamming_table, gre_orders)


def hd_path_cost(hd, order):
    return order_cost(hd, order)


def greedy_nn(hd: torch.Tensor, starts=range(TILE)) -> torch.Tensor:
    """Nearest-neighbour chain from each start row in `starts`; keep the one with the
    smallest HD path sum (a single start = the cheapest hardware version)."""
    B, dev = hd.shape[0], hd.device
    ar = torch.arange(B, device=dev)
    best_cost = torch.full((B,), 1 << 30, dtype=torch.int32, device=dev)
    best = None
    big = torch.tensor(1 << 20, dtype=hd.dtype, device=dev)
    for s in starts:
        order = torch.empty(B, TILE, dtype=torch.long, device=dev)
        cur = torch.full((B,), s, dtype=torch.long, device=dev)
        used = torch.zeros(B, TILE, dtype=torch.bool, device=dev)
        used[:, s] = True
        order[:, 0] = cur
        for k in range(1, TILE):
            row = torch.where(used, big, hd[ar, cur])
            cur = row.argmin(1)
            used[ar, cur] = True
            order[:, k] = cur
        c = hd_path_cost(hd, order)
        better = c < best_cost
        best_cost = torch.where(better, c, best_cost)
        best = order if best is None else torch.where(better[:, None], order, best)
    return best


def anchor_move(hd: torch.Tensor, thr: int, max_iter: int = 8, init_order=None):
    """Anchor filter (HD sum of both neighbours > thr) -> take the worst anchor, remove its
    row and re-insert at the best gap. Accept if the HD path sum strictly decreases."""
    B, dev = hd.shape[0], hd.device
    order = (torch.arange(TILE, device=dev)[None].expand(B, -1).clone()
             if init_order is None else init_order.clone())
    ar = torch.arange(B, device=dev)
    pos = torch.arange(TILE, device=dev)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    for _ in range(max_iter):
        hd_o = hd[ar[:, None, None], order[:, :, None], order[:, None, :]]
        s = torch.zeros(B, TILE, dtype=torch.int32, device=dev)
        s[:, 1:] += hd_o[:, pos[1:], pos[:-1]]
        s[:, :-1] += hd_o[:, pos[:-1], pos[1:]]
        anchor = s > thr
        has_anchor = anchor.any(1)
        p_star = torch.where(anchor, s, torch.full_like(s, -1)).argmax(1)  # worst anchor position
        row = order[ar, p_star]
        keep = pos[None].expand(B, -1) != p_star[:, None]
        rest = order[keep].reshape(B, TILE - 1)
        cur_cost = hd_path_cost(hd, order)
        best_cost, best_order = cur_cost.clone(), order.clone()
        for gap in range(TILE):
            cand = torch.cat([rest[:, :gap], row[:, None], rest[:, gap:]], 1)
            c = hd_path_cost(hd, cand)
            better = c < best_cost
            best_cost = torch.where(better, c, best_cost)
            best_order = torch.where(better[:, None], cand, best_order)
        improved = best_cost < cur_cost
        apply = active & has_anchor & improved
        order = torch.where(apply[:, None], best_order, order)
        active = apply
        if not active.any():
            break
    return order


def reversal_engine(hd: torch.Tensor, anchor_mode: str, thr: int, max_iter: int = 8, init_order=None):
    """Same sub-sequence reversal engine as GRE, with pluggable anchor selection:
      sum      : left+right HD > thr                       (original)
      sum-noadj: as sum, but of two adjacent anchors only the larger sum stays
      max      : max(left, right) HD > thr
      top2     : the two largest sums (non-adjacent), regardless of value
      edge     : cut at edges with HD > thr; segments between cuts are the candidates
                 (no fixed rows; alternate parity of segments per iteration to avoid
                 conflicting flips of neighbouring segments)
    Returns final order (B, 8)."""
    B, dev = hd.shape[0], hd.device
    order = (torch.arange(TILE, device=dev)[None].expand(B, -1).clone()
             if init_order is None else init_order.clone())
    ar = torch.arange(B, device=dev)
    pos = torch.arange(TILE, device=dev)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    prev_anchor = torch.zeros(B, TILE, dtype=torch.bool, device=dev)
    for it in range(max_iter):
        hd_o = hd[ar[:, None, None], order[:, :, None], order[:, None, :]]
        left = torch.zeros(B, TILE, dtype=torch.int32, device=dev)
        right = torch.zeros_like(left)
        left[:, 1:] = hd_o[:, pos[1:], pos[:-1]]
        right[:, :-1] = hd_o[:, pos[:-1], pos[1:]]
        s = left + right
        if anchor_mode == "edge":
            cut = right[:, :-1] > thr  # cut between p and p+1
            has_work = cut.any(1)
            anchor = torch.zeros(B, TILE, dtype=torch.bool, device=dev)
        else:
            if anchor_mode == "sum":
                anchor = s > thr
            elif anchor_mode == "max":
                anchor = torch.maximum(left, right) > thr
            elif anchor_mode == "top2-shift" and it > 0:
                anchor = torch.roll(prev_anchor, 1, dims=1)  # rotate the anchor pattern by one position
            elif anchor_mode in ("sum-noadj", "top2", "top2-rotate", "top2-shift"):
                if anchor_mode.startswith("top2"):
                    anchor = torch.zeros(B, TILE, dtype=torch.bool, device=dev)
                    s_sel = s.clone()
                    if anchor_mode == "top2-rotate":
                        # positions that were anchors last round are not eligible this round
                        s_sel = torch.where(prev_anchor, torch.full_like(s_sel, -1), s_sel)
                    top = s_sel.topk(2, dim=1).indices
                    anchor[ar, top[:, 0]] = True
                    second_ok = (top[:, 1] - top[:, 0]).abs() > 1
                    anchor[ar[second_ok], top[second_ok, 1]] = True
                else:
                    anchor = s > thr
                # drop the smaller of two adjacent anchors
                l_adj = torch.zeros_like(anchor)
                l_adj[:, 1:] = anchor[:, :-1] & (s[:, :-1] >= s[:, 1:])
                r_adj = torch.zeros_like(anchor)
                r_adj[:, :-1] = anchor[:, 1:] & (s[:, 1:] > s[:, :-1])
                anchor = anchor & ~(l_adj | r_adj)
            has_work = anchor.any(1)
            cut = anchor[:, :-1] | anchor[:, 1:]  # segment boundary next to an anchor
        # segments: sp = last boundary before p, ep = next boundary after p
        sp = torch.zeros_like(order)
        ep = torch.full_like(order, TILE - 1)
        last = torch.zeros((B,), dtype=torch.long, device=dev)
        seg_id = torch.zeros(B, TILE, dtype=torch.long, device=dev)
        for p in range(1, TILE):
            last = torch.where(cut[:, p - 1], torch.full_like(last, p), last)
            sp[:, p] = last
            seg_id[:, p] = seg_id[:, p - 1] + cut[:, p - 1].long()
        nxt = torch.full((B,), TILE - 1, dtype=torch.long, device=dev)
        for p in reversed(range(TILE - 1)):
            nxt = torch.where(cut[:, p], torch.full_like(nxt, p), nxt)
            ep[:, p] = nxt
        o_sp, o_ep = order.gather(1, sp), order.gather(1, ep)
        has_l, has_r = sp > 0, ep < TILE - 1
        o_l = order.gather(1, (sp - 1).clamp(min=0))
        o_r = order.gather(1, (ep + 1).clamp(max=TILE - 1))
        hd_b = hd[ar[:, None], o_sp, o_l] * has_l + hd[ar[:, None], o_ep, o_r] * has_r
        hd_a = hd[ar[:, None], o_sp, o_r] * has_r + hd[ar[:, None], o_ep, o_l] * has_l
        rev = (hd_a < hd_b) & ~anchor & active[:, None]
        if anchor_mode == "edge":
            rev &= (seg_id % 2) == (it % 2)
        newpos = torch.where(rev, sp + ep - pos[None], pos[None].expand(B, -1))
        new_order = torch.empty_like(order)
        new_order.scatter_(1, newpos, order)
        changed = (new_order != order).any(1)
        order = new_order
        prev_anchor = anchor
        if anchor_mode == "edge":
            active = active & has_work & (changed | (it % 2 == 0))  # give the other parity a try
        elif anchor_mode in ("top2-rotate", "top2-shift"):
            active = active & has_work  # the anchor set changes every round, so keep going
        else:
            active = active & has_work & changed
        if not active.any():
            break
    return order


SEGS = [(i, j) for i in range(TILE) for j in range(i + 1, TILE)]  # all 28 contiguous sub-sequences


def two_opt(hd: torch.Tensor, max_iter: int = 16, init_order=None):
    """Best-improvement 2-opt on the HD path: each round evaluate reversing every one of the
    28 contiguous sub-sequences (2 HD lookups each), apply the best if it reduces the sum."""
    B, dev = hd.shape[0], hd.device
    order = (torch.arange(TILE, device=dev)[None].expand(B, -1).clone()
             if init_order is None else init_order.clone())
    ar = torch.arange(B, device=dev)
    pos = torch.arange(TILE, device=dev)
    for _ in range(max_iter):
        deltas = []
        for i, j in SEGS:
            o_i, o_j = order[:, i], order[:, j]
            d = torch.zeros(B, dtype=torch.int32, device=dev)
            if i > 0:
                o_l = order[:, i - 1]
                d += hd[ar, o_j, o_l] - hd[ar, o_i, o_l]
            if j < TILE - 1:
                o_r = order[:, j + 1]
                d += hd[ar, o_i, o_r] - hd[ar, o_j, o_r]
            deltas.append(d)
        deltas = torch.stack(deltas, 1)  # (B, 28)
        best, idx = deltas.min(1)
        improve = best < 0
        if not improve.any():
            break
        seg = torch.tensor(SEGS, device=dev)[idx]  # (B, 2)
        i, j = seg[:, 0:1], seg[:, 1:2]
        inside = (pos[None] >= i) & (pos[None] <= j)
        newpos = torch.where(inside & improve[:, None], i + j - pos[None], pos[None].expand(B, -1))
        new_order = torch.empty_like(order)
        new_order.scatter_(1, newpos, order)
        order = new_order
    return order


def gre_sched(hd, thrs=(14, 13, 12, 11, 10, 9), iters_per_thr=1, init_order=None):
    order = init_order
    for t in thrs:
        snaps, _ = gre_orders(hd, t, max_iter=iters_per_thr, init_order=order)
        order = snaps[max(snaps)]
    return order


class Stats:
    def __init__(self, thr, gre_thr, suite="engines"):
        self.thr, self.gre_thr, self.suite = thr, gre_thr, suite
        self.methods = []
        self.tot = defaultdict(lambda: defaultdict(int))
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
        acc = defaultdict(int)
        for s in range(0, N, 32):
            exs = ex[s:s + 32]
            n = exs.shape[0]
            e = exs[:, None, :] + ew[None, :, :]
            valid = vx[s:s + 32, None, :] & vw[None, :, :]
            tiles = lambda a: a.reshape(n, ot, TILE, gt, TILE).permute(0, 1, 3, 2, 4).reshape(-1, TILE, TILE)
            e_t, v_t = tiles(e), tiles(valid)
            cost = pair_costs(e_t, self.thr, v_t)
            hd = hamming_table(e_t.transpose(1, 2), v_t.transpose(1, 2))
            t = self.gre_thr
            orders = {}
            if self.suite == "anchors":
                orders["gre"] = gre_orders(hd, t)[0][8]
                orders["sum-noadj"] = reversal_engine(hd, "sum-noadj", t)
                for mt in (6, 7, 8):
                    orders[f"max>{mt}"] = reversal_engine(hd, "max", mt)
                orders["top2"] = reversal_engine(hd, "top2", t)
                for et in (4, 5, 6, 7):
                    orders[f"edge>{et}"] = reversal_engine(hd, "edge", et)
            elif self.suite == "shift":
                orders["gre"] = gre_orders(hd, t)[0][8]
                for n_it in (8, 16, 32):
                    orders[f"top2-shift x{n_it}"] = reversal_engine(hd, "top2-shift", t, max_iter=n_it)
                orders["2opt"] = two_opt(hd)
            elif self.suite == "rotate":
                orders["gre"] = gre_orders(hd, t)[0][8]
                orders["top2"] = reversal_engine(hd, "top2", t)
                for n_it in (4, 8, 16):
                    orders[f"top2-rotate x{n_it}"] = reversal_engine(hd, "top2-rotate", t, max_iter=n_it)
                    orders[f"top2-shift x{n_it}"] = reversal_engine(hd, "top2-shift", t, max_iter=n_it)
                orders["2opt"] = two_opt(hd)
                orders["nn"] = greedy_nn(hd)
                orders["nn+2opt"] = two_opt(hd, init_order=orders["nn"])
            else:
                orders["gre"] = gre_orders(hd, t)[0][8]
                orders["gre-sched"] = gre_sched(hd)
                nn_o = greedy_nn(hd)
                orders["nn"] = nn_o
                orders["nn-start0"] = greedy_nn(hd, starts=(0,))
                orders["nn-2starts"] = greedy_nn(hd, starts=(0, 7))
                orders["nn+gre"] = gre_orders(hd, t, init_order=nn_o)[0][8]
                orders["anchor-move"] = anchor_move(hd, t)
                orders["nn+anchor-move"] = anchor_move(hd, t, init_order=nn_o)
                orders["gre+anchor-move"] = anchor_move(hd, t, init_order=orders["gre"])
            self.methods = list(orders)
            acc["natural"] += natural_cost(cost).sum().item()
            acc["optimal"] += min_path_cost(cost).sum().item()
            for k, o in orders.items():
                acc[k] += order_cost(cost, o).sum().item()
            acc["pairs"] += cost.shape[0] * TILE * (TILE - 1)
        for k in (kind, "ALL"):
            for m, v in acc.items():
                self.tot[k][m] += v


def install(model, stats, prefixes):
    n = 0
    for name, mod in model.named_modules():
        if not name.startswith(prefixes):
            continue
        if isinstance(mod, nn.Linear):
            def hook(m, inp, name=name):
                if stats.enabled:
                    stats.record(kind_of(name), inp[0].reshape(-1, inp[0].shape[-1]).float(),
                                 m.weight.float(), name)
            mod.register_forward_pre_hook(hook)
            n += 1
        elif isinstance(mod, nn.Conv1d) and mod.kernel_size == (1,):
            def hook(m, inp, name=name):
                if stats.enabled:
                    x = inp[0].transpose(1, 2).reshape(-1, inp[0].shape[1]).float()
                    stats.record(kind_of(name), x, m.weight.reshape(m.weight.shape[0], -1).float(), name)
            mod.register_forward_pre_hook(hook)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-list", default="datasets/cvss-eval/test100/wav_list.txt")
    ap.add_argument("--n-utts", type=int, default=1)
    ap.add_argument("--thr", type=int, default=1)
    ap.add_argument("--gre-thr", type=int, default=13)
    ap.add_argument("--part", choices=list(PARTS), default="asr")
    ap.add_argument("--suite", choices=["engines", "anchors", "rotate", "shift"], default="engines")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    with open(f"{DATA_BIN}/config_gcmvn.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.BaseLoader)
    gcmvn = np.load(cfg["global_cmvn"]["stats_npz_path"])
    task, model = load_task_and_model(CKPT_A)
    stats = Stats(args.thr, args.gre_thr, args.suite)
    n = install(model, stats, PARTS[args.part])
    wavs = [l.strip() for l in open(args.wav_list) if l.strip()][: args.n_utts]
    print(f"part={args.part} g=8 |Δe|>{args.thr}, anchor thr={args.gre_thr}, hooked {n}, {len(wavs)} utts", flush=True)
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

    methods = ["natural"] + stats.methods + ["optimal"]
    kinds = [k for k in stats.tot if k != "ALL"] + ["ALL"]
    print(f"\n{'':36s}" + "".join(f"{m:>16s}" for m in methods))
    for k in kinds:
        t = stats.tot[k]
        print(f"{k:36s}" + "".join(f"{100*t[m]/t['pairs']:15.2f}%" for m in methods))
    t = stats.tot["ALL"]
    print("\nrelative reduction vs natural (ALL): " + ", ".join(
        f"{m} -{100*(1-t[m]/t['natural']):.0f}%" for m in methods[1:]))


if __name__ == "__main__":
    main()
