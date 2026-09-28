# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Row-feeding-order optimisation for an 8x8 input-broadcast MX array (ASR path).

Hardware model (confirmed):
  Per cycle the array does an 8-vector x 8x8-matrix product for ONE K-group g:
  the activation group x_g (8 elements, exponent e_x[t,g]) is broadcast to 8
  columns; column j holds the weight group W[j,g] (exponent e_w[j,g]) and adds
  the partial dot product into its accumulator. Aligning consecutive partial
  sums in column j costs |Δe| = |(e_x+e_w)[t,j,g_k] - (e_x+e_w)[t,j,g_{k+1}]|.

  A tile = 8 consecutive K-groups x 8 output channels (64 weight groups); it
  takes 8 cycles (one "row" of 8 weight groups per cycle) and the order pi of
  those 8 rows is free and may be re-chosen for every token. Per tile:
  8 columns x 7 adjacent cycles = 56 alignment pairs.

For every (token, channel tile, K tile) we find the exact order minimising the
number of pairs with |Δe| > thr (Held-Karp DP over the 8x8 pairwise cost
matrix = exact shortest Hamiltonian path) and compare with the natural order.
Only within-tile pairs are counted (the boundary pair between consecutive
tiles along K is not).

Usage:
  PYTHONPATH=fairseq python <this repo>/model/group_reordering/mx_tile_order.py --n-utts 10 --group 8 --thr 1
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

TILE = 8
NATURAL = torch.arange(TILE)


def _build_dp_tables(n=TILE):
    """Held-Karp transition tables, one level per path length."""
    masks_by_pop = [[m for m in range(1 << n) if bin(m).count("1") == p] for p in range(n + 1)]
    idx_of = {}
    for p, ms in enumerate(masks_by_pop):
        for i, m in enumerate(ms):
            idx_of[m] = i
    levels = []
    for p in range(1, n):
        src_state, src_last, nxt, dst_state = [], [], [], []
        for m in masks_by_pop[p]:
            for last in range(n):
                if not m >> last & 1:
                    continue
                for j in range(n):
                    if m >> j & 1:
                        continue
                    src_state.append(idx_of[m] * n + last)
                    src_last.append(last)
                    nxt.append(j)
                    dst_state.append(idx_of[m | 1 << j] * n + j)
        levels.append((torch.tensor(src_state), torch.tensor(src_last) * n + torch.tensor(nxt),
                       torch.tensor(dst_state), len(masks_by_pop[p + 1]) * n))
    return levels


DP_LEVELS = _build_dp_tables()


def min_path_cost(cost: torch.Tensor) -> torch.Tensor:
    """cost: (B, 8, 8) symmetric pairwise costs -> (B,) exact min Hamiltonian-path cost."""
    B = cost.shape[0]
    dev = cost.device
    flat = cost.reshape(B, -1)
    dp = torch.zeros(B, TILE * TILE, dtype=cost.dtype, device=dev)  # level 1: singletons, cost 0
    for src_state, edge, dst_state, n_dst in DP_LEVELS:
        cand = dp[:, src_state.to(dev)] + flat[:, edge.to(dev)]
        dp = torch.full((B, n_dst), 1 << 30, dtype=cost.dtype, device=dev)
        dp.scatter_reduce_(1, dst_state.to(dev)[None].expand(B, -1), cand, reduce="amin")
    return dp.min(1).values


def pair_costs(e_tiles: torch.Tensor, thr: int, valid: torch.Tensor = None) -> torch.Tensor:
    """e_tiles: (B, M, 8) exponents of M accumulators over the 8 steps -> (B, 8, 8)
    cost[b, g1, g2] = #accumulators with |e[g1]-e[g2]| > thr.
    valid (B, M, 8) bool: steps whose block is all-zero contribute no partial sum,
    so a pair touching one costs nothing."""
    out = []
    for s in range(0, e_tiles.shape[0], 65536):
        et = e_tiles[s:s + 65536]
        d = (et[:, :, :, None] - et[:, :, None, :]).abs() > thr
        if valid is not None:
            v = valid[s:s + 65536]
            d &= v[:, :, :, None] & v[:, :, None, :]
        out.append(d.sum(1, dtype=torch.int32))
    return torch.cat(out)


def natural_cost(cost: torch.Tensor) -> torch.Tensor:
    idx = (NATURAL[:-1] * TILE + NATURAL[1:]).to(cost.device)
    return cost.reshape(cost.shape[0], -1)[:, idx].sum(-1)


def order_cost(cost: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """cost (B,8,8), order (B,8) -> (B,) cost of feeding rows in that order."""
    idx = order[:, :-1] * TILE + order[:, 1:]
    return cost.reshape(cost.shape[0], -1).gather(1, idx).sum(-1)


GRE_SNAPSHOTS = (1, 2, 4, 8)


def hamming_table(e_rows: torch.Tensor, valid: torch.Tensor = None) -> torch.Tensor:
    """e_rows: (B, 8 rows=K-steps, 8 cols) SF matrix -> (B, 8, 8) HD table.
    2-bit bins = clamp(e - tile min, 0, 3); HD = number of columns whose bins differ.
    All-zero blocks (valid=False) get bin 0."""
    if valid is None:
        valid = torch.ones_like(e_rows, dtype=torch.bool)
    emin = torch.where(valid, e_rows, torch.full_like(e_rows, float("inf"))).amin((1, 2), keepdim=True)
    bins = torch.where(valid, (e_rows - emin).clamp(0, 3), torch.zeros_like(e_rows))
    return (bins[:, :, None, :] != bins[:, None, :, :]).sum(-1, dtype=torch.int32)


def gre_orders(hd: torch.Tensor, thr: int, max_iter: int = max(GRE_SNAPSHOTS), init_order=None):
    """Group Reordering Engine: anchor filtering + local (sub-sequence reversal)
    reordering, iterated. hd: (B, 8, 8) Hamming table. Returns {n_iter: order (B,8)}
    snapshots after at most n_iter iterations (tiles stop early when there are no
    anchors or an iteration reverses nothing) and the mean iterations actually run."""
    B, dev = hd.shape[0], hd.device
    order = (torch.arange(TILE, device=dev)[None].expand(B, -1).clone()
             if init_order is None else init_order.clone())
    active = torch.ones(B, dtype=torch.bool, device=dev)
    ar = torch.arange(B, device=dev)
    pos = torch.arange(TILE, device=dev)
    snaps, iters = {}, torch.zeros(B, dtype=torch.int32, device=dev)
    for it in range(1, max_iter + 1):
        hd_o = hd[ar[:, None, None], order[:, :, None], order[:, None, :]]  # HD in current order
        left = torch.zeros(B, TILE, dtype=torch.int32, device=dev)
        right = torch.zeros_like(left)
        left[:, 1:] = hd_o[:, pos[1:], pos[:-1]]
        right[:, :-1] = hd_o[:, pos[:-1], pos[1:]]
        anchor = (left + right) > thr
        has_anchor = anchor.any(1)
        # runs of non-anchor positions: SP = last anchor before p + 1, EP = next anchor after p - 1
        sp = torch.zeros_like(order)
        ep = torch.full_like(order, TILE - 1)
        last = torch.full((B,), -1, dtype=torch.long, device=dev)
        for p in range(TILE):
            sp[:, p] = last + 1
            last = torch.where(anchor[:, p], torch.full_like(last, p), last)
        nxt = torch.full((B,), TILE, dtype=torch.long, device=dev)
        for p in reversed(range(TILE)):
            ep[:, p] = nxt - 1
            nxt = torch.where(anchor[:, p], torch.full_like(nxt, p), nxt)
        # reversal arbiter on each run: compare HD at the run boundaries before/after reversal
        o_sp = order.gather(1, sp)
        o_ep = order.gather(1, ep)
        has_l, has_r = sp > 0, ep < TILE - 1
        o_l = order.gather(1, (sp - 1).clamp(min=0))
        o_r = order.gather(1, (ep + 1).clamp(max=TILE - 1))
        hd_b = hd[ar[:, None], o_sp, o_l] * has_l + hd[ar[:, None], o_ep, o_r] * has_r
        hd_a = hd[ar[:, None], o_sp, o_r] * has_r + hd[ar[:, None], o_ep, o_l] * has_l
        rev = (hd_a < hd_b) & ~anchor & active[:, None]
        newpos = torch.where(rev, sp + ep - pos[None], pos[None].expand(B, -1))
        new_order = torch.empty_like(order)
        new_order.scatter_(1, newpos, order)
        changed = (new_order != order).any(1)
        order = new_order
        iters += active.int()
        active = active & has_anchor & changed
        if it in GRE_SNAPSHOTS:
            snaps[it] = order.clone()
        if not active.any():
            for k in GRE_SNAPSHOTS:
                snaps.setdefault(k, order.clone()) if k >= it else None
            break
    for k in GRE_SNAPSHOTS:
        snaps.setdefault(k, order.clone())
    return snaps, iters.float().mean().item()


class TileStats:
    def __init__(self, g, thr, gre_thrs=(), gre_true=False):
        self.g, self.thr = g, thr
        self.gre_thrs = tuple(gre_thrs)
        self.gre_true = gre_true  # diagnostic: drive GRE with the true pair-cost matrix instead of HD
        self.tot = defaultdict(lambda: defaultdict(int))
        self.w_exp_cache = {}
        self.enabled = True

    def record(self, kind, x2d, w2d, name):
        ex = block_exp(x2d, self.g)
        if name not in self.w_exp_cache:
            self.w_exp_cache[name] = block_exp(w2d, self.g)
        ew = self.w_exp_cache[name]
        N, G = ex.shape
        O = ew.shape[0]
        assert G % TILE == 0 and O % TILE == 0, (name, G, O)
        vx, vw = ex > -500, ew > -500  # all-zero blocks carry no partial sum
        self.tot[name]["zero_act_blocks"] += int((~vx).sum())
        self.tot[name]["act_blocks"] += vx.numel()
        ex = torch.where(vx, ex, torch.zeros_like(ex))
        ew = torch.where(vw, ew, torch.zeros_like(ew))
        ot, gt = O // TILE, G // TILE
        nat = opt = 0
        gre, gre_iters = {}, {}
        for s in range(0, N, 32):  # token chunks to bound memory
            exs = ex[s:s + 32]
            n = exs.shape[0]
            e = exs[:, None, :] + ew[None, :, :]  # (n, O, G)
            valid = vx[s:s + 32, None, :] & vw[None, :, :]
            # -> (n, ot, gt, 8 columns, 8 steps)
            tiles = lambda a: a.reshape(n, ot, TILE, gt, TILE).permute(0, 1, 3, 2, 4).reshape(-1, TILE, TILE)
            e_t, v_t = tiles(e), tiles(valid)
            cost = pair_costs(e_t, self.thr, v_t)  # (B, 8 steps, 8 steps)
            nat += natural_cost(cost).sum().item()
            opt += min_path_cost(cost).sum().item()
            if self.gre_thrs:
                hd = cost if self.gre_true else hamming_table(e_t.transpose(1, 2), v_t.transpose(1, 2))
                for gthr in self.gre_thrs:
                    snaps, mean_it = gre_orders(hd, gthr)
                    for k, o in snaps.items():
                        gre[(gthr, k)] = gre.get((gthr, k), 0) + order_cost(cost, o).sum().item()
                    gre_iters[gthr] = gre_iters.get(gthr, 0.0) + mean_it * cost.shape[0]
        B = N * ot * gt
        for k in (kind, "ALL", name):
            t = self.tot[k]
            t["tiles"] += B
            t["pairs"] += B * TILE * (TILE - 1)
            t["natural"] += nat
            t["optimal"] += opt
            for key, v in gre.items():
                t[f"gre{key[0]}_it{key[1]}"] += v
            for gthr, v in gre_iters.items():
                t[f"gre{gthr}_iters"] += v


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
    ap.add_argument("--n-utts", type=int, default=10)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--thr", type=int, default=1, help="count pairs with |Δe| > thr")
    ap.add_argument("--part", choices=list(PARTS), default="asr")
    ap.add_argument("--ckpt", default=CKPT_A)
    ap.add_argument("--per-module", action="store_true")
    ap.add_argument("--gre-thr", type=int, nargs="*", default=[],
                    help="anchor thresholds for the Group Reordering Engine (HD sum of both neighbours > thr)")
    ap.add_argument("--gre-true-cost", action="store_true",
                    help="diagnostic: GRE decisions use the true |Δe|>thr column counts instead of 2-bit HD")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    with open(f"{DATA_BIN}/config_gcmvn.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.BaseLoader)
    gcmvn = np.load(cfg["global_cmvn"]["stats_npz_path"])
    task, model = load_task_and_model(args.ckpt)
    stats = TileStats(args.group, args.thr, args.gre_thr, args.gre_true_cost)
    n = install(model, stats, PARTS[args.part])
    wavs = [l.strip() for l in open(args.wav_list) if l.strip()][: args.n_utts]
    print(f"part={args.part} group={args.group} tile={TILE}Kgroups x {TILE}channels thr={args.thr} "
          f"hooked {n} matmuls, {len(wavs)} utterances", flush=True)

    mt = model.target_unigram_decoder
    eos = task.multitask_tasks["target_unigram"].tgt_dict.eos()
    for i, w in enumerate(wavs):
        feat = features(w, gcmvn)
        enc = encode(model, feat)
        if args.part == "asr":
            model.source_unigram_decoder(enc["encoder_out"][0])
        else:
            stats.enabled = False
            toks = greedy_mt(mt, enc, eos)
            stats.enabled = True
            mt(torch.LongTensor([[eos] + toks]).cuda(), encoder_out=enc)
        print(f"  utt {i} done", flush=True)

    def table(keys, title):
        print(f"\n### {title}: within-tile adjacent-cycle pairs with |Δe| > {args.thr}  (g={args.group}; "
              f"pairs touching an all-zero activation block count as no-shift)")
        print(f"{'':52s} {'tiles':>12s} {'pairs':>14s} | {'natural order':>22s} | {'optimal order/tile':>22s} | reduction | zero act blocks")
        for k in keys:
            t = stats.tot[k]
            p = t["pairs"]
            z = f"{100*t['zero_act_blocks']/t['act_blocks']:5.1f}%" if t.get("act_blocks") else ""
            print(f"{k.replace('encoder.conformer_layers.', 'L'):52s} {t['tiles']:>12,} {p:>14,} | "
                  f"{t['natural']:>13,} {100*t['natural']/p:7.2f}% | "
                  f"{t['optimal']:>13,} {100*t['optimal']/p:7.2f}% | "
                  f"-{100*(1-t['optimal']/max(t['natural'],1)):.1f}% | {z}")

    kinds = [k for k in stats.tot if "." not in k and k != "ALL"] + ["ALL"]
    table(kinds, "by matmul type")
    mods = sorted((k for k in stats.tot if "." in k), key=lambda s: (len(s.split(".")), s))
    if args.per_module:
        table(mods, "per matmul")

    if args.gre_thr:
        def gre_table(keys, title):
            print(f"\n### {title}: GRE (2-bit bins, column-mismatch HD) |Δe| > {args.thr} share by anchor "
                  f"threshold / iteration cap  [natural -> optimal for reference]")
            hdr = " | ".join(f"thr={g}: " + " ".join(f"it{k:<6d}" for k in GRE_SNAPSHOTS) + " avg_it"
                             for g in args.gre_thr)
            print(f"{'':44s} {'natural':>8s} {'optimal':>8s} | {hdr}")
            for k in keys:
                t = stats.tot[k]
                p = t["pairs"]
                cells = []
                for g in args.gre_thr:
                    cells.append(f"thr={g}: " + " ".join(f"{100*t[f'gre{g}_it{it}']/p:6.2f}% " for it in GRE_SNAPSHOTS)
                                 + f"{t[f'gre{g}_iters']/t['tiles']:5.2f}")
                print(f"{k.replace('encoder.conformer_layers.', 'L'):44s} {100*t['natural']/p:7.2f}% "
                      f"{100*t['optimal']/p:7.2f}% | " + " | ".join(cells))
        gre_table(kinds, "by matmul type")
        if args.per_module:
            gre_table(mods, "per matmul")


if __name__ == "__main__":
    main()
