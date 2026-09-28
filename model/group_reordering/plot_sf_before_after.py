# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Figure: Scaling-Factor Distribution Before and After Group Reordering
(MXINT g=8, 8x8 tiles, Top-2 anchor + shift x16 reordering).

Top: SF (= e_x + e_w, the shared exponent of each partial sum) of one real compute
     block: one token, R quantization-group rows x C CIM columns.
     Left = natural K order, right = after reordering. Every 8x8 tile is fed to the
     array independently, so each tile's 8 rows are permuted on their own (thin grid
     = tile boundaries). Shared colour scale.
Middle (--fp-maps): map of the accumulations that need an FP add
     (|dSF| > T_SF vs the previous row of the same tile), before / after.
Bottom: |PSF - ASF| histogram over all tiles of the 10-utterance run, before (grey)
     vs after (blue), with the INT / FP boundary at T_SF.

--select best : scan every token and every block position of the hooked matmul and
                take the block with the largest FP-count reduction ratio.
--clean       : additionally export the four panels as bare images (no axes / text)
                plus a separate colour bar, for composing the figure elsewhere.

Usage: PYTHONPATH=fairseq python <this repo>/model/group_reordering/plot_sf_before_after.py --part asr --rows 64 --cols 64 --clean
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantization"))
from compare_logit_drift import load_task_and_model, features, encode, greedy_mt, CKPT_A, DATA_BIN  # noqa: E402
from mx_group_exp_gap import block_exp, PARTS  # noqa: E402
from mx_tile_order import TILE, hamming_table  # noqa: E402
from mx_reorder_variants import reversal_engine  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_BEFORE, C_AFTER, C_FP, C_FP_BG = "#9a9892", "#2a78d6", "#e34948", "#f1f0ec"
T_SF = 1
MODULE = {"asr": "encoder.conformer_layers.5.ffn1.w_2", "mt": "target_unigram_decoder.layers.1.fc1"}
TAIL = 5
SF_CMAP = LinearSegmentedColormap.from_list("seq", ["#eaf1fb", "#7fb0e8", "#2a78d6", "#0b2f5e"])
SF_CMAP.set_bad("white")
FP_CMAP = ListedColormap([C_FP_BG, C_FP])


def run_and_capture(model, task, gcmvn, part):
    """Run the first utterance, return (activation matrix (tokens, K), weight (O, K))."""
    cap = {}
    mod = model.get_submodule(MODULE[part])
    mod.register_forward_pre_hook(lambda m, inp: cap.setdefault("x", inp[0].reshape(-1, inp[0].shape[-1]).float()))
    wav = [l.strip() for l in open("datasets/cvss-eval/test100/wav_list.txt") if l.strip()][0]
    feat = features(wav, gcmvn)
    enc = encode(model, feat)
    if part == "mt":
        mt = model.target_unigram_decoder
        eos = task.multitask_tasks["target_unigram"].tgt_dict.eos()
        toks = greedy_mt(mt, enc, eos)
        cap.clear()
        mt(torch.LongTensor([[eos] + toks]).cuda(), encoder_out=enc)
    return cap["x"], mod.weight.float()


def reorder_matrix(e, valid):
    """e, valid: (R, C) SF matrix / validity of one token, R and C multiples of 8.
    Returns (e_after, valid_after, fp_before, fp_after) with each 8x8 tile's rows
    permuted by the Top-2 + shift x16 engine."""
    R, C = e.shape
    rt, ct = R // TILE, C // TILE
    tiles = lambda a: a.reshape(rt, TILE, ct, TILE).permute(0, 2, 3, 1).reshape(-1, TILE, TILE)  # (B, cols, steps)
    e_t, v_t = tiles(e), tiles(valid)
    hd = hamming_table(e_t.transpose(1, 2), v_t.transpose(1, 2))
    order = reversal_engine(hd, "top2-shift", 13, max_iter=16)
    idx = order[:, None, :].expand(-1, TILE, -1)
    untile = lambda a: a.reshape(rt, ct, TILE, TILE).permute(0, 3, 1, 2).reshape(R, C)
    e_after, v_after = untile(e_t.gather(2, idx)), untile(v_t.gather(2, idx))

    def fp_mask(m, v):
        d = torch.zeros_like(m, dtype=torch.bool)
        d[1:] = (m[1:] - m[:-1]).abs() > T_SF
        d[1:] &= v[1:] & v[:-1]
        d[::TILE] = False  # first row of a tile starts a fresh accumulator
        return d

    return e_after, v_after, fp_mask(e, valid), fp_mask(e_after, v_after)


def sf_matrix(ex_t, ew):
    """ex_t: (G,) activation exponents of one token, ew: (O, G) -> SF (G, O), valid (G, O)."""
    valid = (ex_t > -500)[None, :] & (ew > -500)
    e = torch.where(valid, ex_t[None, :] + ew, torch.zeros_like(ew))
    return e.T.contiguous(), valid.T.contiguous()


def select_block(ex, ew, rows, cols, mode, token=None):
    """Return (token, row0, col0). mode 'mid': middle token, offset 0.
    mode 'best': the block (over all tokens and positions) with the largest
    FP-count ratio before/after, among blocks with >= 5% FP share before."""
    n, G = ex.shape
    O = ew.shape[0]
    if mode != "best":
        return (n // 2 if token is None else token), 0, 0
    best = (0.0, n // 2, 0, 0)
    rb, cb = rows // TILE, cols // TILE
    for t in range(n):
        e, valid = sf_matrix(ex[t], ew)
        _, _, fp_b, fp_a = reorder_matrix(e, valid)
        # FP counts per tile, then sum over block windows of rb x cb tiles
        tb = fp_b.reshape(G // TILE, TILE, O // TILE, TILE).sum((1, 3)).float()
        ta = fp_a.reshape(G // TILE, TILE, O // TILE, TILE).sum((1, 3)).float()
        pool = lambda a: torch.nn.functional.avg_pool2d(a[None, None], (rb, cb), stride=1)[0, 0] * rb * cb
        wb, wa = pool(tb), pool(ta)
        n_acc = rows * cols - cols * rb
        ratio = torch.where(wb >= 0.05 * n_acc, wb / wa.clamp(min=1), torch.zeros_like(wb))
        r, i = ratio.flatten().max(0)
        if r.item() > best[0]:
            best = (r.item(), t, (i // ratio.shape[1]).item() * TILE, (i % ratio.shape[1]).item() * TILE)
    print(f"best block: ratio {best[0]:.2f}, token {best[1]}, rows {best[2]}.., cols {best[3]}..")
    return best[1], best[2], best[3]


def capture_block(model, task, gcmvn, part, rows, cols, mode, token=None):
    x, w = run_and_capture(model, task, gcmvn, part)
    ex, ew = block_exp(x, 8), block_exp(w, 8)
    n, G = ex.shape
    O = ew.shape[0]
    rows, cols = min(rows, G), min(cols, O)
    t, r0, c0 = select_block(ex, ew, rows, cols, mode, token)
    e, valid = sf_matrix(ex[t], ew)
    e, valid = e[r0:r0 + rows, c0:c0 + cols].contiguous(), valid[r0:r0 + rows, c0:c0 + cols].contiguous()
    e_after, v_after, fp_b, fp_a = reorder_matrix(e, valid)
    return {"before": e.cpu().numpy(), "after": e_after.cpu().numpy(),
            "fp_before": fp_b.cpu().numpy(), "fp_after": fp_a.cpu().numpy(),
            "valid_before": valid.cpu().numpy(), "valid_after": v_after.cpu().numpy(),
            "token": t, "n_tokens": n, "row0": r0, "col0": c0, "module": MODULE[part]}


def draw_grid(ax, rows, cols, lw=0.35):
    for k in range(TILE, rows, TILE):
        ax.axhline(k - 0.5, color="white", linewidth=lw, alpha=0.8)
    for k in range(TILE, cols, TILE):
        ax.axvline(k - 0.5, color="white", linewidth=lw, alpha=0.8)


def export_clean(blk, vmin, vmax, stem, grid=True):
    """Bare panels (no axes / text) + a separate colour bar."""
    rows, cols = blk["before"].shape
    for key in ("before", "after"):
        for kind in ("sf", "fp"):
            fig = plt.figure(figsize=(4, 4 * rows / cols), dpi=300)
            ax = fig.add_axes([0, 0, 1, 1])
            if kind == "sf":
                mat = np.ma.masked_where(~blk[f"valid_{key}"], blk[key].astype(float))
                ax.imshow(mat, cmap=SF_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
            else:
                ax.imshow(blk[f"fp_{key}"].astype(int), cmap=FP_CMAP, vmin=0, vmax=1,
                          interpolation="nearest", aspect="auto")
            if grid:
                draw_grid(ax, rows, cols, lw=0.6)
            ax.axis("off")
            out = f"{stem}_{kind}_{key}"
            fig.savefig(out + ".png", dpi=300)
            fig.savefig(out + ".pdf")
            plt.close(fig)
            print("saved", out + ".png")
    fig = plt.figure(figsize=(0.9, 3.2), dpi=300)
    cax = fig.add_axes([0.25, 0.05, 0.22, 0.9])
    sm = plt.cm.ScalarMappable(cmap=SF_CMAP, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, cax=cax, ticks=list(range(vmin, vmax + 1, max(1, (vmax - vmin) // 7))))
    cb.outline.set_visible(False)
    cax.tick_params(length=0, labelsize=8, colors=INK2)
    fig.savefig(f"{stem}_colorbar.png", dpi=300, transparent=True)
    fig.savefig(f"{stem}_colorbar.pdf", transparent=True)
    plt.close(fig)
    print("saved", f"{stem}_colorbar.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=list(PARTS), default="asr")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--token", type=int, default=None)
    ap.add_argument("--select", choices=["mid", "best"], default="mid")
    ap.add_argument("--fp-maps", action="store_true", help="add a row with the FP-add maps")
    ap.add_argument("--clean", action="store_true", help="also export bare panels + colour bar")
    ap.add_argument("--module", default=None, help="override the hooked matmul (module name)")
    args = ap.parse_args()
    if args.module:
        MODULE[args.part] = args.module
    torch.set_grad_enabled(False)
    with open(f"{DATA_BIN}/config_gcmvn.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.BaseLoader)
    gcmvn = np.load(cfg["global_cmvn"]["stats_npz_path"])
    task, model = load_task_and_model(CKPT_A)
    blk = capture_block(model, task, gcmvn, args.part, args.rows, args.cols, args.select, args.token)
    rows, cols = blk["before"].shape
    n_fp = {k: int(blk[f"fp_{k}"].sum()) for k in ("before", "after")}
    n_acc = int(blk["valid_before"].sum()) - cols * (rows // TILE)
    vb = blk["before"][blk["valid_before"]]
    vmin, vmax = int(vb.min()), int(vb.max())
    print(f"{blk['module']}, token {blk['token']}/{blk['n_tokens']}, block {rows}x{cols} at "
          f"(row {blk['row0']}, col {blk['col0']}): FP adds {n_fp['before']} -> {n_fp['after']} "
          f"of {n_acc} accumulations ({100*n_fp['before']/n_acc:.1f}% -> {100*n_fp['after']/n_acc:.1f}%, "
          f"x{n_fp['before']/max(n_fp['after'],1):.2f}); SF range {vmin}..{vmax}")

    tag = f"{rows}x{cols}" + ("_best" if args.select == "best" else "")
    stem = f"res/quant-mxint4/plots/sf_before_after_g8_top2shift16_{args.part}_{tag}"
    if args.clean:
        export_clean(blk, vmin, vmax, stem + "_clean")

    h = torch.load(f"res/quant-mxint4/reorder_hist_top2shift16_g8_{args.part}.pt")["ALL"]
    n = h["natural"].sum().item()

    def share(k):
        p = [100 * h[k][i].item() / n for i in range(TAIL)]
        return p + [100 * h[k][TAIL:].sum().item() / n]

    before, after = share("natural"), share("shift")
    fp_before, fp_after = sum(before[T_SF + 1:]), sum(after[T_SF + 1:])

    nrow = 3 if args.fp_maps else 2
    aspect = rows / cols
    fig_h = 3.6 * aspect * (2 if args.fp_maps else 1) + 3.6
    fig = plt.figure(figsize=(7.4, fig_h), dpi=200, facecolor=SURFACE)
    ratios = [aspect, aspect, 0.72] if args.fp_maps else [aspect, 0.72]
    gs = fig.add_gridspec(nrow, 3, width_ratios=[1, 0.13, 1], height_ratios=ratios,
                          hspace=0.32, wspace=0.10, left=0.075, right=0.975,
                          top=0.90 if args.fp_maps else 0.875, bottom=0.07 if args.fp_maps else 0.085)

    panels = [("before", "Original order"), ("after", "After reordering (Top-2 anchor + shift, 16 it.)")]
    im = None
    for col, (key, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * col])
        mat = np.ma.masked_where(~blk[f"valid_{key}"], blk[key].astype(float))
        im = ax.imshow(mat, cmap=SF_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        draw_grid(ax, rows, cols)
        ax.set_xticks([0, cols // 2, cols - 1])
        ax.set_yticks([0, rows // 2, rows - 1])
        ax.tick_params(length=0, colors=INK2, labelsize=7)
        ax.set_xlabel("CIM column", fontsize=8, color=INK2)
        if col == 0:
            ax.set_ylabel("quantization-group row (feeding order)", fontsize=8, color=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{title}\n{n_fp[key]:,} FP accumulations in this block", fontsize=8.5, color=INK, pad=5)
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_mid.axis("off")
    ax_mid.text(0.5, 1.0, "\u2192", ha="center", va="top", fontsize=18, color=INK2, transform=ax_mid.transAxes)
    cax = ax_mid.inset_axes([0.36, 0.12, 0.22, 0.58])
    cb = fig.colorbar(im, cax=cax, ticks=list(range(vmin, vmax + 1, max(1, (vmax - vmin) // 6))))
    cb.outline.set_visible(False)
    cax.tick_params(length=0, labelsize=6.5, colors=INK2)
    cax.set_title("SF", fontsize=7, color=INK2, pad=3)
    cax.set_xlabel("shared\nexponent", fontsize=6, color=INK2, labelpad=3)

    if args.fp_maps:
        for col, (key, title) in enumerate(panels):
            ax = fig.add_subplot(gs[1, 2 * col])
            ax.imshow(blk[f"fp_{key}"].astype(int), cmap=FP_CMAP, vmin=0, vmax=1, interpolation="nearest")
            draw_grid(ax, rows, cols)
            ax.set_xticks([0, cols // 2, cols - 1])
            ax.set_yticks([0, rows // 2, rows - 1])
            ax.tick_params(length=0, colors=INK2, labelsize=7)
            ax.set_xlabel("CIM column", fontsize=8, color=INK2)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"FP accumulations (red), {'before' if key == 'before' else 'after'}: "
                         f"{100 * n_fp[key] / n_acc:.1f}% of this block", fontsize=8.5, color=INK, pad=5)

    short = blk["module"].replace("encoder.conformer_layers.", "enc L").replace("target_unigram_decoder.layers.", "MT dec L")
    fig.text(0.5, 0.965, "Scaling-Factor Distribution Before and After Group Reordering",
             ha="center", fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.5, 0.94, f"MXINT g=8, 8x8 tiles.  Top: {rows}x{cols} block of {short}, one token.  "
             f"Bottom: all {n/1e8:.1f}e8 accumulations of 10 CVSS utterances",
             ha="center", fontsize=7.5, color=INK2)

    ax = fig.add_subplot(gs[nrow - 1, :])
    ax.set_facecolor(SURFACE)
    x = np.arange(TAIL + 1)
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, before, width=w, color=C_BEFORE, linewidth=0, zorder=3,
                label=f"before reordering ({fp_before:.1f}% FP)")
    b2 = ax.bar(x + w / 2 + 0.01, after, width=w, color=C_AFTER, linewidth=0, zorder=3,
                label=f"after reordering ({fp_after:.1f}% FP)")
    for bars, vals in ((b1, before), (b2, after)):
        for bar, p in zip(bars, vals):
            ax.annotate(f"{p:.1f}%" if p >= 1 else f"{p:.2f}%", (bar.get_x() + bar.get_width() / 2, p),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=INK)
    ymax = max(before + after) * 1.25
    ax.set_ylim(0, ymax)
    ax.axvline(T_SF + 0.5, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=4)
    ax.text(T_SF + 0.5 - 0.08, ymax * 0.96, "INT accumulation", ha="right", va="top", fontsize=8.5, color=INK2)
    ax.text(T_SF + 0.5 + 0.08, ymax * 0.96, "FP accumulation", ha="left", va="top", fontsize=8.5, color=INK2)
    ax.text(T_SF + 0.5, ymax * 0.80, f"$T_{{SF}}$ = {T_SF}", ha="center", va="top", fontsize=8.5, color=INK2,
            bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(TAIL)] + [f"\u2265{TAIL}"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=INK2, labelsize=8.5)
    ax.set_xlabel("|PSF \u2212 ASF|  (scaling-factor gap between partial sum and accumulator)",
                  fontsize=8.5, color=INK2)
    ax.set_ylabel("share of accumulations", fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)
    out = stem + ("_fpmaps" if args.fp_maps else "") + ".png"
    fig.savefig(out, facecolor=SURFACE)
    fig.savefig(out.replace(".png", ".pdf"), facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
