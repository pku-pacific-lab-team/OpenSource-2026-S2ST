# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Grouped bar charts: P(|Δe| > T) under natural order vs Top-2 + shift x16 reordering
(g=8, 8x8 tiles, within-tile adjacent-cycle pairs), one chart per part."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_NAT, C_SHIFT = "#2a78d6", "#eb6834"  # categorical slots 1, 2
T = [1, 2, 3, 4, 6, 8, 12, 16]
TITLES = {"asr": "ASR path (encoder Conformer + CTC head)", "mt": "MT path (target_unigram_decoder)"}


def fmt(p):
    if p >= 10:
        return f"{p:.1f}%"
    if p >= 1:
        return f"{p:.2f}%"
    if p >= 0.01:
        return f"{p:.3f}%"
    return f"{p:.4f}%" if p >= 0.0001 else (f"{p:.5f}%" if p >= 0.00001 else f"{p:.6f}%")


def plot(part):
    h = torch.load(f"res/quant-mxint4/reorder_hist_top2shift16_g8_{part}.pt")["ALL"]
    n = h["natural"].sum().item()
    series = {}
    for key in ("natural", "shift"):
        series[key] = [100 * h[key][t + 1:].sum().item() / n for t in T]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = torch.arange(len(T)).float()
    w = 0.34
    b1 = ax.bar(x - w / 2 - 0.01, series["natural"], width=w, color=C_NAT, linewidth=0, zorder=3,
                label="Natural order (rows fed in K-index order)")
    b2 = ax.bar(x + w / 2 + 0.01, series["shift"], width=w, color=C_SHIFT, linewidth=0, zorder=3,
                label="Top-2 anchors + per-round shift, 16 rounds")
    floor = 8e-7
    ax.set_yscale("log")
    ax.set_ylim(floor, 300)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v) if v >= 1e-6 else ""))
    ax.set_xticks(x.tolist())
    ax.set_xticklabels([f">{t}" for t in T])
    for bars, vals in ((b1, series["natural"]), (b2, series["shift"])):
        for bar, p in zip(bars, vals):
            if p <= 0:
                ax.annotate("0", (bar.get_x() + bar.get_width() / 2, floor * 1.3), ha="center", va="bottom",
                            fontsize=7.5, color=INK2)
                continue
            ax.annotate(fmt(p), (bar.get_x() + bar.get_width() / 2, p), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", colors=INK2, length=0, labelsize=9.5)
    ax.set_xlabel("Threshold T (combined exponent gap between consecutive cycles in a column, |Δe| > T)", color=INK2, fontsize=10)
    ax.set_ylabel("Fraction of consecutive-cycle pairs (log scale)", color=INK2, fontsize=10)
    ax.set_title(f"{TITLES[part]}, MX group size = 8, 8×8 tile\n"
                 f"P(|Δe| > T) before and after row reordering, {n/1e9:.2f}B consecutive pairs, 10 CVSS test utterances",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
    fig.tight_layout()
    out = f"res/quant-mxint4/plots/reorder_top2shift16_exceedance_{part}.png"
    fig.savefig(out, facecolor=SURFACE)
    print("saved", out)


for part in (sys.argv[1:] or ["asr", "mt"]):
    plot(part)
