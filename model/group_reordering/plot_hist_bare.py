# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Bare grouped bar charts of the |dSF| distribution before (grey) vs after (blue)
reordering (g=8, 8x8 tiles, Top-2 anchor + shift x16), linear axis, bins >= 5 merged.

Two variants per part:
  *_ticks.png/pdf : bars + tick numbers only (no title, labels, legend, value labels)
  *_naked.png/pdf : bars only (transparent background)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "DejaVu Sans"
TAIL = 5
C_BEFORE, C_AFTER, INK2, GRID = "#c8c7c2", "#5f5e5a", "#52514e", "#e6e5e1"


def shares(part):
    h = torch.load(f"res/quant-mxint4/reorder_hist_top2shift16_g8_{part}.pt")["ALL"]
    n = h["natural"].sum().item()
    out = {}
    for k in ("natural", "shift"):
        p = [100 * h[k][i].item() / n for i in range(TAIL)]
        out[k] = p + [100 * h[k][TAIL:].sum().item() / n]
    return out


def draw(part, variant):
    s = shares(part)
    x = np.arange(TAIL + 1)
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=300)
    ax.bar(x - w / 2 - 0.01, s["natural"], width=w, color=C_BEFORE, linewidth=0, zorder=3)
    ax.bar(x + w / 2 + 0.01, s["shift"], width=w, color=C_AFTER, linewidth=0, zorder=3)
    ymax = max(s["natural"] + s["shift"]) * 1.08
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, TAIL + 0.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    if variant == "ticks":
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(TAIL)] + [f"\u2265{TAIL}"], fontsize=10, color=INK2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.tick_params(length=0, colors=INK2, labelsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.spines["bottom"].set_color(GRID)
        bg = "white"
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["bottom"].set_visible(False)
        bg = "none"
    fig.tight_layout(pad=0.3)
    stem = f"res/quant-mxint4/plots/g8_hist_combined_{part}_{variant}"
    fig.savefig(stem + ".png", facecolor=bg, transparent=(variant == "naked"))
    fig.savefig(stem + ".pdf", facecolor=bg, transparent=(variant == "naked"))
    plt.close(fig)
    print("saved", stem + ".png")


for part in ("asr", "mt"):
    for variant in ("naked",):
        draw(part, variant)
