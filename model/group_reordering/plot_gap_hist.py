# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Histograms of |Δe| (x = exact value 0,1,2,...):
  g=4 adjacent-group gap, ASR / MT          (group_exp_gap_g4_fp_{part}.hist.pt)
  g=8 8x8-tile feeding order, natural and Top-2+shift x16 as separate charts, ASR / MT
                                            (reorder_hist_top2shift16_g8_{part}.pt)
Default: linear axis, bins >= TAIL merged into one bar.  --log: log axis, all bins."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

LOG = "--log" in sys.argv
TAIL = 5  # linear mode: values >= TAIL merged into one bar
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C1, C2 = "#2a78d6", "#eb6834"
TITLES = {"asr": "ASR path (encoder Conformer + CTC head)", "mt": "MT path (target_unigram_decoder)"}


def fmt_log(p):
    if p >= 10:
        return f"{p:.1f}%"
    if p >= 1:
        return f"{p:.2f}%"
    if p >= 0.01:
        return f"{p:.3f}%"
    if p >= 0.0001:
        return f"{p:.4f}%"
    if p >= 0.00001:
        return f"{p:.5f}%"
    return f"{p:.6f}%" if p >= 0.000001 else f"{p:.0e}%"


def fmt_lin(p):
    return f"{p:.1f}%" if p >= 1 else f"{p:.2f}%"


def prepare(pct):
    """(values to plot, tick labels) for the current mode."""
    if LOG:
        return pct, [str(i) for i in range(len(pct))]
    merged = pct[:TAIL] + [sum(pct[TAIL:])]
    return merged, [str(i) for i in range(TAIL)] + [f"≥{TAIL}"]


def axes_setup(ax, xticks, ymax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    if LOG:
        ax.set_yscale("log")
        ax.set_ylim(3e-8, 400)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_log(v) if v >= 3e-8 else ""))
    else:
        ax.set_ylim(0, ymax * 1.18)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xticks(list(range(len(xticks))))
    ax.set_xticklabels(xticks)
    ax.set_xlim(-0.7, len(xticks) - 0.3)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", colors=INK2, length=0, labelsize=9.5)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel + (" (log scale)" if LOG else ""), color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)


def label_bars(ax, bars, vals):
    for bar, p in zip(bars, vals):
        if p <= 0:
            continue
        xc = bar.get_x() + bar.get_width() / 2
        if LOG:
            ax.annotate(fmt_log(p), (xc, p), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=6.5, color=INK, rotation=90)
        else:
            ax.annotate(fmt_lin(p), (xc, p), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9, color=INK)


def single_chart(pct, ticks, ymax, color, xlabel, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(10, 5.2) if LOG else (7.2, 4.6), dpi=160, facecolor=SURFACE)
    axes_setup(ax, ticks, ymax, xlabel, ylabel, title)
    bars = ax.bar(range(len(pct)), pct, width=0.62, color=color, linewidth=0, zorder=3)
    label_bars(ax, bars, pct)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("saved", out)


def plot_g4(part):
    hs = torch.load(f"res/quant-mxint4/group_exp_gap_g4_fp_{part}.hist.pt")
    h = sum(hs.values())
    n = h.sum().item()
    xmax = torch.nonzero(h).flatten()[-1].item()
    pct, ticks = prepare([100 * h[i].item() / n for i in range(xmax + 1)])
    single_chart(pct, ticks, max(pct), C1,
                 "Combined exponent gap between adjacent groups |Δe| (= alignment shift before accumulation)", "Fraction of adjacent group pairs",
                 f"{TITLES[part]}, MX group size = 4\nDistribution of |Δe| over {n/1e9:.1f}B adjacent group pairs, 10 CVSS test utterances",
                 f"res/quant-mxint4/plots/g4_hist_{part}{'_log' if LOG else ''}.png")


def plot_reorder(part):
    """One chart per feeding order (natural / Top-2+shift x16); shared y range."""
    h = torch.load(f"res/quant-mxint4/reorder_hist_top2shift16_g8_{part}.pt")["ALL"]
    n = h["natural"].sum().item()
    xmax = max(torch.nonzero(h[k]).flatten()[-1].item() for k in h)
    series = {k: prepare([100 * h[k][i].item() / n for i in range(xmax + 1)]) for k in ("natural", "shift")}
    ymax = max(max(v) for v, _ in series.values())
    names = {"natural": ("Natural order (rows fed in K-index order)", C1, "natural"),
             "shift": ("Reordered (Top-2 anchors + per-round shift, 16 rounds)", C2, "top2shift16")}
    for key, (pct, ticks) in series.items():
        label, color, tag = names[key]
        single_chart(pct, ticks, ymax, color,
                     "Combined exponent gap between consecutive cycles in a column |Δe| (= alignment shift before accumulation)", "Fraction of consecutive-cycle pairs",
                     f"{TITLES[part]}, MX group size = 8, 8×8 tile\n{label}\n"
                     f"Distribution of |Δe| over {n/1e9:.2f}B consecutive pairs, 10 CVSS test utterances",
                     f"res/quant-mxint4/plots/g8_hist_{tag}_{part}{'_log' if LOG else ''}.png")


for part in ("asr", "mt"):
    plot_g4(part)
    plot_reorder(part)
