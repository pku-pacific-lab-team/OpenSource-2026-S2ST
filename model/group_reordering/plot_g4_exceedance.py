# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Bar charts of P(|Δe| > T) at g=4 for the ASR and MT paths (exact counts from
res/quant-mxint4/group_exp_gap_g4_fp_{asr,mt}.txt)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, INK2, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
T = [1, 2, 3, 4, 6, 8, 12, 16]
DATA = {
    "asr": ("ASR path (encoder Conformer + CTC head)", 9_306_630_816,
            [3_035_766_249, 1_177_910_844, 455_899_299, 186_924_710, 35_568_436, 8_479_420, 596_922, 29_595]),
    "mt": ("MT path (target_unigram_decoder)", 703_091_360,
           [188_308_051, 50_945_639, 11_765_379, 2_984_025, 282_539, 42_338, 1_494, 142]),
}


def fmt(p):
    if p >= 10:
        return f"{p:.1f}%"
    if p >= 0.01:
        return f"{p:.2f}%" if p >= 1 else f"{p:.3f}%"
    return f"{p:.4f}%" if p >= 0.0001 else f"{p:.5f}%"


def plot(key):
    title, total, counts = DATA[key]
    pct = [100 * c / total for c in counts]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = range(len(T))
    ax.bar(x, pct, width=0.55, color=BAR, linewidth=0, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(5e-6, 200)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v) if v >= 1e-5 else ""))
    ax.set_xticks(list(x))
    ax.set_xticklabels([f">{t}" for t in T])
    for i, p in enumerate(pct):
        ax.annotate(fmt(p), (i, p), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", colors=INK2, length=0, labelsize=9.5)
    ax.set_xlabel("Threshold T (combined exponent gap between adjacent groups, |Δe| > T)", color=INK2, fontsize=10)
    ax.set_ylabel("Fraction of adjacent group pairs (log scale)", color=INK2, fontsize=10)
    ax.set_title(f"{title}, MX group size = 4\nP(|Δe| > T) over {total/1e9:.1f}B adjacent group pairs, 10 CVSS test utterances",
                 color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    out = f"res/quant-mxint4/plots/g4_exceedance_{key}.png"
    fig.savefig(out, facecolor=SURFACE)
    print("saved", out)


for k in DATA:
    plot(k)
