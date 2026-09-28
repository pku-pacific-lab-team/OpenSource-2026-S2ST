# -*- coding: utf-8 -*-
# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Two-panel small multiples: relative quality degradation vs FP baseline.

Both metrics are converted to 'relative degradation (%)' so higher = worse in
both panels, sharing one y scale; the 5% acceptance line becomes comparable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
C = {"W4": "#2a78d6", "W4A4": "#1baf7a", "W4A4, FP CTC": "#eda100"}

G = [32, 16, 8, 4]
X = range(len(G))
FP_WER, FP_BLEU = 29.95, 27.171

wer = {
    "W4": [31.88, 31.35, 30.92, 30.45],
    "W4A4": [39.94, 36.43, 34.80, 33.19],
    "W4A4, FP CTC": [38.77, 35.69, 34.01, 32.64],
}
bleu = {
    "W4": [25.693, 26.319, 26.433, 26.341],
    "W4A4": [24.388, 24.861, 25.339, 25.239],
    "W4A4, FP CTC": [23.744, 24.954, 25.606, 25.334],
}
asr = {k: [100 * (w - FP_WER) / FP_WER for w in v] for k, v in wer.items()}
s2tt = {k: [100 * (FP_BLEU - b) / FP_BLEU for b in v] for k, v in bleu.items()}

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, dpi=200)
fig.patch.set_facecolor(SURFACE)

for ax, data, title in [
    (axes[0], asr, "ASR: relative WER increase"),
    (axes[1], s2tt, "S2TT: relative BLEU decrease"),
]:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)

    ax.axhline(5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))

    for name in C:
        ax.plot(X, data[name], color=C[name], linewidth=2,
                marker="o", markersize=6, markerfacecolor=C[name],
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=name, clip_on=False, zorder=3)

    # direct labels at the right line ends, repelled top-down to a minimum gap;
    # a hairline leader connects a label to its line end when it had to move
    ends = sorted(((data[n][-1], n) for n in C), reverse=True)
    prev = None
    for y0, name in ends:
        y = y0 if prev is None else min(y0, prev - 2.0)
        prev = y
        arrow = dict(arrowstyle="-", color=BASE, lw=0.8,
                     shrinkA=2, shrinkB=1) if abs(y - y0) > 0.5 else None
        ax.annotate(name, xy=(X[-1] + 0.06, y0), xytext=(X[-1] + 0.30, y),
                    va="center", ha="left", fontsize=8.5, color=INK2,
                    arrowprops=arrow, annotation_clip=False)

    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_xticks(list(X), [str(g) for g in G])
    ax.set_xlabel("MX group size g (finer groups to the right)", fontsize=9, color=INK2)
    ax.set_xlim(-0.25, len(G) + 0.05)

axes[0].set_ylabel("Relative degradation (%, lower is better)", fontsize=9.5, color=INK2)
axes[0].set_ylim(0, 36)
axes[0].annotate("5% acceptance line", (1.55, 5.3), va="bottom", ha="left",
                 fontsize=8, color=MUTED)

fig.suptitle("MXINT4 quality degradation vs. FP (CVSS-C fr-en, 1000 utterances, chunk = 320 ms)",
             fontsize=12.5, color=INK, y=0.99)
axes[1].legend(loc="upper right", fontsize=8.5, frameon=False,
               labelcolor=INK2, handlelength=1.6)

fig.subplots_adjust(left=0.07, right=0.93, top=0.85, bottom=0.13, wspace=0.14)
fig.savefig("quant_degradation.png", facecolor=SURFACE)
print("saved quant_degradation.png")
