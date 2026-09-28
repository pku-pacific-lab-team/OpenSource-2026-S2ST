# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Per-input-frame compute trace of one utterance, w/o vs w/ speculative speech
generation (SSG), from the real agent traces.

Frame = one 320 ms input chunk.  For every frame both configurations pay
encoder + ASR head; frames on which the ASR emitted a new word also run the MT
search; frames on which text was committed run TTS (T2U + unit CTC + vocoder).
With SSG, MT includes the extra search steps (min 5) and TTS splits into
"committed tokens not pre-synthesised" and "speculative pre-synthesis".

Usage:
  python plot_trace.py --list                    # rank candidate utterances
  python plot_trace.py --utt 17 --out fig.png    # draw one
"""
import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("STREAMSPEECH_ROOT", os.getcwd())
RUN_WO = os.path.join(ROOT, "res/conf-trigger/test100/stepwise-prod-0.5-tts")
RUN_W = os.path.join(ROOT, "res/conf-trigger/test100/stepwise-prod-0.5-earlytok-0.1")
CHUNK_MS = 320
SSG_TAG = "τ = 0.1"


def interp(table, x):
    xs = np.array(table["len"] if "len" in table else table["fbank_len"])
    return float(np.interp(x, xs, np.array(table["flops"])))


def total_at(table, n):
    return 0.0 if n <= 0 else interp(table, n)


def marginal(table, pos):
    return total_at(table, pos + 1) - total_at(table, pos)


def read_groups(run):
    groups, cur, cid = [], [], None
    for line in open(os.path.join(run, "trace.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["sent"] != cid:
            if cur:
                groups.append(cur)
            cur, cid = [], r["sent"]
        cur.append(r)
    if cur:
        groups.append(cur)
    inst = [json.loads(l) for l in open(os.path.join(run, "instances.log"), encoding="utf-8")]
    return groups, inst


def frame_of(src_ms):
    return int(np.ceil(src_ms / CHUNK_MS))


def per_frame_costs(rows, dur_ms, cal, with_ssg):
    """-> dict frame_id -> {asr, mt, tts, spec}, plus annotations per frame."""
    m, v = cal["mt_dec_token"], cal["vocoder"]
    n_frames = frame_of(dur_ms)
    cost = {f: {"asr": 0.0, "mt": 0.0, "tts": 0.0, "spec": 0.0} for f in range(1, n_frames + 1)}
    notes = {f: {"commit": None, "hits": 0, "spec_toks": [], "flush": False} for f in cost}
    for f in cost:
        ms = min(f * CHUNK_MS, dur_ms)
        cost[f]["asr"] = interp(cal["encoder_marginal"], ms / 10.0) + cal["asr_head_per_enc_frame"] * 8
    for r in rows:
        if "steps" not in r and "shapes" not in r:
            continue
        f = frame_of(r["src_ms"])
        s = r.get("shapes")
        if s:
            fw, P, T, steps = s["dec_token_fwds"], s["prefix_len"], s["enc_frames"], s["beam_steps"]
            extra = s.get("dec_token_fwds_extra", 0) if with_ssg else 0
        else:  # READ row of the no-SSG run: search ran len(steps) steps
            steps = len(r["steps"])
            fw, extra = 1 + 5 * max(steps - 1, 0), 0
            P, T = len(r["prefix"].split()), round(r["src_ms"] / 40)
        per_tok = m["a"] + m["b_prefix"] * (P + max(steps - 1, 0) / 2.0) + m["c_enc"] * T
        cost[f]["mt"] += (fw + extra) * per_tok
        if r.get("action") == "WRITE" and s:
            n = s["new_tokens"]
            tts_pos = s.get("tts_positions", list(range(n))) if with_ssg else list(range(n))
            tts = sum(marginal(cal["t2u_total"], P + i) + marginal(cal["unit_ctc_total"], P + i) for i in tts_pos)
            if with_ssg and "tts_units" in s:
                u, e = s["tts_units"], s["tts_units_expanded"]
            else:
                u, e = s.get("new_units", 0), s.get("new_units_expanded", 0)
            if u:
                tts += v["a"] + v["b_units"] * u + v["c_expanded"] * e
            cost[f]["tts"] += tts
            notes[f]["commit"] = r.get("committed_text", "")
            notes[f]["hits"] = s.get("spec_hits", 0) if with_ssg else 0
            notes[f]["flush"] = bool(s.get("final_flush", False))
        if with_ssg:
            for pos, u, e in r.get("spec_new", []):
                c = marginal(cal["t2u_total"], pos) + marginal(cal["unit_ctc_total"], pos)
                if u:
                    c += v["a"] + v["b_units"] * u + v["c_expanded"] * e
                cost[f]["spec"] += c
            notes[f]["spec_toks"] += [e["tok"].replace("▁", "") for e in r.get("early", []) if e.get("new")]
    return cost, notes


def draw_timeline(frames, c0, c1, n1, to_ms, inst_i, i, a, C_ASR, C_MT, C_TTS, INK, INK2, MUTED, SURF):
    """Hardware-occupancy view: real time left->right, one 320 ms cell per input
    frame, one lane per configuration.  Inside a cell the busy segments (ASR ->
    MT -> TTS -> speculative TTS) start at the frame's arrival and are drawn at
    `zoom` x real time so that they are visible; the white remainder is idle."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    if a.trim_lead:
        while len(frames) > 1 and all(c[frames[0]][k] == 0 for c in (c0, c1) for k in ("mt", "tts", "spec")):
            frames = frames[1:]
    if a.gray:
        C_ASR, C_MT, C_TTS = "#3a3a3a", "#8a8a8a", "#c4c4c4"
    f_first = frames[0]
    n = len(frames)
    keys = [("asr", C_ASR, None), ("mt", C_MT, None), ("tts", C_TTS, None), ("spec", C_TTS, "////")]
    busy = lambda c, f, ks: sum(to_ms(c[f][k]) for k in ks)  # noqa: E731
    max_busy = max(max(busy(c0, f, ("asr", "mt", "tts")) for f in frames),
                   max(busy(c1, f, ("asr", "mt", "tts", "spec")) for f in frames))
    zoom = a.zoom or (0.88 * CHUNK_MS / max_busy)

    fig, ax = plt.subplots(figsize=(12.5, 1.8 if a.bare else (2.6 if a.no_annot else 3.6)), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SURF)
    H = 0.5
    lanes = [("w/o SSG", c0, 1.0), ("w/ SSG", c1, 0.0)]
    x_end = n * CHUNK_MS
    for name, cost, yc in lanes:
        for f in frames:
            x = (f - f_first) * CHUNK_MS
            for k, col, hatch in keys:
                if k == "spec" and cost is c0:
                    continue
                wms = to_ms(cost[f][k]) * zoom
                if wms <= 0:
                    continue
                ax.add_patch(Rectangle((x, yc - H / 2), wms, H, facecolor=col, edgecolor="white",
                                       linewidth=0.5, hatch=hatch, zorder=3))
                x += wms
            over = x - (f - f_first + 1) * CHUNK_MS
            if over > 0:  # the frame's work does not fit into its 320 ms slot
                x_end = max(x_end, x)
                xb = (f - f_first + 1) * CHUNK_MS
                ax.plot([xb, xb], [yc - H / 2 - 0.05, yc + H / 2 + 0.05], color=("#0b0b0b" if a.gray else "#d03b3b"),
                        lw=1.0, zorder=4)
                if not a.no_annot:
                    ax.text(x + 8, yc, f"overruns frame by {over:.0f} ms", ha="left", va="center", fontsize=6.5,
                            color=("#0b0b0b" if a.gray else "#d03b3b"))
    if a.arrows:
        for f in frames:
            x0 = (f - f_first) * CHUNK_MS
            end_wo = x0 + busy(c0, f, ("asr", "mt", "tts")) * zoom
            end_w = x0 + busy(c1, f, ("asr", "mt", "tts")) * zoom
            saved = (end_wo - end_w) / zoom
            if saved < a.arrow_min:
                continue
            ax.annotate("", xy=(end_w, 0.5), xytext=(end_wo, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, shrinkA=0, shrinkB=0,
                                        mutation_scale=7), zorder=5)
            ax.plot([end_wo, end_wo], [0.5, 1.0 - H / 2], color=INK, lw=0.6, ls=(0, (2, 2)), zorder=5)
            if not a.bare:
                ax.text((end_wo + end_w) / 2, 0.56, f"−{saved:.0f} ms", ha="center", va="bottom", fontsize=6,
                        color=INK)
    for f in frames:
        ax.axvline((f - f_first) * CHUNK_MS, color="#dedcd6", lw=0.6, zorder=1)
        if not a.bare:
            ax.text((f - f_first + 0.5) * CHUNK_MS, 1.0 + H / 2 + 0.08, f"frame {f}" if f == f_first else str(f),
                    ha="center", va="bottom", fontsize=7, color=INK2)
    ax.axvline(n * CHUNK_MS, color="#dedcd6", lw=0.6, zorder=1)

    # commit labels under the w/ SSG lane (commits are identical in both lanes)
    for f in frames:
        if a.no_annot or not n1[f]["commit"]:
            continue
        label = n1[f]["commit"]
        last = f == frames[-1]
        if n1[f]["flush"]:
            label = "flush: " + label
        if len(label) > 60:
            label = label[:58] + "…"
        xa = (f - f_first + 1) * CHUNK_MS - 6 if last else (f - f_first) * CHUNK_MS + 6
        # the flush label is long: give it its own row so it cannot collide with earlier commits
        ax.text(xa, -H / 2 - (0.62 if n1[f]["flush"] else 0.08), label, ha=("right" if last else "left"),
                va="top", fontsize=6.5, color=INK2)
        if n1[f]["hits"]:
            ax.text(xa, -H / 2 - 0.34, f"−{n1[f]['hits']} tok (SSG)", ha=("right" if last else "left"),
                    va="top", fontsize=6, color=C_TTS)
    # speculative tokens above the w/ SSG lane
    for f in frames:
        if n1[f]["spec_toks"] and not a.no_annot:
            s = " ".join(n1[f]["spec_toks"])
            nxt = f + 1 in n1 and bool(n1[f + 1]["spec_toks"])
            lim = 15 if nxt else 30  # keep the label inside its own cell if the next cell has one too
            if len(s) > lim:
                s = s[:lim - 1] + "…"
            ax.text((f - f_first) * CHUNK_MS + 6, H / 2 + 0.05, "▸ " + s, ha="left", va="bottom", fontsize=5.8,
                    color=(INK2 if a.gray else C_TTS), style="italic")

    ax.set_xlim(0, x_end + (0.55 * CHUNK_MS if (x_end > n * CHUNK_MS and not a.no_annot) else 0))
    ax.set_ylim((-0.45 if a.no_annot else -1.15), 1.95)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["w/o SSG", "w/ SSG\n(τ = 0.1)"], fontsize=8, color=INK)
    ax.set_xticks([(f - f_first) * CHUNK_MS for f in frames] + [n * CHUNK_MS])
    ax.set_xticklabels([f"{(f - 1) * CHUNK_MS / 1000:.2f}" if (f - 1) % 4 == 0 else "" for f in frames]
                       + [f"{(frames[-1]) * CHUNK_MS / 1000:.2f}"], fontsize=7, color=INK2)
    if abs(zoom - 1.0) > 1e-6:
        xl = f"time (s) — busy segments drawn at ×{zoom:.0f} real duration, white = idle (not to scale)"
    else:
        xl = f"time (s), true scale — chip throughput {a.tops * 1e3:.0f} GOPS, white = idle"
        if a.tts_scale != 1.0:
            xl += f"; TTS compute scaled by 1/{1 / a.tts_scale:.2f}"
    ax.set_xlabel(xl, fontsize=8, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2)
    # scale bar: 5 ms of compute
    if abs(zoom - 1.0) > 1e-6:
        sb = 5 * zoom
        x0 = (n - 0.98) * CHUNK_MS - sb
        ax.plot([x0, x0 + sb], [1.72, 1.72], color=INK, lw=1.2)
        ax.text(x0 + sb / 2, 1.76, f"5 ms compute @ {a.tops:.3g} TOPS", ha="center", va="bottom", fontsize=6.5, color=INK2)
    handles = [Patch(color=C_ASR, label="encoder + ASR"), Patch(color=C_MT, label="MT (beam search, KV cache)"),
               Patch(color=C_TTS, label="TTS (T2U + unit CTC + vocoder)"),
               Patch(facecolor=C_TTS, hatch="////", edgecolor="white", label="TTS, speculative (SSG)")]
    leg = fig.legend(handles=handles, loc="lower center", fontsize=7, frameon=False, ncol=4,
                     bbox_to_anchor=(0.5, -0.01))
    for t in leg.get_texts():
        t.set_color(INK2)
    if a.bare:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.spines["bottom"].set_visible(False)
        leg.remove()
        ax.set_ylim(-H / 2 - 0.15, 1.0 + H / 2 + 0.15)
    tot0 = sum(busy(c0, f, ("asr", "mt", "tts")) for f in frames)
    tot1 = sum(busy(c1, f, ("asr", "mt", "tts")) for f in frames)
    sp1 = sum(to_ms(c1[f]["spec"]) for f in frames)
    if not a.no_annot:
        ax.set_title(f"utt {i}: “{inst_i['prediction'].strip()}”   —   critical-path compute {tot0:.1f} ms → {tot1:.1f} ms "
                     f"(+{sp1:.1f} ms speculative, off the critical path)", fontsize=8, color=INK, loc="left", pad=10)
    else:
        print(f"critical-path compute {tot0:.1f} ms -> {tot1:.1f} ms (+{sp1:.1f} ms speculative)")
    fig.tight_layout(rect=(0, 0, 1, 1) if a.bare else (0, 0.06, 1, 1))
    fig.savefig(a.out, facecolor="white")
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf", facecolor="white")
    print("saved", a.out, f"(zoom x{zoom:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--utt", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "res/conf-trigger/fig_trace.png"))
    ap.add_argument("--tops", type=float, default=2.92)
    ap.add_argument("--calib", default=os.path.join(HERE, "flops_calib.json"))
    ap.add_argument("--sub", default="test100",
                    help="result subset dir under res/conf-trigger (test100, es-test100, de-test100)")
    ap.add_argument("--run-wo", default="stepwise-prod-0.5-tts")
    ap.add_argument("--run-w", default="stepwise-prod-0.5-earlytok-0.1")
    ap.add_argument("--timeline", action="store_true",
                    help="horizontal hardware-occupancy view: time left->right, one lane per config")
    ap.add_argument("--zoom", type=float, default=None,
                    help="timeline: draw busy segments at this factor of real time (default: auto)")
    ap.add_argument("--tts-scale", type=float, default=1.0,
                    help="multiply all TTS compute (T2U + unit CTC + vocoder, incl. speculative) by this")
    ap.add_argument("--trim-lead", action="store_true",
                    help="timeline: drop leading frames that carry no MT/TTS work in either lane")
    ap.add_argument("--no-annot", action="store_true",
                    help="timeline: no in-plot text (title, commit / speculation / hit labels)")
    ap.add_argument("--gray", action="store_true", help="timeline: grayscale palette")
    ap.add_argument("--bare", action="store_true",
                    help="timeline: no text at all (no frame numbers, lane names, legend, axis, arrow labels)")
    ap.add_argument("--arrows", action="store_true",
                    help="timeline: arrow between the lanes showing the critical-path time saved per frame")
    ap.add_argument("--arrow-min", type=float, default=3.0, help="only draw arrows for savings >= this (ms)")
    ap.add_argument("--fit-lane", default="both", choices=["both", "w", "wo"],
                    help="which lane's busiest frame --fit-tops refers to (default both)")
    ap.add_argument("--fit-tops", type=float, default=None,
                    help="choose the chip throughput so that the busiest frame takes this fraction of "
                         "the 320 ms frame period (e.g. 0.95); implies --zoom 1")
    a = ap.parse_args()
    global RUN_WO, RUN_W
    RUN_WO = os.path.join(ROOT, "res/conf-trigger", a.sub, a.run_wo)
    RUN_W = os.path.join(ROOT, "res/conf-trigger", a.sub, a.run_w)
    cal = json.load(open(a.calib))
    ops = a.tops * 1e12
    g_wo, inst = read_groups(RUN_WO)
    g_w, _ = read_groups(RUN_W)
    durs = []
    for r in inst:
        durs.append(1000 * float([re.search(r"duration: ([\d.]+) s", x).group(1) for x in r["source"] if "duration" in x][0]))

    if a.list:
        cands = []
        for i in range(len(inst)):
            c0, n0 = per_frame_costs(g_wo[i], durs[i], cal, False)
            c1, n1 = per_frame_costs(g_w[i], durs[i], cal, True)
            fl = [f for f in n1 if n1[f]["flush"]]
            flush_wo = c0[fl[0]]["tts"] if fl else 0.0
            flush_w = c1[fl[0]]["tts"] if fl else 0.0
            hits = sum(n1[f]["hits"] for f in n1)
            spec = sum(len(n1[f]["spec_toks"]) for f in n1)
            tot0 = sum(sum(c.values()) for c in c0.values())
            tot1 = sum(c["asr"] + c["mt"] + c["tts"] for c in c1.values())
            cands.append((i, len(c0), hits, spec, flush_wo / 1e9, flush_w / 1e9, tot0 / 1e9, tot1 / 1e9, inst[i]["prediction"].strip()))
        cands.sort(key=lambda t: -(t[4] - t[5]))
        print("utt frames hits spec  flushTTS_wo  flushTTS_w  crit_wo  crit_w  | text")
        for t in cands[:15]:
            print(f"{t[0]:3d} {t[1]:6d} {t[2]:4d} {t[3]:4d}  {t[4]:10.2f}  {t[5]:10.2f}  {t[6]:7.2f}  {t[7]:6.2f}  | {t[8]}")
        return

    i = a.utt
    c0, n0 = per_frame_costs(g_wo[i], durs[i], cal, False)
    c1, n1 = per_frame_costs(g_w[i], durs[i], cal, True)
    frames = sorted(c0)
    for c in (c0, c1):
        for f in c:
            c[f]["tts"] *= a.tts_scale
            c[f]["spec"] *= a.tts_scale
    if a.fit_tops:
        b0 = max(sum(c0[f].values()) for f in frames)
        b1 = max(sum(c1[f].values()) for f in frames)
        busiest = {"both": max(b0, b1), "w": b1, "wo": b0}[a.fit_lane]
        a.tops = busiest / (a.fit_tops * CHUNK_MS / 1000.0) / 1e12
        a.zoom = 1.0
        print(f"busiest frame {busiest / 1e9:.2f} GOPs -> chip set to {a.tops * 1e3:.1f} GOPS "
              f"so that it takes {100 * a.fit_tops:.0f}% of a {CHUNK_MS} ms frame")
    ops = a.tops * 1e12
    to_ms = lambda x: x / ops * 1e3  # noqa: E731

    print(f"utt {i}: {inst[i]['prediction'].strip()!r}  ref: {inst[i]['reference']!r}  dur {durs[i]:.0f} ms, {len(frames)} frames")
    print("frame | w/o SSG  asr   mt   tts  | w/ SSG  asr   mt   tts  spec | commit / hits / speculated")
    for f in frames:
        a0, b0 = c0[f], c1[f]
        print(f"{f:5d} | {to_ms(a0['asr']):5.2f} {to_ms(a0['mt']):5.2f} {to_ms(a0['tts']):5.2f} ms | "
              f"{to_ms(b0['asr']):5.2f} {to_ms(b0['mt']):5.2f} {to_ms(b0['tts']):5.2f} {to_ms(b0['spec']):5.2f} ms | "
              f"{n1[f]['commit']!r} hits={n1[f]['hits']} spec={n1[f]['spec_toks']}{' FLUSH' if n1[f]['flush'] else ''}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    C_ASR, C_MT, C_TTS = "#2a78d6", "#eb6834", "#1baf7a"
    INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"

    if a.timeline:
        draw_timeline(frames, c0, c1, n1, to_ms, inst[i], i, a, C_ASR, C_MT, C_TTS, INK, INK2, MUTED, SURF)
        return
    x = np.arange(len(frames))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SURF)

    def stack(xs, cost, keys, hatch=None):
        bottom = np.zeros(len(xs))
        for k, col in keys:
            vals = np.array([to_ms(cost[f][k]) for f in frames])
            ax.bar(xs, vals, w, bottom=bottom, color=col, edgecolor="white", linewidth=0.8,
                   hatch=("///" if k == "spec" else None))
            bottom += vals
        return bottom

    top0 = stack(x - w / 2 - 0.02, c0, [("asr", C_ASR), ("mt", C_MT), ("tts", C_TTS)])
    top1 = stack(x + w / 2 + 0.02, c1, [("asr", C_ASR), ("mt", C_MT), ("tts", C_TTS), ("spec", C_TTS)])

    ymax = max(top0.max(), top1.max())
    for j, f in enumerate(frames):
        if n1[f]["commit"]:
            label = n1[f]["commit"]
            if n1[f]["flush"]:
                label = "(flush) " + label
            if len(label) > 40:
                label = label[:38] + "…"
            last = j == len(frames) - 1
            y_lab = max(top0[j], top1[j]) + ymax * (0.09 if n1[f]["hits"] else 0.03)
            ax.text(x[j] + (w if last else 0), y_lab, label, ha=("right" if last else "center"),
                    va="bottom", fontsize=6.5, color=INK2)
            if n1[f]["hits"]:
                ax.text(x[j] + w / 2 + 0.02, top1[j] + ymax * 0.01,
                        f"−{n1[f]['hits']} tok\npre-synth.", ha="center", va="bottom",
                        fontsize=5.8, color=C_TTS, linespacing=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in frames], fontsize=8, color=INK2)
    ax.set_xlabel("input frame (320 ms chunk)", fontsize=9, color=INK2)
    ax.set_ylabel(f"compute per frame (ms @ {a.tops} TOPS)", fontsize=9, color=INK2)
    ax.set_ylim(0, ymax * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(0.0, 1.02, f"available time per frame: {CHUNK_MS} ms (off-scale)", transform=ax.transAxes,
            fontsize=7.5, color=MUTED, ha="left", va="bottom")
    handles = [Patch(color=C_ASR, label="encoder + ASR"), Patch(color=C_MT, label="MT (beam search, KV cache)"),
               Patch(color=C_TTS, label="TTS (T2U + unit CTC + vocoder)"),
               Patch(facecolor=C_TTS, hatch="///", edgecolor="white", label="TTS, speculative (SSG)")]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False, ncol=2,
                    bbox_to_anchor=(0.0, 0.98))
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(1.0, 1.02, "left bar: w/o SSG   right bar: w/ SSG (τ = 0.1, min 5 steps)", transform=ax.transAxes,
            fontsize=7.5, color=INK2, ha="right", va="bottom")
    ax.set_title(f"utt {i}: “{inst[i]['prediction'].strip()}”", fontsize=9, color=INK, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(a.out, facecolor="white")
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf", facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
