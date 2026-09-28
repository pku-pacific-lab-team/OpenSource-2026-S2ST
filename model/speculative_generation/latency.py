# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Critical-path compute latency per TTS commit for the stepwise policy.

latency(commit) = FLOPs on the critical path / hardware throughput, where the
critical path (all incremental, nothing recomputed) is:
  encoder(new 320 ms chunk, attending to cached history)
  + ASR CTC head (new 8 encoder frames)
  + MT decoder: one-token forwards with KV cache up to the stepwise decision
  + T2U encoder + unit CTC decoder on the committed tokens that still need TTS
  + vocoder on their units (duration prediction included)

With speculative pre-synthesis (--early-mass-thr in the agent), tokens that were
pre-synthesised at an earlier trigger (same token, same position) are excluded
from the commit's TTS cost; the search steps beyond the stepwise decision and
the pre-synthesis itself are reported separately as background compute.

Usage: python <this repo>/model/speculative_generation/latency.py <run_dir> [--calib flops_calib.json] [--tops 2.92]
"""
import argparse
import json
import os

import numpy as np


def interp(table, x):
    xs = np.array(table["len"] if "len" in table else table["fbank_len"])
    return float(np.interp(x, xs, np.array(table["flops"])))


def total_at(table, n):
    return 0.0 if n <= 0 else interp(table, n)


def marginal(table, pos):
    """cost of the token at absolute position pos (0-based) in a causal module."""
    return total_at(table, pos + 1) - total_at(table, pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--calib", default=os.path.join(os.path.dirname(__file__), "flops_calib.json"))
    ap.add_argument("--tops", type=float, default=2.92, help="hardware throughput, tera-ops/s (1 MAC = 2 ops)")
    ap.add_argument("--dump", default=None, help="write per-commit csv here")
    a = ap.parse_args()
    cal = json.load(open(a.calib))
    ops = a.tops * 1e12
    m, v = cal["mt_dec_token"], cal["vocoder"]

    def voc_cost(units, expanded):
        return v["a"] + v["b_units"] * units + v["c_expanded"] * expanded

    rows, bg_rows = [], []
    tok_total = tok_hit = 0
    wasted = 0
    for line in open(os.path.join(a.run_dir, "trace.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        wasted += r.get("spec_overwritten", 0) + r.get("spec_leftover", 0)
        s = r.get("shapes")
        # ---- background compute of this trigger (extra search + pre-synthesis)
        if s is not None or r.get("spec_new"):
            bg = 0.0
            if s is not None:
                bg += s.get("dec_token_fwds_extra", 0) * (
                    m["a"] + m["b_prefix"] * s["prefix_len"] + m["c_enc"] * s["enc_frames"])
            for pos, units, expanded in r.get("spec_new", []):
                bg += marginal(cal["t2u_total"], pos) + marginal(cal["unit_ctc_total"], pos)
                if units:
                    bg += voc_cost(units, expanded)
            bg_rows.append(bg)
        if r.get("action") != "WRITE" or s is None:
            continue
        wasted += s.get("spec_wasted", 0)
        P, n = s["prefix_len"], s["new_tokens"]
        tok_total += n
        tok_hit += s.get("spec_hits", 0)
        tts_pos = s.get("tts_positions", list(range(n)))
        f = {}
        f["encoder"] = interp(cal["encoder_marginal"], s["fbank_frames"])
        f["asr_head"] = cal["asr_head_per_enc_frame"] * 8
        p_eff = P + max(s["beam_steps"] - 1, 0) / 2.0
        f["mt_decoder"] = s["dec_token_fwds"] * (m["a"] + m["b_prefix"] * p_eff + m["c_enc"] * s["enc_frames"])
        f["t2u"] = sum(marginal(cal["t2u_total"], P + i) for i in tts_pos)
        f["unit_ctc"] = sum(marginal(cal["unit_ctc_total"], P + i) for i in tts_pos)
        units = s.get("tts_units", s.get("new_units", 0))
        expanded = s.get("tts_units_expanded", s.get("new_units_expanded", 0))
        f["vocoder"] = voc_cost(units, expanded) if units else 0.0
        tot = sum(f.values())
        rows.append({
            "sent": r["sent"], "src_ms": r["src_ms"], "final_flush": bool(s.get("final_flush", False)),
            "new_tokens": n, "tts_tokens": len(tts_pos), "beam_steps": s["beam_steps"],
            "new_units": s.get("new_units"), "new_units_expanded": s.get("new_units_expanded"),
            "tts_units": units, "tts_units_expanded": expanded,
            **{k + "_GF": val / 1e9 for k, val in f.items()}, "total_GF": tot / 1e9,
            "latency_ms": tot / ops * 1e3,
        })

    lat = np.array([r["latency_ms"] for r in rows])
    fl = np.array([r["final_flush"] for r in rows])
    comps = ["encoder", "asr_head", "mt_decoder", "t2u", "unit_ctc", "vocoder"]

    def st(x):
        return f"mean {x.mean():.2f}  median {np.median(x):.2f}  p90 {np.percentile(x, 90):.2f}  max {x.max():.2f} ms"

    print(f"run: {a.run_dir}   commits: {len(rows)}   throughput: {a.tops} TOPS")
    print(f"all commits       ({len(lat)}): {st(lat)}")
    if (~fl).any():
        print(f"streaming commits ({(~fl).sum()}): {st(lat[~fl])}")
    if fl.any():
        print(f"final flush       ({fl.sum()}): {st(lat[fl])}")
    print("\n| component | mean GFLOPs (streaming) | mean ms | share |")
    print("|---|---|---|---|")
    sel = ~fl if (~fl).any() else np.ones(len(rows), bool)
    tot_mean = np.mean([r["total_GF"] for r, k in zip(rows, sel) if k])
    for c in comps:
        g = np.mean([r[c + "_GF"] for r, k in zip(rows, sel) if k])
        print(f"| {c} | {g:.4f} | {g * 1e9 / ops * 1e3:.2f} | {100 * g / tot_mean:.1f}% |")
    print(f"| total | {tot_mean:.4f} | {tot_mean * 1e9 / ops * 1e3:.2f} | 100% |")
    print(f"\nmean new tokens/commit {np.mean([r['new_tokens'] for r in rows]):.2f}, "
          f"mean tokens needing TTS/commit {np.mean([r['tts_tokens'] for r in rows]):.2f}")
    if tok_total:
        print(f"speculation: {tok_hit}/{tok_total} committed tokens were pre-synthesised "
              f"({100 * tok_hit / tok_total:.1f}%), wasted pre-syntheses: {wasted}")
    if bg_rows:
        bg = np.array(bg_rows)
        print(f"background compute per trigger: mean {bg.mean() / 1e9:.3f} GFLOPs "
              f"({bg.mean() / ops * 1e3:.2f} ms), total {bg.sum() / 1e9:.1f} GFLOPs over {len(bg)} triggers")
    if a.dump:
        import csv
        with open(a.dump, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
