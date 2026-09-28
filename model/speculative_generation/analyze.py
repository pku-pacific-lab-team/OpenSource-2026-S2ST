# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Summarise a conf-trigger sweep: BLEU/AL plus how often early commits were wrong.

Usage: python <this repo>/model/speculative_generation/analyze.py res/conf-trigger/test100 [--offline offline-b5]

A streaming commit is counted as *wrong* when the text committed so far is not a
word-level prefix of what the same beam-5 decoder produces with the full utterance
(the `offline` run).  This isolates errors caused by committing too early from
errors the MT model would make anyway.
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def read_instances(d):
    p = os.path.join(d, "instances.log")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        out[r["index"]] = r
    return out


def read_scores(d):
    p = os.path.join(d, "scores.tsv")
    if not os.path.exists(p):
        return {}
    lines = open(p).read().strip().splitlines()
    return dict(zip(lines[0].split("\t"), [float(x) for x in lines[1].split("\t")]))


def read_trace(d):
    """Group trace rows by utterance in order of appearance -> list of row lists."""
    p = os.path.join(d, "trace.jsonl")
    if not os.path.exists(p):
        return []
    groups, cur, cur_id = [], [], None
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["sent"] != cur_id:
            if cur:
                groups.append(cur)
            cur, cur_id = [], r["sent"]
        cur.append(r)
    if cur:
        groups.append(cur)
    return groups


def words(s):
    return s.strip().split()


def is_prefix(a, b):
    return len(a) <= len(b) and b[: len(a)] == a


def analyse(run_dir, offline_inst):
    inst = read_instances(run_dir)
    sc = read_scores(run_dir)
    groups = read_trace(run_dir)
    n = len(inst)
    res = {"run": os.path.basename(run_dir), "n": n}
    res.update({k: sc.get(k) for k in ("BLEU", "AL", "LAAL")})
    if n == 0:
        return res

    # final output vs offline output of the same decoder
    diff = 0
    for i, r in inst.items():
        if i in offline_inst and words(r["prediction"]) != words(offline_inst[i]["prediction"]):
            diff += 1
    res["final!=offline%"] = 100.0 * diff / n

    if not groups:
        return res
    n_commits = n_wrong = n_words = n_assessed = 0
    sents_with_wrong = 0
    triggers = 0
    for i, rows in enumerate(groups):
        off = words(offline_inst[i]["prediction"]) if i in offline_inst else None
        wrong_here = False
        for r in rows:
            triggers += 1
            if r["action"] != "WRITE" or r["source_finished"]:
                continue
            n_commits += 1
            n_words += len(words(r.get("committed_text", "")))
            if wrong_here or off is None:
                # once the committed text has diverged from the offline output,
                # later commits cannot be judged against it -> not assessed
                continue
            n_assessed += 1
            committed_so_far = words(r["prefix"] + " " + r.get("committed_text", ""))
            if not is_prefix(committed_so_far, off):
                n_wrong += 1
                wrong_here = True
        sents_with_wrong += wrong_here
    res["triggers/sent"] = triggers / len(groups)
    res["commits/sent"] = n_commits / len(groups)
    res["words/commit"] = (n_words / n_commits) if n_commits else 0.0
    # wrong-commit% = first-divergence rate among commits made while still on track
    res["wrong-commit%"] = (100.0 * n_wrong / n_assessed) if n_assessed else 0.0
    res["sents-w/-wrong%"] = 100.0 * sents_with_wrong / len(groups)
    return res


def show_wrong(run_dir, offline_inst, limit):
    """Print the wrong early commits of one run for qualitative inspection."""
    inst = read_instances(run_dir)
    shown = 0
    for i, rows in enumerate(read_trace(run_dir)):
        off = words(offline_inst[i]["prediction"]) if i in offline_inst else None
        for r in rows:
            if r["action"] != "WRITE" or r["source_finished"]:
                continue
            so_far = words(r["prefix"] + " " + r.get("committed_text", ""))
            if off is None or is_prefix(so_far, off):
                continue
            print(f"--- utt {i}  t={r['src_ms']}ms  conf={r['conf']}")
            print(f"  ASR so far : {r['asr']}")
            print(f"  committed  : [{r['prefix']}] + [{r.get('committed_text', '')}]")
            print(f"  top-1 hyp  : {r['top1']}")
            print(f"  offline    : {' '.join(off)}")
            print(f"  final      : {inst[i]['prediction'].strip() if i in inst else '?'}")
            print(f"  reference  : {inst[i]['reference'] if i in inst else '?'}")
            shown += 1
            if shown >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--offline", default="offline-b5")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--show", default=None, help="run name: dump its wrong commits")
    ap.add_argument("--limit", type=int, default=15)
    a = ap.parse_args()

    offline_inst = read_instances(os.path.join(a.sweep_dir, a.offline))
    if a.show:
        show_wrong(os.path.join(a.sweep_dir, a.show), offline_inst, a.limit)
        return
    runs = sorted(
        d for d in os.listdir(a.sweep_dir) if os.path.isdir(os.path.join(a.sweep_dir, d))
    )
    rows = [analyse(os.path.join(a.sweep_dir, d), offline_inst) for d in runs]
    cols = [
        "run", "n", "BLEU", "AL", "LAAL", "final!=offline%",
        "commits/sent", "words/commit", "wrong-commit%", "sents-w/-wrong%", "triggers/sent",
    ]

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.1f}"
        return str(v)

    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    if a.csv:
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(fmt(r.get(c)) for c in cols) + "\n")


if __name__ == "__main__":
    main()
