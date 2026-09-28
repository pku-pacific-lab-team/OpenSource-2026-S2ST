# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Merge sharded replay_eval.py runs (same tag, disjoint --start/--n ranges) into one
corpus-level ASR-BLEU and exact pooled sparsity stats.

Usage: python <this repo>/model/lp_decomposition/merge_shards.py --tag abs_thr0.02 --out <dir> <shard_dir> [<shard_dir> ...]
Each shard dir is a replay_eval --out directory containing <tag>/{scores.json,asr_transcripts.txt,lp_stats.json}.
"""
import os
import argparse
import json
from pathlib import Path

import sacrebleu

ROOT = Path(os.environ.get("STREAMSPEECH_ROOT", os.getcwd()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=str(ROOT / "datasets/cvss-eval/test1k/target.txt"))
    args = ap.parse_args()

    refs_all = [l.rstrip("\n") for l in open(args.target, encoding="utf-8")]
    hyps, refs, rows, groups = [], [], [], {}
    for sd in args.shards:
        d = Path(sd) / args.tag
        row = json.load(open(d / "scores.json"))
        rows.append(row)
        lines = [l.rstrip("\n") for l in open(d / "asr_transcripts.txt", encoding="utf-8")]
        assert len(lines) == row["n_utts"], (d, len(lines), row["n_utts"])
        hyps += lines
        refs += refs_all[row["start"] : row["start"] + row["n_utts"]]
        st = json.load(open(d / "lp_stats.json")) if (d / "lp_stats.json").exists() else None
        if st:
            for g, v in list(st["by_group"].items()) + [("overall", st["overall"])]:
                acc = groups.setdefault(g, {"pred": 0, "zero": 0, "total": 0, "macs": 0.0, "macs_zero": 0.0})
                for k in acc:
                    acc[k] += v["counts"][k]

    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score
    stats = {
        g: {
            "residual_sparsity": c["zero"] / c["pred"] if c["pred"] else 0.0,
            "sparsity_incl_unpredicted": c["zero"] / c["total"] if c["total"] else 0.0,
            "mac_weighted_sparsity": c["macs_zero"] / c["macs"] if c["macs"] else 0.0,
            "gmacs": c["macs"] / 1e9,
        }
        for g, c in groups.items()
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "asr_transcripts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")
    result = {
        "tag": args.tag,
        "n_utts": len(hyps),
        "asr_bleu": bleu,
        "residual_sparsity": stats.get("overall", {}).get("residual_sparsity"),
        "mac_weighted_sparsity": stats.get("overall", {}).get("mac_weighted_sparsity"),
        "by_group": stats,
        "shards": rows,
    }
    with open(out / "scores.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ("by_group", "shards")}))
    for g, v in stats.items():
        print(f"  {g:16s} res {v['residual_sparsity']:.3f}  mac {v['mac_weighted_sparsity']:.3f}")


if __name__ == "__main__":
    main()
