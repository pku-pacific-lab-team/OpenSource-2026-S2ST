# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Compute WER for SimulEval ASR runs from instances.log.

Usage: python <this repo>/model/quantization/compute_wer.py <run-name> [<run-name> ...]
Reference: datasets/cvss-eval/test1k/src.txt (authors' normalization).
"""
import json
import sys

REF = "datasets/cvss-eval/test1k/src.txt"


def edit_distance(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)]


def main():
    refs = [l.split() for l in open(REF, encoding="utf-8").read().splitlines()]
    n_words = sum(len(r) for r in refs)
    print(f"{'run':16s} {'WER':>7s}  ({n_words} ref words)")
    for run in sys.argv[1:]:
        preds = {}
        with open(f"res/fulltest/{run}/instances.log", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                preds[d["index"]] = d["prediction"].split()
        assert len(preds) == len(refs), f"{run}: {len(preds)} preds vs {len(refs)} refs"
        errs = sum(edit_distance(refs[i], preds[i]) for i in sorted(preds))
        print(f"{run:16s} {100*errs/n_words:6.2f}%")


if __name__ == "__main__":
    main()
