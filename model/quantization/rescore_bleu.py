# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Offline BLEU rescoring of SimulEval S2TT runs against real CVSS-C references.

Reads predictions from res/fulltest/<run>/instances.log and scores with
sacrebleu (13a tokenizer, same as SimulEval's BLEU) against both the
CoVoST-approx references and the real CVSS test.tsv references.

Usage: python <this repo>/model/quantization/rescore_bleu.py <run-name> [<run-name> ...]
"""
import json
import sys

import sacrebleu

REF_COVOST = "datasets/cvss-eval/test1k/target.txt"
REF_CVSS = "datasets/cvss-eval/test1k/target.cvss.txt"


def main():
    refs_covost = open(REF_COVOST, encoding="utf-8").read().splitlines()
    refs_cvss = open(REF_CVSS, encoding="utf-8").read().splitlines()
    print(f"{'run':16s} {'BLEU(covost-ref)':>17s} {'BLEU(CVSS-ref)':>15s}")
    for run in sys.argv[1:]:
        preds = {}
        with open(f"res/fulltest/{run}/instances.log", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                preds[d["index"]] = d["prediction"].strip()
        hyp = [preds[i] for i in sorted(preds)]
        assert len(hyp) == len(refs_cvss), f"{run}: {len(hyp)} preds vs {len(refs_cvss)} refs"
        b_cov = sacrebleu.corpus_bleu(hyp, [refs_covost]).score
        b_cvs = sacrebleu.corpus_bleu(hyp, [refs_cvss]).score
        print(f"{run:16s} {b_cov:17.3f} {b_cvs:15.3f}")


if __name__ == "__main__":
    main()
