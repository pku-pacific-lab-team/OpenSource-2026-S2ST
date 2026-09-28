# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Build the evaluation subsets used by the scripts in this directory tree.

Reads the full test split written by build_fulltest_data.py and the CVSS-C
test.tsv, then writes:

  datasets/cvss-eval/test/target.cvss.txt   CVSS-C references aligned to wav_list.txt
  datasets/cvss-eval/test1k/                1000 utterances, random.Random(42) sample
  datasets/cvss-eval/test100/               first 100 utterances

Each subset has wav_list.txt, src.txt and target.txt (CoVoST-style reference);
test1k additionally has target.cvss.txt and indices.txt.

Usage (from the StreamSpeech root):
  python <this repo>/model/data_prep/make_subsets.py [--pair fr-en]
"""
import argparse
import ntpath
import random
from pathlib import Path


def read(p):
    return [l.rstrip("\n") for l in open(p, encoding="utf-8")]


def write(p, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(l + "\n" for l in lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="fr-en")
    ap.add_argument("--root", default="datasets/cvss-eval")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    full = root / "test"
    wav = read(full / "wav_list.txt")
    src = read(full / "src.txt")
    tgt = read(full / "target.covost.txt")

    tsv = Path(f"datasets/cvss/cvss-c/{args.pair}/test.tsv")
    refs = dict(l.split("\t", 1) for l in read(tsv))
    cvss = [refs.get(ntpath.basename(w), "<MISSING>") for w in wav]
    write(full / "target.cvss.txt", cvss)

    idx = sorted(random.Random(args.seed).sample(range(len(wav)), args.n))
    sub = root / "test1k"
    write(sub / "indices.txt", [str(i) for i in idx])
    write(sub / "wav_list.txt", [wav[i] for i in idx])
    write(sub / "src.txt", [src[i] for i in idx])
    write(sub / "target.txt", [tgt[i] for i in idx])
    write(sub / "target.cvss.txt", [cvss[i] for i in idx])

    sub = root / "test100"
    write(sub / "wav_list.txt", wav[:100])
    write(sub / "src.txt", src[:100])
    write(sub / "target.txt", tgt[:100])
    print("wrote", root / "test1k", root / "test100")


if __name__ == "__main__":
    main()
