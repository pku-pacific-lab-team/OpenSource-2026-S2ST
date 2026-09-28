# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Build the CVSS-C fr-en full-testset SimulEval inputs from fixie-ai/covost2 parquet.

Writes:
  datasets/cvss-eval/test/clips/<id>.mp3   original 48kHz Common Voice v4 mp3 bytes
  datasets/cvss-eval/test/wav_list.txt     absolute paths, sorted by utterance id
  datasets/cvss-eval/test/src.txt          French transcript, authors' normalization
                                           (lowercase, strip all punctuation)
  datasets/cvss-eval/test/target.covost.txt  English translation, CVSS-style
                                           normalization (lowercase, keep apostrophes)

CVSS-C references and the test1k / test100 subsets are then built by
make_subsets.py.

Usage (from the StreamSpeech root):
  python <this repo>/model/data_prep/build_fulltest_data.py
"""
import re
from pathlib import Path

import pyarrow.parquet as pq

PARQUET_DIR = Path("datasets/covost2/fr_en_parquet")
OUT = Path("datasets/cvss-eval/test")


def main():
    rows = []
    for f in sorted(PARQUET_DIR.glob("test-*.parquet")):
        t = pq.read_table(f, columns=["id", "sentence", "translation", "audio"])
        rows.extend(t.to_pylist())
        print(f"{f.name}: {t.num_rows} rows", flush=True)
    rows.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate utterance ids"
    print(f"total {len(rows)} utterances")

    clips = OUT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    with open(OUT / "wav_list.txt", "w", encoding="utf-8") as f_wav, \
         open(OUT / "src.txt", "w", encoding="utf-8") as f_src, \
         open(OUT / "target.covost.txt", "w", encoding="utf-8") as f_tgt:
        for r in rows:
            mp3 = clips / f'{r["id"]}.mp3'
            if not mp3.exists() or mp3.stat().st_size != len(r["audio"]["bytes"]):
                mp3.write_bytes(r["audio"]["bytes"])
            f_wav.write(str(mp3.resolve()) + "\n")
            src = re.sub(r"[^\w\s]", "", r["sentence"].lower())
            f_src.write(" ".join(src.split()) + "\n")
            tgt = re.sub(r"[^\w\s']", "", r["translation"].lower())
            f_tgt.write(" ".join(tgt.split()) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
