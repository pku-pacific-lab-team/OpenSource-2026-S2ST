# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Build a 100-utterance SimulEval eval subset for another CoVoST2/CVSS language
pair from one fixie-ai/covost2 test parquet shard.

Usage: python <this repo>/model/speculative_generation/build_lang_test100.py es   # or de
Writes datasets/cvss-eval/<lang>-test100/{clips/, wav_list.txt, src.txt, target.txt}

Notes
- Takes the first 100 utterances of the shard sorted by utterance id (the fr-en
  test100 is the first 100 ids of the full test split, built the same way).
- target.txt is the CoVoST2 English translation with CVSS-style normalisation
  (lowercase, punctuation stripped except apostrophes).  The real CVSS-C text
  differs from this for ~20% of sentences (fr-en experience), so absolute BLEU
  is only indicative; the latency / speculation statistics do not depend on it.
"""
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

lang = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
PARQUET_DIR = Path(f"datasets/covost2/{lang}_en_parquet")
OUT = Path(f"datasets/cvss-eval/{lang}-test{n}")

rows = []
for f in sorted(PARQUET_DIR.glob("test-*.parquet")):
    t = pq.read_table(f, columns=["id", "sentence", "translation", "audio"])
    rows.extend(t.to_pylist())
    print(f"{f.name}: {t.num_rows} rows", flush=True)
rows.sort(key=lambda r: r["id"])
rows = rows[:n]
clips = OUT / "clips"
clips.mkdir(parents=True, exist_ok=True)
with open(OUT / "wav_list.txt", "w", encoding="utf-8") as f_wav, \
     open(OUT / "src.txt", "w", encoding="utf-8") as f_src, \
     open(OUT / "target.txt", "w", encoding="utf-8") as f_tgt:
    for r in rows:
        mp3 = clips / f'{r["id"]}.mp3'
        if not mp3.exists() or mp3.stat().st_size != len(r["audio"]["bytes"]):
            mp3.write_bytes(r["audio"]["bytes"])
        f_wav.write(str(mp3.resolve()) + "\n")
        src = re.sub(r"[^\w\s]", "", r["sentence"].lower())
        f_src.write(" ".join(src.split()) + "\n")
        tgt = re.sub(r"[^\w\s']", "", r["translation"].lower())
        f_tgt.write(" ".join(tgt.split()) + "\n")
print("wrote", OUT, len(rows), "utterances; first:", rows[0]["sentence"], "->", rows[0]["translation"])
