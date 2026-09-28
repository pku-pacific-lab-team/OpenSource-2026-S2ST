# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Fast S2ST quality eval loop for vocoder-side feature transforms.

Re-synthesises the streaming S2ST output from recorded per-emission unit
sequences (LP_RECORD_UNITS run) exactly as the SimulEval agent does (full
re-synthesis per emission, keep the tail for the new units, silence gaps from
instances.log), so only the vocoder + Whisper scoring run per experiment.

Example (one baseline + threshold sweep, first 250 utterances):
  PYTHONPATH=fairseq python <this repo>/model/lp_decomposition/replay_eval.py \
      --rec <recording dir> --n 250 --out res/lp/sweep_abs \
      --thresh 0 0.05 0.1 0.2 --mode abs
`--thresh -1` means hooks disabled (pure baseline).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch

ROOT = Path(os.environ.get("STREAMSPEECH_ROOT", os.getcwd()))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "fairseq"))

from lp_sparse import LPConfig, install_hooks, remove_hooks, collect_stats  # noqa: E402

VOC = ROOT / "pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en"
SR = 16000


def load_vocoder(device):
    from agent.tts.vocoder import CodeHiFiGANVocoderWithDur

    with open(VOC / "config.json") as f:
        cfg = json.load(f)
    voc = CodeHiFiGANVocoderWithDur(str(VOC / "g_00500000"), cfg).to(device)
    voc.eval()
    return voc


def load_recording(rec_dir, n, start):
    rec_dir = Path(rec_dir)
    units = {}
    with open(str(rec_dir) + ".units.jsonl") as f:
        for line in f:
            d = json.loads(line)
            units[d["index"]] = d["calls"]
    logs = {}
    with open(rec_dir / "instances.log") as f:
        for line in f:
            d = json.loads(line)
            logs[d["index"]] = d
    idx = sorted(i for i in logs if i in units and start <= i < start + n)
    if len(idx) < n:
        print(f"[warn] only {len(idx)} recorded utterances available in [{start}, {start+n})")
    return idx, units, logs


@torch.no_grad()
def synth_utterance(voc, calls, log, device):
    """Replay the agent's emissions and assemble the wav like SpeechOutputInstance.summarize."""
    segs = []
    prev = None
    for unit in calls:
        cur = unit if prev is None else unit[len(prev):]
        x = {"code": torch.tensor(unit, dtype=torch.long, device=device).view(1, -1)}
        wav, dur = voc(x, True)
        n = int(dur[:, -len(cur):].sum()) * 320
        segs.append(wav[-n:].float().cpu().numpy())
        prev = unit

    samples = []
    seg_i = 0
    prev_end = None
    for delay, duration in zip(log["delays"], log["durations"]):
        if prev_end is None:
            prev_end = delay
        start = max(prev_end, delay)
        if start > prev_end:
            samples.append(np.zeros(int(SR * (start - prev_end) / 1000), dtype=np.float32))
        if duration > 0:
            seg = segs[seg_i]
            seg_i += 1
            assert abs(len(seg) / SR * 1000 - duration) < 1e-6, (
                f"segment length mismatch: got {len(seg)/16} ms, log says {duration} ms"
            )
            samples.append(seg)
        prev_end = start + duration
    assert seg_i == len(segs), f"used {seg_i} of {len(segs)} synthesised segments"
    return np.concatenate(samples) if samples else np.zeros(0, dtype=np.float32)


def transcribe_and_score(wav_dir, idx, refs, model, out_dir):
    import sacrebleu
    from simuleval.evaluator.scorers.quality_scorer import remove_punctuations

    hyps = []
    for i in idx:
        r = model.transcribe(str(wav_dir / f"{i}_pred.wav"), language="en")
        hyps.append(remove_punctuations(r["text"].lower()).strip())
    with open(out_dir / "asr_transcripts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")
    return sacrebleu.corpus_bleu(hyps, [[refs[i] for i in idx]], tokenize="13a").score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, help="recording dir (has instances.log; <dir>.units.jsonl beside it)")
    ap.add_argument("--target", default=str(ROOT / "datasets/cvss-eval/test1k/target.txt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--thresh", type=float, nargs="+", default=[-1.0], help="-1 = hooks off")
    ap.add_argument("--mode", default="abs", choices=["abs", "rel"])
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--layers", default="resblock", choices=["resblock", "all"])
    ap.add_argument("--closed-loop", action="store_true", help="DPCM-style prediction from reconstructed past")
    ap.add_argument("--whisper-size", default="small")
    ap.add_argument("--no-whisper", action="store_true")
    ap.add_argument("--score-only", action="store_true", help="skip synthesis; score existing wavs (+ lp_stats.json) in --out")
    ap.add_argument("--compare-wavs", default=None, help="dir with reference *_pred.wav to check replay fidelity")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    voc = load_vocoder(device)
    idx, units, logs = load_recording(args.rec, args.n, args.start)
    refs = [l.rstrip("\n") for l in open(args.target, encoding="utf-8")]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    whisper_model = None
    if not args.no_whisper:
        import whisper

        whisper_model = whisper.load_model(args.whisper_size, device=device)

    summary = []
    for thresh in args.thresh:
        tag = "baseline" if thresh < 0 else f"{args.mode}_thr{thresh:g}" + ("_cl" if args.closed_loop else "")
        out_dir = out_root / tag
        wav_dir = out_dir / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        if thresh < 0:
            remove_hooks(voc.model)
            cfg = None
        else:
            cfg = LPConfig(block=args.block, order=args.order, thresh=thresh, mode=args.mode,
                           share=args.share, ridge=args.ridge, layers=args.layers,
                           closed_loop=args.closed_loop)
            install_hooks(voc.model, cfg)

        t0 = time.time()
        max_diff = 0.0
        if args.score_only:
            missing = [i for i in idx if not (wav_dir / f"{i}_pred.wav").exists()]
            assert not missing, f"--score-only: {len(missing)} wavs missing in {wav_dir}, e.g. {missing[:3]}"
            stats = json.load(open(out_dir / "lp_stats.json")) if (out_dir / "lp_stats.json").exists() else None
        else:
            for i in idx:
                wav = synth_utterance(voc, units[i], logs[i], device)
                soundfile.write(wav_dir / f"{i}_pred.wav", wav, SR)
                if args.compare_wavs:
                    ref, _ = soundfile.read(Path(args.compare_wavs) / f"{i}_pred.wav", dtype="float32")
                    if len(ref) != len(wav):
                        max_diff = float("inf")
                        print(f"[compare] {i}: length {len(wav)} vs ref {len(ref)}")
                    else:
                        max_diff = max(max_diff, float(np.abs(ref - wav).max()))
            stats = collect_stats(voc.model) if cfg else None
            if stats:
                with open(out_dir / "lp_stats.json", "w") as f:
                    json.dump(stats, f, indent=1)
        t_synth = time.time() - t0

        bleu = None
        t_asr = 0.0
        if whisper_model is not None:
            t0 = time.time()
            bleu = transcribe_and_score(wav_dir, idx, refs, whisper_model, out_dir)
            t_asr = time.time() - t0

        row = {
            "tag": tag, "thresh": thresh, "mode": args.mode, "block": args.block,
            "order": args.order, "share": args.share, "layers": args.layers,
            "closed_loop": args.closed_loop,
            "n_utts": len(idx), "start": args.start, "asr_bleu": bleu,
            "residual_sparsity": stats["overall"]["residual_sparsity"] if stats else None,
            "mac_weighted_sparsity": stats["overall"]["mac_weighted_sparsity"] if stats else None,
            "synth_sec": round(t_synth, 1), "asr_sec": round(t_asr, 1),
        }
        if args.compare_wavs:
            row["max_abs_diff_vs_ref"] = max_diff
        summary.append(row)
        with open(out_dir / "scores.json", "w") as f:
            json.dump(row, f, indent=1)
        print(json.dumps(row), flush=True)

    with open(out_root / "summary.jsonl", "a") as f:
        for row in summary:
            f.write(json.dumps(row) + "\n")
    print("\n%-14s %8s %10s %10s" % ("tag", "ASR-BLEU", "res.spars", "mac.spars"))
    for r in summary:
        print("%-14s %8s %10s %10s" % (
            r["tag"], "-" if r["asr_bleu"] is None else f"{r['asr_bleu']:.2f}",
            "-" if r["residual_sparsity"] is None else f"{r['residual_sparsity']:.3f}",
            "-" if r["mac_weighted_sparsity"] is None else f"{r['mac_weighted_sparsity']:.3f}"))


if __name__ == "__main__":
    main()
