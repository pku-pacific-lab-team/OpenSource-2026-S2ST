# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Offline logit-drift analysis: FP StreamSpeech vs MXINT4-quantized variants.

For each utterance in example/wav_list.txt (full audio, chunk-causal encoder
with chunk=8 i.e. 320ms, mirroring the streaming agents):
  - encoder output: relative L2 error, mean per-frame cosine similarity
  - ASR CTC head (source_unigram) + policy CTC head (ctc_target_unigram):
    per-frame top-1 agreement with FP, mean KL(FP || quant)
  - MT decoder (target_unigram): greedy output of each model, plus
    teacher-forced (on the FP greedy tokens) next-token top-1 agreement,
    KL, and NLL under each model.

Usage:
  PYTHONPATH=fairseq python <this repo>/model/quantization/compare_logit_drift.py \
      --ckpt-b pretrain_models/streamspeech.simultaneous.fr-en.mxint4-all.pt [more ...]
"""
import argparse
import ast

import numpy as np
import soundfile
import torch
import yaml

from fairseq import checkpoint_utils, tasks, utils
from fairseq.data.audio.audio_utils import convert_waveform
from examples.speech_to_text.data_utils import extract_fbank_features

ROOT = "."
DATA_BIN = "configs/fr-en"
CKPT_A = "pretrain_models/streamspeech.simultaneous.fr-en.pt"
CHUNK_MS = 320
SHIFT_MS, WINDOW_MS = 10, 25


def load_task_and_model(ckpt, task=None):
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
    if task is None:
        state["cfg"].common["user_dir"] = "researches/ctc_unity"
        utils.import_user_module(state["cfg"].common)
        task_args = state["cfg"]["task"]
        task_args.data = DATA_BIN
        task_args.config_yaml = "config_gcmvn.yaml"
        task_args.multitask_config_yaml = "config_mtl_asr_st_ctcst.yaml"
        task = tasks.setup_task(task_args)
    overrides = ast.literal_eval(state["cfg"].common_eval.model_overrides)
    models, _ = checkpoint_utils.load_model_ensemble(
        [ckpt], arg_overrides=overrides, task=task
    )
    model = models[0].eval().cuda()
    chunk = CHUNK_MS // 40
    model.encoder.chunk_size = chunk
    for conv in model.encoder.subsample.conv_layers:
        conv.chunk_size = min(chunk, 16)
    for layer in model.encoder.conformer_layers:
        layer.conv_module.depthwise_conv.chunk_size = min(chunk, 16)
    return task, model


def features(wav_path, gcmvn):
    samples, sr = soundfile.read(wav_path, dtype="float32")
    n_shift = int(SHIFT_MS * sr / 1000)
    n_win = int(WINDOW_MS * sr / 1000)
    num_frames = (len(samples) - (n_win - n_shift)) // n_shift
    samples = samples[: num_frames * n_shift + (n_win - n_shift)]
    waveform, _ = convert_waveform(
        torch.tensor(samples)[None], sr, to_mono=True, to_sample_rate=16000
    )
    feat = extract_fbank_features(waveform, 16000)
    feat = (feat - gcmvn["mean"]) / gcmvn["std"]
    return torch.tensor(feat, dtype=torch.float32).cuda()


def encode(model, feat):
    return model.encoder.forward_torchscript(
        {
            "src_tokens": feat[None],
            "src_lengths": torch.LongTensor([feat.size(0)]).cuda(),
        }
    )


def ctc_lprobs(model, enc, task_name):
    head = getattr(model, f"{task_name}_decoder")
    out = head(enc["encoder_out"][0])
    return model.get_normalized_probs(
        [out["encoder_out"].transpose(0, 1)], log_probs=True
    )[0]  # (T, V)


def ctc_collapse(ids, tgt_dict):
    toks = ids.tolist()
    toks = [v for i, v in enumerate(toks) if i == 0 or v != toks[i - 1]]
    toks = [v for v in toks if v != 0 and v != tgt_dict.pad_index]
    return "".join(tgt_dict[t] for t in toks).replace("▁", " ").strip()


def greedy_mt(mt_decoder, enc, eos, max_len=60):
    tokens = torch.LongTensor([[eos]]).cuda()
    out = []
    for _ in range(max_len):
        logits, _ = mt_decoder(tokens, encoder_out=enc)
        nxt = logits[0, -1].argmax().item()
        if nxt == eos:
            break
        out.append(nxt)
        tokens = torch.cat([tokens, torch.LongTensor([[nxt]]).cuda()], dim=1)
    return out


def kl(lp_a, lp_b):  # KL(A||B), inputs log-probs (T, V)
    return (lp_a.exp() * (lp_a - lp_b)).sum(-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-b", nargs="+", required=True)
    ap.add_argument("--act-quant-b", action="store_true",
                    help="also fake-quantize activations of model B to MXINT4")
    ap.add_argument("--act-block", type=int, default=32,
                    help="MX group size for activation quantization")
    ap.add_argument("--act-exempt", type=str, default="",
                    help="comma-separated module-name prefixes to exempt")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    with open(f"{DATA_BIN}/config_gcmvn.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.BaseLoader)
    gcmvn = np.load(cfg["global_cmvn"]["stats_npz_path"])

    task, model_a = load_task_and_model(CKPT_A)
    asr_dict = task.multitask_tasks["source_unigram"].tgt_dict
    mt_dict = task.multitask_tasks["target_unigram"].tgt_dict

    wavs = [l.strip() for l in open("example/wav_list.txt") if l.strip()]

    for ckpt_b in args.ckpt_b:
        _, model_b = load_task_and_model(ckpt_b, task)
        tag = ""
        if args.act_quant_b:
            from mx_act_quant import install_hooks
            exempt = tuple(p for p in args.act_exempt.split(",") if p)
            n = install_hooks(model_b, block=args.act_block, exempt=exempt)
            tag = (f" + MXINT4 activations (block={args.act_block}, "
                   f"{n} hooked modules, exempt={list(exempt)})")
        print("=" * 100)
        print(f"FP32 vs {ckpt_b}{tag}")
        agg = {"asr_agree": [], "st_agree": [], "asr_kl": [], "st_kl": [],
               "mt_agree": [], "mt_kl": [], "nll_a": [], "nll_b": [],
               "enc_rel": [], "enc_cos": []}
        for i, wav in enumerate(wavs):
            feat = features(wav, gcmvn)
            enc_a, enc_b = encode(model_a, feat), encode(model_b, feat)
            ha = enc_a["encoder_out"][0][:, 0]  # (T, C)
            hb = enc_b["encoder_out"][0][:, 0]
            rel = ((ha - hb).norm() / ha.norm()).item()
            cos = torch.nn.functional.cosine_similarity(ha, hb, dim=-1).mean().item()
            agg["enc_rel"].append(rel)
            agg["enc_cos"].append(cos)
            print(f"--- utt {i} ({feat.size(0)} frames -> {ha.size(0)} enc frames)")
            print(f"    encoder: rel-L2 err {rel:.4f}, mean cos {cos:.4f}")

            for name, key in [("source_unigram", "asr"), ("ctc_target_unigram", "st")]:
                lp_a, lp_b = ctc_lprobs(model_a, enc_a, name), ctc_lprobs(model_b, enc_b, name)
                agree = (lp_a.argmax(-1) == lp_b.argmax(-1)).float().mean().item()
                k = kl(lp_a, lp_b)
                agg[f"{key}_agree"].append(agree)
                agg[f"{key}_kl"].append(k)
                print(f"    {name:20s}: frame top-1 agreement {agree*100:6.2f}%, KL(fp||q) {k:.4f}")
                if name == "source_unigram":
                    print(f"      fp  greedy: {ctc_collapse(lp_a.argmax(-1), asr_dict)}")
                    print(f"      q   greedy: {ctc_collapse(lp_b.argmax(-1), asr_dict)}")

            mt_a = getattr(model_a, "target_unigram_decoder")
            mt_b = getattr(model_b, "target_unigram_decoder")
            eos = mt_dict.eos()
            toks_a = greedy_mt(mt_a, enc_a, eos)
            toks_b = greedy_mt(mt_b, enc_b, eos)
            detok = lambda t: "".join(mt_dict[x] for x in t).replace("▁", " ").strip()
            print(f"    MT fp greedy: {detok(toks_a)}")
            print(f"    MT q  greedy: {detok(toks_b)}")
            prev = torch.LongTensor([[eos] + toks_a]).cuda()
            tgts = torch.LongTensor(toks_a + [eos]).cuda()
            lp_a = utils.log_softmax(mt_a(prev, encoder_out=enc_a)[0][0], dim=-1)
            lp_b = utils.log_softmax(mt_b(prev, encoder_out=enc_b)[0][0], dim=-1)
            agree = (lp_a.argmax(-1) == lp_b.argmax(-1)).float().mean().item()
            k = kl(lp_a, lp_b)
            nll_a = -lp_a.gather(1, tgts[:, None]).mean().item()
            nll_b = -lp_b.gather(1, tgts[:, None]).mean().item()
            agg["mt_agree"].append(agree)
            agg["mt_kl"].append(k)
            agg["nll_a"].append(nll_a)
            agg["nll_b"].append(nll_b)
            print(f"    MT teacher-forced (on fp tokens): top-1 agreement {agree*100:.2f}%, "
                  f"KL {k:.4f}, NLL fp {nll_a:.4f} -> q {nll_b:.4f}")

        m = lambda k: sum(agg[k]) / len(agg[k])
        print(f"[avg] enc rel-err {m('enc_rel'):.4f} cos {m('enc_cos'):.4f} | "
              f"ASR-CTC agree {m('asr_agree')*100:.2f}% KL {m('asr_kl'):.4f} | "
              f"policy-CTC agree {m('st_agree')*100:.2f}% KL {m('st_kl'):.4f} | "
              f"MT agree {m('mt_agree')*100:.2f}% KL {m('mt_kl'):.4f} "
              f"NLL {m('nll_a'):.4f}->{m('nll_b'):.4f}")


if __name__ == "__main__":
    main()
