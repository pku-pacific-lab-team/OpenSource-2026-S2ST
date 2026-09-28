# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Fake-quantize the ASR + translation path of a StreamSpeech checkpoint to MXINT4.

MXINT4 (OCP Microscaling style): blocks of 32 values along the reduction dim
share one E8M0 power-of-two scale; each element is a 4-bit two's-complement
value with 2 fraction bits (representable values k/4, k in [-8, 7]).
Direct-cast round-to-nearest with saturation.

Quantized components (ASR + MT path only):
  encoder.*                       chunk-causal Conformer (shared trunk)
  source_unigram_decoder.proj     ASR CTC head
  ctc_target_unigram_decoder.proj policy CTC head
  target_unigram_decoder.*        MT decoder
Left in FP (TTS path): synthesizer_encoder.*, decoder.* (unit CTC), vocoder.

Usage:
  python <this repo>/model/quantization/mxint4_quantize_ckpt.py --scope all  --out <path>
  python <this repo>/model/quantization/mxint4_quantize_ckpt.py --scope body --out <path>

scope=all  : every matmul/conv weight in the ASR+MT path, incl. embeddings,
             vocab output projections and CTC heads.
scope=body : transformer/conformer body only; embeddings, vocab projections
             and CTC heads stay FP.
"""
import argparse
import math
from collections import defaultdict

import torch

CKPT = "pretrain_models/streamspeech.simultaneous.fr-en.pt"
BLOCK = 32

ASR_MT_PREFIXES = (
    "encoder.",
    "source_unigram_decoder.",
    "ctc_target_unigram_decoder.",
    "target_unigram_decoder.",
)
# tensors that are not matmul/conv weights
SKIP_SUBSTR = ("layer_norm", "batch_norm", "pos_bias", "version")
# vocab-facing tensors excluded under scope=body
VOCAB_SUBSTR = (
    "embed_tokens",
    "output_projection",
    "source_unigram_decoder.proj",
    "ctc_target_unigram_decoder.proj",
)


def mxint4(w: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Fake-quantize a weight tensor to MXINT4, blocking along the reduction dim."""
    orig_shape, orig_dtype = w.shape, w.dtype
    x = w.float().reshape(orig_shape[0], -1)  # (out, in*k) for conv, (out, in) for linear
    rows, cols = x.shape
    pad = (block - cols % block) % block
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    xb = x.view(rows, -1, block)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    exp = torch.floor(torch.log2(amax.clamp(min=2.0 ** -126)))
    scale = torch.pow(2.0, exp.clamp(min=-126, max=127))
    q = torch.clamp(torch.round(xb / scale * 4.0), -8, 7)
    deq = q * 0.25 * scale
    deq = torch.where(amax > 0, deq, torch.zeros_like(deq))
    return deq.view(rows, -1)[:, :cols].reshape(orig_shape).to(orig_dtype)


def group_of(key: str) -> str:
    if key.startswith("encoder.subsample"):
        return "encoder.subsample (conv)"
    if key.startswith("encoder.conformer_layers"):
        if "conv_module" in key:
            return "encoder.conformer conv_module"
        if "self_attn" in key:
            return "encoder.conformer self_attn"
        return "encoder.conformer ffn"
    if key.startswith("encoder."):
        return "encoder.linear"
    if key.startswith("target_unigram_decoder."):
        if any(s in key for s in ("embed_tokens", "output_projection")):
            return "MT decoder embed/out_proj"
        return "MT decoder body"
    return "CTC heads (ASR + policy)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["all", "body"], default="all")
    ap.add_argument("--block", type=int, default=BLOCK, help="MX group size")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"]

    stats = defaultdict(lambda: [0, 0, 0.0, 0.0])  # tensors, params, sum_sq_err, sum_sq_sig
    total, quantized = 0, 0
    for k in list(sd.keys()):
        v = sd[k]
        total += v.numel()
        if not k.startswith(ASR_MT_PREFIXES):
            continue
        if not k.endswith(".weight") or v.ndim < 2:
            continue
        if any(s in k for s in SKIP_SUBSTR):
            continue
        if args.scope == "body" and any(s in k for s in VOCAB_SUBSTR):
            continue
        vq = mxint4(v, block=args.block)
        g = stats[group_of(k)]
        g[0] += 1
        g[1] += v.numel()
        g[2] += (vq.float() - v.float()).pow(2).sum().item()
        g[3] += v.float().pow(2).sum().item()
        sd[k] = vq
        quantized += v.numel()

    bits = 4 + 8 / args.block
    print(f"scope={args.scope}  quantized {quantized/1e6:.2f}M / {total/1e6:.2f}M params "
          f"({100*quantized/total:.1f}%)  block={args.block}, ~{bits:.2f} bits/weight")
    print(f"{'group':35s} {'tensors':>7s} {'params':>9s} {'rel RMS err':>12s} {'SQNR dB':>8s}")
    for name in sorted(stats):
        n, p, se, ss = stats[name]
        rel = math.sqrt(se / ss) if ss else 0.0
        sqnr = 10 * math.log10(ss / se) if se else float("inf")
        print(f"{name:35s} {n:7d} {p/1e6:8.2f}M {rel:12.4f} {sqnr:8.2f}")

    # keep only what inference needs; the optimizer state is ~560MB dead weight
    ckpt.pop("last_optimizer_state", None)
    torch.save(ckpt, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
