# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""MXINT4 activation fake-quantization via forward_pre_hooks.

Quantizes the *input* of every nn.Linear (along the last dim = reduction dim)
and nn.Conv1d (along the channel dim) inside the ASR+MT path, using the same
MXINT4 format as the weights: blocks of 32, shared E8M0 power-of-two scale,
INT4 elements (k/4, k in [-8,7]), round-to-nearest with saturation.

Left in FP: the two batched matmuls inside attention (QK^T and attn@V),
softmax/layernorm/GLU intermediates, and everything in the TTS path.
"""
import os

import torch
import torch.nn as nn

BLOCK = int(os.environ.get("MXINT4_BLOCK", "32"))

# comma-separated module-name prefixes exempted from activation quantization,
# e.g. "source_unigram_decoder.,ctc_target_unigram_decoder." to keep the two
# CTC heads' input activations in FP
EXEMPT = tuple(p for p in os.environ.get("MXINT4_EXEMPT", "").split(",") if p)

PREFIXES = (
    "encoder.",
    "source_unigram_decoder.",
    "ctc_target_unigram_decoder.",
    "target_unigram_decoder.",
)


def mxint4_tensor(x: torch.Tensor, dim: int, block: int = None) -> torch.Tensor:
    """Fake-quantize `x` to MXINT4 with blocks of `block` along dimension `dim`."""
    block = block or BLOCK
    orig_dtype = x.dtype
    y = x.float().movedim(dim, -1)
    shape = y.shape
    cols = shape[-1]
    pad = (block - cols % block) % block
    if pad:
        y = torch.nn.functional.pad(y, (0, pad))
    yb = y.reshape(*y.shape[:-1], -1, block)
    amax = yb.abs().amax(-1, keepdim=True)
    exp = torch.floor(torch.log2(amax.clamp(min=2.0 ** -126)))
    scale = torch.pow(2.0, exp.clamp(min=-126, max=127))
    q = torch.clamp(torch.round(yb / scale * 4.0), -8, 7)
    deq = torch.where(amax > 0, q * 0.25 * scale, torch.zeros_like(q))
    return deq.reshape(*shape[:-1], -1)[..., :cols].movedim(-1, dim).to(orig_dtype)


def _hook(dim, block):
    def pre_hook(module, inputs):
        return (mxint4_tensor(inputs[0], dim, block),) + tuple(inputs[1:])
    return pre_hook


def install_hooks(model: nn.Module, prefixes=PREFIXES, block: int = None,
                  exempt=None) -> int:
    """Register activation-quant pre-hooks; returns number of hooked modules."""
    block = block or BLOCK
    exempt = EXEMPT if exempt is None else tuple(exempt)
    n = 0
    for name, mod in model.named_modules():
        if not name.startswith(prefixes):
            continue
        if exempt and name.startswith(exempt):
            continue
        if isinstance(mod, nn.Linear):
            mod.register_forward_pre_hook(_hook(-1, block))
        elif isinstance(mod, nn.Conv1d):
            mod.register_forward_pre_hook(_hook(1, block))
        else:
            continue
        n += 1
    return n
