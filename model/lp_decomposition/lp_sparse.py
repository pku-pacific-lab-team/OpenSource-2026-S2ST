# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Block-wise linear prediction + residual sparsification of conv input features.

For a feature map x[C, T] (one utterance, channels x time):
  1. Split T into blocks of `block` steps.
  2. Inside each block fit an `order`-th order linear predictor per channel
     (or one shared across channels with `share=True`) via the autocorrelation
     (Yule-Walker) equations:  x[t] ~ sum_{k=1..order} a_k * x[t-k].
  3. For t >= order within the block, form the prediction x_hat[t] from the
     original previous `order` features (open loop) and the residual
     e[t] = x[t] - x_hat[t]. The first `order` steps of a block have no
     prediction and are passed through untouched.
  4. Zero every residual with |e| < thresh (`mode='abs'`), or
     |e| < thresh * rms(block, channel) (`mode='rel'`).
  5. Feed x_hat + sparsified residual to the conv.

`install_hooks` puts this as a forward pre-hook on every Conv1d inside the
HiFi-GAN ResBlocks (96.5% of vocoder FLOPs) and accumulates sparsity stats.
"""
import math
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LPConfig:
    block: int = 64
    order: int = 4
    thresh: float = 0.0
    mode: str = "abs"  # abs | rel
    share: bool = False  # one coefficient set per block shared across channels
    ridge: float = 1e-4  # white-noise correction added to r[0] (relative)
    layers: str = "resblock"  # resblock | all
    closed_loop: bool = False  # predict from reconstructed (not original) past features

    @classmethod
    def from_env(cls):
        return cls(
            block=int(os.environ.get("LP_BLOCK", 64)),
            order=int(os.environ.get("LP_ORDER", 4)),
            thresh=float(os.environ.get("LP_THRESH", 0.0)),
            mode=os.environ.get("LP_MODE", "abs"),
            share=os.environ.get("LP_SHARE", "0") == "1",
            ridge=float(os.environ.get("LP_RIDGE", 1e-4)),
            layers=os.environ.get("LP_LAYERS", "resblock"),
            closed_loop=os.environ.get("LP_CLOSED", "0") == "1",
        )


def yule_walker(r, ridge):
    """r: [..., order+1] autocorrelation -> a: [..., order] predictor coefficients."""
    order = r.shape[-1] - 1
    idx = torch.arange(order, device=r.device)
    R = r[..., (idx[:, None] - idx[None, :]).abs()]  # Toeplitz [..., order, order]
    r0 = r[..., :1]
    eye = torch.eye(order, device=r.device, dtype=r.dtype)
    R = R + eye * (ridge * r0 + 1e-12).unsqueeze(-1)
    a = torch.linalg.solve(R, r[..., 1:].unsqueeze(-1)).squeeze(-1)
    return torch.where(r0 > 0, a, torch.zeros_like(a))


def lp_sparse(x, cfg: LPConfig, stats=None):
    """x: [B, C, T] -> same shape. Optionally accumulates counts into `stats`."""
    B, C, T = x.shape
    order, block = cfg.order, cfg.block
    if T <= order or block <= order:
        return x
    nb = math.ceil(T / block)
    pad = nb * block - T
    xb = (F.pad(x, (0, pad)) if pad else x).view(B, C, nb, block)

    r = torch.stack(
        [(xb[..., k:] * xb[..., : block - k]).sum(-1) for k in range(order + 1)], -1
    )  # [B, C, nb, order+1]
    if cfg.share:
        r = r.sum(1, keepdim=True)
    a = yule_walker(r, cfg.ridge)  # [B, C|1, nb, order]

    if cfg.mode == "abs":
        thr = cfg.thresh
    elif cfg.mode == "rel":
        thr = cfg.thresh * xb.pow(2).mean(-1, keepdim=True).sqrt()
    else:
        raise ValueError(cfg.mode)

    if cfg.closed_loop:
        # DPCM-style: each step predicts from the already reconstructed past, so the
        # receiver can rebuild y from (a, sparse residual) alone and errors don't accumulate.
        a_flip = a.flip(-1)  # a_flip[..., m] multiplies y[t-order+m]
        y = xb.clone()
        keeps = []
        thr_t = thr if cfg.mode == "abs" else thr.squeeze(-1)
        for t in range(order, block):
            xhat_t = (y[..., t - order : t] * a_flip).sum(-1)
            e_t = xb[..., t] - xhat_t
            keep_t = e_t.abs() >= thr_t
            y[..., t] = xhat_t + e_t * keep_t
            keeps.append(keep_t)
        keep = torch.stack(keeps, -1)  # [B, C, nb, block-order]
        out = y.view(B, C, nb * block)[..., :T]
    else:
        # win[..., j, :] = xb[..., j : j+order]; it predicts xb[..., j+order]
        win = xb.unfold(-1, order, 1)[..., : block - order, :]
        xhat = (win * a.flip(-1).unsqueeze(-2)).sum(-1)  # [B, C, nb, block-order]
        res = xb[..., order:] - xhat
        keep = res.abs() >= thr
        out = torch.cat([xb[..., :order], xhat + res * keep], -1).view(B, C, nb * block)[..., :T]

    if stats is not None:
        pos = torch.arange(nb * block, device=x.device).view(nb, block)[:, order:]
        valid = pos < T  # [nb, block-order]; drop zero-padded tail positions
        n_pred = int(valid.sum()) * B * C
        n_zero = int(((~keep) & valid).sum())
        stats["pred"] += n_pred
        stats["zero"] += n_zero
        stats["total"] += B * C * T
        stats["calls"] += 1
        # MACs of this conv call, and the share of them whose input residual is zero
        macs = stats["_macs_per_t"] * T
        stats["macs"] += macs
        stats["macs_zero"] += macs * n_zero / (B * C * T)
    return out


def _new_stats(conv):
    return {
        "pred": 0, "zero": 0, "total": 0, "calls": 0, "macs": 0.0, "macs_zero": 0.0,
        "_macs_per_t": conv.in_channels * conv.out_channels * conv.kernel_size[0] / conv.groups,
    }


def install_hooks(generator: nn.Module, cfg: LPConfig):
    """Attach LP pre-hooks to the generator's convs. Returns number of hooked convs."""
    remove_hooks(generator)
    n = 0
    for name, m in generator.named_modules():
        if not isinstance(m, nn.Conv1d):
            continue
        if name.startswith("dur_predictor"):
            continue
        if cfg.layers == "resblock" and not name.startswith("resblocks."):
            continue
        stats = _new_stats(m)
        m._lp_stats = stats
        m._lp_name = name

        def pre_hook(mod, inputs, _stats=stats):
            return (lp_sparse(inputs[0], cfg, _stats),) + tuple(inputs[1:])

        m._lp_handle = m.register_forward_pre_hook(pre_hook)
        n += 1
    generator._lp_cfg = cfg
    return n


def remove_hooks(generator: nn.Module):
    for m in generator.modules():
        h = getattr(m, "_lp_handle", None)
        if h is not None:
            h.remove()
            del m._lp_handle
            del m._lp_stats
            del m._lp_name


def collect_stats(generator: nn.Module):
    """Aggregate per-layer counters into per-stage and overall sparsity numbers."""
    per_layer = {}
    for m in generator.modules():
        s = getattr(m, "_lp_stats", None)
        if s is not None:
            per_layer[m._lp_name] = {k: v for k, v in s.items() if not k.startswith("_")}

    def agg(entries):
        pred = sum(e["pred"] for e in entries)
        zero = sum(e["zero"] for e in entries)
        total = sum(e["total"] for e in entries)
        macs = sum(e["macs"] for e in entries)
        macs_zero = sum(e["macs_zero"] for e in entries)
        return {
            "residual_sparsity": zero / pred if pred else 0.0,  # among predicted positions
            "sparsity_incl_unpredicted": zero / total if total else 0.0,  # over all positions
            "mac_weighted_sparsity": macs_zero / macs if macs else 0.0,
            "gmacs": macs / 1e9,
            "n_layers": len(entries),
            "counts": {"pred": pred, "zero": zero, "total": total, "macs": macs, "macs_zero": macs_zero},
        }

    stages = {}
    for name, e in per_layer.items():
        key = name.split(".")[0]
        if name.startswith("resblocks."):
            key = f"resblock_stage{int(name.split('.')[1]) // 3}"
        stages.setdefault(key, []).append(e)
    out = {
        "config": asdict(generator._lp_cfg) if hasattr(generator, "_lp_cfg") else None,
        "overall": agg(list(per_layer.values())),
        "by_group": {k: agg(v) for k, v in sorted(stages.items())},
    }
    return out
