# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Auto-imported when this directory is on PYTHONPATH (Python's `site` hook).

Two independent, env-gated model patches applied at load time by wrapping
fairseq.checkpoint_utils.load_model_ensemble (no agent code changes):

- MXINT4_W4A4=1: install MXINT4 activation fake-quant hooks on the ASR+MT path
  (see mx_act_quant.py; MXINT4_BLOCK sets group size, MXINT4_EXEMPT exempts
  module-name prefixes).
- MODEL_DTYPE=bf16|fp16: convert the whole model to that dtype and cast the
  float32 fbank features entering the first subsample conv to match.

No-op unless at least one env var is set.
"""
import os

_W4A4 = os.environ.get("MXINT4_W4A4")
_DTYPE = os.environ.get("MODEL_DTYPE", "")

if _W4A4 or _DTYPE:
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import torch
        import fairseq.checkpoint_utils as _cu

        _orig = _cu.load_model_ensemble

        def _patched(*args, **kwargs):
            models, cfg = _orig(*args, **kwargs)
            for m in models:
                if _DTYPE:
                    td = {"bf16": torch.bfloat16, "fp16": torch.float16}[_DTYPE]
                    m.to(td)

                    def _cast(module, inputs, _td=td):
                        return tuple(
                            x.to(_td) if torch.is_tensor(x) and x.is_floating_point() else x
                            for x in inputs
                        )

                    m.encoder.subsample.conv_layers[0].register_forward_pre_hook(_cast)
                    print(f"[model-dtype] converted model to {_DTYPE}", flush=True)
                if _W4A4:
                    from mx_act_quant import install_hooks, BLOCK, EXEMPT
                    n = install_hooks(m)
                    print(f"[mxint4-w4a4] installed {n} activation-quant hooks "
                          f"(block={BLOCK}, exempt={list(EXEMPT)})", flush=True)
            return models, cfg

        _cu.load_model_ensemble = _patched
        print("[sitecustomize] load_model_ensemble patched", flush=True)
    except Exception as e:  # never break the host process
        print(f"[sitecustomize] patch failed: {e}", flush=True)
