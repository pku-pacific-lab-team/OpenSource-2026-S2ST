# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0

"""Auto-imported when this directory is on PYTHONPATH (Python's `site` hook).

Two opt-in behaviours for the unmodified SimulEval S2ST agent, both wired in at
SentenceLevelEvaluator.__call__ time (when the agent and its vocoder exist):

  LP_RECORD_UNITS=<path.jsonl>
      Record, per utterance, the unit sequence passed to the vocoder at every
      emission, so replay_eval.py can re-synthesise the streaming output
      without re-running the speech model.

  LP_ENABLE=1  (+ LP_BLOCK / LP_ORDER / LP_THRESH / LP_MODE / LP_SHARE / LP_RIDGE)
      Install the linear-prediction residual-sparsification hooks on the
      vocoder's ResBlock convs for a full-fidelity SimulEval run; sparsity
      stats are dumped to <output>/lp_stats.json.

No-op unless one of the env vars is set.
"""
import os

if os.environ.get("LP_RECORD_UNITS") or os.environ.get("LP_ENABLE"):
    try:
        import sys
        import json

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from simuleval.evaluator import evaluator as _ev

        _orig_call = _ev.SentenceLevelEvaluator.__call__
        _orig_write_log = _ev.SentenceLevelEvaluator.write_log

        record_path = os.environ.get("LP_RECORD_UNITS")
        pending = []

        def _patched_call(self, system):
            voc = getattr(system, "vocoder", None)
            if voc is None:
                print("[lp] agent has no .vocoder attribute; nothing patched", flush=True)
                return _orig_call(self, system)

            if record_path:
                orig_fwd = voc.forward

                def rec_forward(x, dur_prediction=False):
                    pending.append(x["code"].view(-1).tolist())
                    return orig_fwd(x, dur_prediction)

                voc.forward = rec_forward
                open(record_path, "w").close()
                print(f"[lp] recording vocoder unit inputs to {record_path}", flush=True)

            if os.environ.get("LP_ENABLE"):
                from lp_sparse import LPConfig, install_hooks

                cfg = LPConfig.from_env()
                n = install_hooks(voc.model, cfg)
                print(f"[lp] installed LP hooks on {n} convs: {cfg}", flush=True)
                self._lp_model = voc.model

            return _orig_call(self, system)

        def _patched_write_log(self, instance):
            if record_path:
                with open(record_path, "a") as f:
                    f.write(json.dumps({"index": instance.index, "calls": pending}) + "\n")
                pending.clear()
            _orig_write_log(self, instance)

        _orig_dump = _ev.SentenceLevelEvaluator.dump_results

        def _patched_dump(self):
            _orig_dump(self)
            m = getattr(self, "_lp_model", None)
            if m is not None and self.output is not None:
                from lp_sparse import collect_stats

                with open(self.output / "lp_stats.json", "w") as f:
                    json.dump(collect_stats(m), f, indent=1)

        _ev.SentenceLevelEvaluator.__call__ = _patched_call
        _ev.SentenceLevelEvaluator.write_log = _patched_write_log
        _ev.SentenceLevelEvaluator.dump_results = _patched_dump
        print("[lp] SentenceLevelEvaluator patched", flush=True)
    except Exception as e:  # never break the host process
        print(f"[lp] patch failed: {e}", flush=True)
