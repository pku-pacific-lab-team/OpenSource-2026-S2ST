# OpenSource-2026-S2ST

Selected RTL, model evaluation scripts, and measured power data for a synthesizable compute-in-memory accelerator for wearable speech-to-speech translation.

The computing cluster, `pe_array_group`, contains four synthesizable CIM cores sharing a group reordering engine. Each CIM core contains an 8x8 array with four 4b weight slots per multiplier, a feature fusing unit, per-column INT accumulators with a shared BF16 accumulator, and MXINT4 writeback. The group reordering engine compresses scaling factors into 2b bins, builds a Hamming-distance table, and reorders quantization groups to increase INT accumulations. The probability cache, `prob_cache`, accumulates translated-token probabilities across candidate sequences and queues probability-qualified tokens for speculative speech generation.

| Directory | Contents |
|---|---|
| `rtl/cluster/` | Computing cluster top level and buffer address mapping |
| `rtl/cim_core/` | CIM array, feature fusing, hierarchical accumulation, and quantization |
| `rtl/gre/` | Group reordering engine |
| `rtl/prob_cache/` | Probability cache for speculative speech generation |
| `rtl/mem/` | Behavioral SRAM models |
| `tb/` | Self-checking testbenches |
| `sim/run_tests.sh` | Regression script for VCS, Vivado xsim, or Icarus Verilog |
| `model/` | LP decomposition, MXINT4 quantization, group reordering, and speculative speech generation evaluation on StreamSpeech |
| `data/power/` | Measured voltage-frequency power sweep and workload-dependent power |
| `docs/notes.md` | Power rail mapping and data definitions |

RTL and testbenches are provided under `Apache-2.0 WITH SHL-2.1` (Solderpad Hardware License v2.1). See `LICENSE` and `LICENSE.SHL`. Scripts under `model/` are provided under `Apache-2.0`; see `model/README.md`. Data is provided under CC BY 4.0 (`CC-BY-4.0`); see `data/LICENSE`.

Copyright 2026 School of Integrated Circuits, Peking University
