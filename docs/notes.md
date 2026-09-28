# Data Notes

## Power Rails

| Channel | Supply |
|---|---|
| CH1 | S2ST NPU (accelerator) |
| CH2 | CPU |
| CH3 | IO |

Power is in mW, voltage in V, and frequency in MHz. `dynamic` = `active` - `static`. `total_ch1_ch2` sums the NPU and CPU rails.

## `data/power/vf_power.csv`

Voltage-frequency sweep of the S2ST NPU. NPU and CPU rails are set to `target_voltage_v`.

## `data/power/workload_power.csv`

NPU power at 0.8 V for different input statistics. `weight_zero_percent` and `activation_zero_percent` are the percentages of zero weights and activations. `scaling_factor_mode` is `zero` when all scaling factors are 0 and `random_-2_to_2` when they are random in [-2, 2].

Each `active` row is paired with the `static` row named in `paired_static_file`; `active_minus_static_*` is the difference of their means. `sample_count` and `excluded_sample_count` give the number of power samples averaged and discarded. `source_file` and `paired_static_file` identify the raw measurement logs.
