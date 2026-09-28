// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Maintain 8 BF16 accumulation lanes for serialized flush requests and expose all lane totals in parallel.
//
// Key I/O semantics:
// `flush_valid_i` applies one INT12-plus-scale flush request to the addressed BF16 lane.
// `fp_result_o[lane]` reflects the stored BF16 total for that lane; no explicit read control is required.
//
// Timing / latency:
// Flush accumulation updates occur on `posedge clk_i`.
// `fp_result_o` changes when the registered BF16 lane state changes.

module fp_accumulate_bank (
  input  logic               clk_i,
  input  logic               clear_i,
  input  logic               flush_valid_i,
  input  logic         [2:0] flush_lane_idx_i,
  input  logic signed [11:0] flush_result_i,
  input  logic         [4:0] flush_sf_i,
  output logic [7:0][15:0]   fp_result_o
);

  logic [7:0][15:0] fp_acc_d;
  logic [7:0][15:0] fp_acc_q;
  logic      [15:0] converted_bf16;
  logic      [15:0] selected_fp_acc;
  logic      [15:0] accumulated_bf16;
  integer           lane_idx_comb;
  integer           lane_idx_ff;

  int12_scale_to_bf16 converter (
    .int12_i         (flush_result_i),
    .scaling_factor_i($signed(flush_sf_i)),
    .bf16_o          (converted_bf16)
  );

  bf16_add adder (
    .a_i  (selected_fp_acc),
    .b_i  (converted_bf16),
    .sum_o(accumulated_bf16)
  );

  always_comb begin
    selected_fp_acc = fp_acc_q[flush_lane_idx_i];

    for (lane_idx_comb = 0; lane_idx_comb < 8; lane_idx_comb++) begin
      fp_acc_d[lane_idx_comb] = fp_acc_q[lane_idx_comb];
    end

    if (clear_i) begin
      for (lane_idx_comb = 0; lane_idx_comb < 8; lane_idx_comb++) begin
        fp_acc_d[lane_idx_comb] = 16'h0000;
      end
    end else if (flush_valid_i) begin
      fp_acc_d[flush_lane_idx_i] = accumulated_bf16;
    end
  end

  always_ff @(posedge clk_i) begin
    for (lane_idx_ff = 0; lane_idx_ff < 8; lane_idx_ff++) begin
      fp_acc_q[lane_idx_ff] <= fp_acc_d[lane_idx_ff];
    end
  end

  assign fp_result_o = fp_acc_q;

endmodule
