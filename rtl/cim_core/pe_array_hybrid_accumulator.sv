// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Integrate `pe_array`, the INT flush bank, the sparse flush serializer, and the BF16 bank
// into one hybrid post-processing accumulator.
//
// Key I/O semantics:
// `compute_accumulate_i` captures the current `pe_array` result vector into the INT bank,
// optionally flushing selected old INT lanes into the serialized FP path.
// `finalize_i` injects the residual INT-bank contents into that same serialized FP path
// once the serializer queue is empty, so no second BF16 merge datapath is needed.
// `fp_path_busy_o` stays high while queued or newly injected FP work is still draining.
// `fp_result_o[lane]` exposes the stored BF16 accumulation result for every lane in parallel.
//
// Timing / latency:
// `pe_array` remains combinational from the programmed weights and current activations.
// INT-bank updates, serializer queue updates, and FP accumulation all occur on `posedge clk_i`.
// The internal serializer request outputs are registered and are intended for
// one-request-per-cycle consumption by the FP bank.

module pe_array_hybrid_accumulator (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               clear_i,
  input  logic               weight_we_i,
  input  logic         [4:0] weight_addr_i,
  input  logic        [31:0] weight_row_i,
  input  logic        [15:0] row_weight_rsel_i,
  input  logic        [31:0] activation_vec_i,
  input  logic        [39:0] sf_vec_i,
  input  logic         [7:0] fp_flush_flag_i,
  input  logic               compute_accumulate_i,
  input  logic               finalize_i,
  output logic               fp_path_busy_o,
  output logic [7:0][15:0]   fp_result_o
);

  logic        [87:0] pe_result_vec;

  logic               int_bank_valid;
  logic        [95:0] int_bank_result_vec;
  logic        [39:0] int_bank_sf_vec;
  logic         [7:0] int_bank_flush_mask;
  logic        [95:0] int_bank_flush_result_vec;
  logic        [39:0] int_bank_flush_sf_vec;

  logic               serializer_enqueue;
  logic         [7:0] serializer_flush_mask;
  logic        [95:0] serializer_flush_result_vec;
  logic        [39:0] serializer_flush_sf_vec;
  logic               finalize_accept;
  logic               serializer_flush_valid;
  logic         [2:0] serializer_flush_lane_idx;
  logic signed [11:0] serializer_flush_result;
  logic         [4:0] serializer_flush_sf;

  pe_array pe_array_u (
    .clk_i             (clk_i),
    .weight_we_i       (weight_we_i),
    .weight_addr_i     (weight_addr_i),
    .weight_row_i      (weight_row_i),
    .activation_vec_i  (activation_vec_i),
    .row_weight_rsel_i (row_weight_rsel_i),
    .result_vec_o      (pe_result_vec)
  );

  int_acc_flush_bank int_bank_u (
    .clk_i             (clk_i),
    .rst_ni            (rst_ni),
    .clear_i           (clear_i),
    .update_i          (compute_accumulate_i),
    .result_vec_i      (pe_result_vec),
    .sf_vec_i          (sf_vec_i),
    .fp_flush_flag_i   (fp_flush_flag_i),
    .bank_valid_o      (int_bank_valid),
    .int_result_vec_o  (int_bank_result_vec),
    .int_sf_vec_o      (int_bank_sf_vec),
    .flush_mask_o      (int_bank_flush_mask),
    .flush_result_vec_o(int_bank_flush_result_vec),
    .flush_sf_vec_o    (int_bank_flush_sf_vec)
  );

  flush_serializer flush_serializer_u (
    .clk_i             (clk_i),
    .rst_ni            (rst_ni),
    .clear_i           (clear_i),
    .enqueue_i         (serializer_enqueue),
    .flush_mask_i      (serializer_flush_mask),
    .flush_result_vec_i(serializer_flush_result_vec),
    .flush_sf_vec_i    (serializer_flush_sf_vec),
    .flush_valid_o     (serializer_flush_valid),
    .flush_lane_idx_o  (serializer_flush_lane_idx),
    .flush_result_o    (serializer_flush_result),
    .flush_sf_o        (serializer_flush_sf)
  );

  fp_accumulate_bank fp_bank_u (
    .clk_i            (clk_i),
    .clear_i          (clear_i),
    .flush_valid_i    (serializer_flush_valid),
    .flush_lane_idx_i (serializer_flush_lane_idx),
    .flush_result_i   (serializer_flush_result),
    .flush_sf_i       (serializer_flush_sf),
    .fp_result_o      (fp_result_o)
  );

  assign finalize_accept          = finalize_i && !serializer_flush_valid;
  assign serializer_flush_mask    = finalize_accept
    ? (int_bank_valid ? 8'hFF : 8'h00)
    : int_bank_flush_mask;
  assign serializer_flush_result_vec = finalize_accept
    ? int_bank_result_vec
    : int_bank_flush_result_vec;
  assign serializer_flush_sf_vec = finalize_accept
    ? int_bank_sf_vec
    : int_bank_flush_sf_vec;
  assign serializer_enqueue = (serializer_flush_mask != 8'h00);
  assign fp_path_busy_o     = serializer_flush_valid || serializer_enqueue;

endmodule
