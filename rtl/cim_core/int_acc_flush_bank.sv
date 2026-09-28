// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Maintain the INT partial-sum front end for the hybrid accumulator and emit
// one packed flush batch when selected lanes move from the INT path to the FP path.
//
// Key I/O semantics:
// `update_i` samples one packed `pe_array` result vector plus packed scaling factors.
// `fp_flush_flag_i[lane]` flushes the old stored lane snapshot and overwrites that lane with the new input.
// The first post-clear update ignores `fp_flush_flag_i` and initializes all 8 lanes together.
//
// Timing / latency:
// INT-bank outputs reflect registered state. Flush outputs are combinational
// snapshots for the current update cycle so downstream logic can capture them
// on the same rising edge that updates the INT bank.

module int_acc_flush_bank (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        clear_i,
  input  logic        update_i,
  input  logic [87:0] result_vec_i,
  input  logic [39:0] sf_vec_i,
  input  logic  [7:0] fp_flush_flag_i,
  output logic        bank_valid_o,
  output logic [95:0] int_result_vec_o,
  output logic [39:0] int_sf_vec_o,
  output logic  [7:0] flush_mask_o,
  output logic [95:0] flush_result_vec_o,
  output logic [39:0] flush_sf_vec_o
);

  logic signed [10:0] result_lane   [0:7];
  logic        [4:0]  sf_lane       [0:7];
  logic signed [11:0] int_result_d  [0:7];
  logic signed [11:0] int_result_q  [0:7];
  logic        [4:0]  int_sf_d      [0:7];
  logic        [4:0]  int_sf_q      [0:7];
  logic               bank_valid_d;
  logic               bank_valid_q;
  logic signed [11:0] input_value;
  logic signed [11:0] aligned_input;
  logic signed [11:0] aligned_stored;
  integer             lane_idx_comb;
  integer             lane_idx_ff;

  genvar unpack_idx;
  generate
    for (unpack_idx = 0; unpack_idx < 8; unpack_idx++) begin : gen_input_unpack
      assign result_lane[unpack_idx] = $signed(result_vec_i[unpack_idx * 11 +: 11]);
      assign sf_lane[unpack_idx]     = sf_vec_i[unpack_idx * 5 +: 5];
    end
  endgenerate

  always_comb begin
    bank_valid_d     = bank_valid_q;
    flush_mask_o     = 8'h00;
    flush_result_vec_o = '0;
    flush_sf_vec_o   = '0;

    for (lane_idx_comb = 0; lane_idx_comb < 8; lane_idx_comb++) begin
      int_result_d[lane_idx_comb] = int_result_q[lane_idx_comb];
      int_sf_d[lane_idx_comb]     = int_sf_q[lane_idx_comb];
    end

    if (clear_i) begin
      bank_valid_d = 1'b0;
      for (lane_idx_comb = 0; lane_idx_comb < 8; lane_idx_comb++) begin
        int_result_d[lane_idx_comb] = 12'sd0;
        int_sf_d[lane_idx_comb]     = 5'd0;
      end
    end else if (update_i) begin
      if (!bank_valid_q) begin
        bank_valid_d = 1'b1;
        for (lane_idx_comb = 0; lane_idx_comb < 8; lane_idx_comb++) begin
          input_value                 = $signed({result_lane[lane_idx_comb][10], result_lane[lane_idx_comb]});
          int_result_d[lane_idx_comb] = input_value;
          int_sf_d[lane_idx_comb]     = sf_lane[lane_idx_comb];
        end
      end else begin
        if (fp_flush_flag_i != 8'h00) begin
          flush_mask_o = fp_flush_flag_i;
          for (lane_idx_comb = 0; lane_idx_comb < 8; lane_idx_comb++) begin
            flush_result_vec_o[lane_idx_comb * 12 +: 12] = int_result_q[lane_idx_comb];
            flush_sf_vec_o[lane_idx_comb * 5 +: 5]       = int_sf_q[lane_idx_comb];
          end
        end

        for (lane_idx_comb = 0; lane_idx_comb < 8; lane_idx_comb++) begin
          input_value    = $signed({result_lane[lane_idx_comb][10], result_lane[lane_idx_comb]});
          aligned_input  = input_value;
          aligned_stored = int_result_q[lane_idx_comb];

          if (fp_flush_flag_i[lane_idx_comb]) begin
            int_result_d[lane_idx_comb] = input_value;
            int_sf_d[lane_idx_comb]     = sf_lane[lane_idx_comb];
          end else if (sf_lane[lane_idx_comb] == int_sf_q[lane_idx_comb]) begin
            int_result_d[lane_idx_comb] = int_result_q[lane_idx_comb] + input_value;
            int_sf_d[lane_idx_comb]     = int_sf_q[lane_idx_comb];
          end else if (sf_lane[lane_idx_comb] == (int_sf_q[lane_idx_comb] + 5'd1)) begin
            aligned_input               = input_value <<< 1;
            int_result_d[lane_idx_comb] = int_result_q[lane_idx_comb] + aligned_input;
            int_sf_d[lane_idx_comb]     = int_sf_q[lane_idx_comb];
          end else if (int_sf_q[lane_idx_comb] == (sf_lane[lane_idx_comb] + 5'd1)) begin
            aligned_stored              = int_result_q[lane_idx_comb] <<< 1;
            int_result_d[lane_idx_comb] = aligned_stored + input_value;
            int_sf_d[lane_idx_comb]     = sf_lane[lane_idx_comb];
          end else begin
            int_result_d[lane_idx_comb] = int_result_q[lane_idx_comb] + input_value;
            int_sf_d[lane_idx_comb]     = int_sf_q[lane_idx_comb];
          end
        end
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      bank_valid_q <= 1'b0;
    end else begin
      bank_valid_q <= bank_valid_d;

      for (lane_idx_ff = 0; lane_idx_ff < 8; lane_idx_ff++) begin
        int_result_q[lane_idx_ff] <= int_result_d[lane_idx_ff];
        int_sf_q[lane_idx_ff]     <= int_sf_d[lane_idx_ff];
      end
    end
  end

  genvar pack_idx;
  generate
    for (pack_idx = 0; pack_idx < 8; pack_idx++) begin : gen_output_pack
      assign int_result_vec_o[pack_idx * 12 +: 12] = int_result_q[pack_idx];
      assign int_sf_vec_o[pack_idx * 5 +: 5]       = int_sf_q[pack_idx];
    end
  endgenerate

  assign bank_valid_o = bank_valid_q;

endmodule
