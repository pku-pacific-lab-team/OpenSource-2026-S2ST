// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Integrate `sf_min_tracker`, `sf_hamming_table`, and `sf_reorder_engine`
// into one small top-level wrapper that runs the sparse-factor reorder flow
// without adding a wrapper-level 8-sample input buffer.
//
// Key I/O semantics:
// `start_i` is accepted only while idle and marks sample 0 of the first pass.
// The external source must then provide 8 first-pass samples for
// `sf_min_tracker`, followed immediately by the same 8 samples replayed for
// `sf_hamming_table`.
//
// Timing / latency:
// The wrapper uses four control states: `IDLE`, `RUN_MIN`, `RUN_TABLE`, and
// `RUN_REORDER`. The hamming-table stage starts on replay sample 0, and the
// reorder stage starts one cycle after `sf_hamming_table.done_o`.
//
// Reset behavior:
// `rst_ni` is an asynchronous active-low reset that returns the wrapper to
// `IDLE`, clears the pass counter, and aborts any in-flight multi-stage
// operation immediately.

module sf_reorder_top (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               start_i,
  input  logic        [39:0] weight_sf_vec_i,
  input  logic signed  [4:0] activation_sf_i,
  input  logic         [2:0] threshold_i,
  input  logic         [2:0] max_iter_i,
  output logic               done_o,
  output logic        [23:0] seq_o
);

  localparam int NUM_SAMPLES = 8;

  localparam logic [1:0] STATE_IDLE        = 2'd0;
  localparam logic [1:0] STATE_RUN_MIN     = 2'd1;
  localparam logic [1:0] STATE_RUN_TABLE   = 2'd2;
  localparam logic [1:0] STATE_RUN_REORDER = 2'd3;

  logic         [1:0] state_d;
  logic         [1:0] state_q;
  logic         [2:0] pass_cnt_d;
  logic         [2:0] pass_cnt_q;
  logic               min_start;
  logic               table_start;
  logic               reorder_start;
  logic               min_done;
  logic               table_done;
  logic               reorder_done;
  logic signed  [4:0] min_sf;
  logic        [83:0] dist_table;
  logic        [23:0] seq;

  sf_min_tracker u_min_tracker (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .start_i        (min_start),
    .weight_sf_vec_i(weight_sf_vec_i),
    .activation_sf_i(activation_sf_i),
    .done_o         (min_done),
    .min_sf_o       (min_sf)
  );

  sf_hamming_table u_hamming_table (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .start_i        (table_start),
    .weight_sf_vec_i(weight_sf_vec_i),
    .activation_sf_i(activation_sf_i),
    .bias_i         (min_sf),
    .done_o         (table_done),
    .dist_table_o   (dist_table)
  );

  sf_reorder_engine u_reorder_engine (
    .clk_i       (clk_i),
    .rst_ni      (rst_ni),
    .start_i     (reorder_start),
    .dist_table_i(dist_table),
    .threshold_i (threshold_i),
    .max_iter_i  (max_iter_i),
    .done_o      (reorder_done),
    .seq_o       (seq)
  );

  always_comb begin
    state_d       = state_q;
    pass_cnt_d    = pass_cnt_q;
    min_start     = 1'b0;
    table_start   = 1'b0;
    reorder_start = 1'b0;

    case (state_q)
      STATE_IDLE: begin
        pass_cnt_d = 3'd0;
        if (start_i) begin
          min_start  = 1'b1;
          state_d    = STATE_RUN_MIN;
          pass_cnt_d = 3'd1;
        end
      end

      STATE_RUN_MIN: begin
        if (pass_cnt_q == 3'd7) begin
          state_d    = STATE_RUN_TABLE;
          pass_cnt_d = 3'd0;
        end else begin
          pass_cnt_d = pass_cnt_q + 3'd1;
        end
      end

      STATE_RUN_TABLE: begin
        if (pass_cnt_q == 3'd0) begin
          table_start = 1'b1;
        end

        if (pass_cnt_q != 3'd7) begin
          pass_cnt_d = pass_cnt_q + 3'd1;
        end else begin
          pass_cnt_d = 3'd7;
        end

        if (table_done) begin
          reorder_start = 1'b1;
          state_d       = STATE_RUN_REORDER;
          pass_cnt_d    = 3'd0;
        end
      end

      STATE_RUN_REORDER: begin
        pass_cnt_d = 3'd0;
        if (reorder_done) begin
          state_d = STATE_IDLE;
        end
      end

      default: begin
        state_d    = STATE_IDLE;
        pass_cnt_d = 3'd0;
      end
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q    <= STATE_IDLE;
      pass_cnt_q <= 3'd0;
    end else begin
      state_q    <= state_d;
      pass_cnt_q <= pass_cnt_d;
    end
  end

  assign done_o = reorder_done;
  assign seq_o  = seq;

endmodule
