// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Build the unique upper-triangular pairwise Hamming-distance table for one fixed
// 8-cycle window of compressed scaling-factor vectors.
//
// Key I/O semantics:
// `start_i` is a one-cycle pulse that both starts a new window and captures sample 0.
// Each accepted cycle converts eight `(weight_sf + activation_sf_i - bias_i)` lane values
// into one packed `curr_vec[15:0]` by taking per-lane bits `[2:1]`.
// While a window is active, overlapping `start_i` pulses are ignored and each new sample
// updates only the newly available upper-triangular pair slots in `dist_table_o`.
//
// Timing / latency:
// The per-lane normalization, 2-bit extraction, and pairwise distance compares are
// combinational within each accepted cycle.
// `dist_table_o` and `done_o` update from registered state on `posedge clk_i`.
// `done_o` pulses for one cycle when sample 7 is captured and the 84-bit table is complete.
//
// Reset behavior:
// `rst_ni` is an asynchronous active-low reset that clears the active window state,
// vector history, done pulse, and distance table immediately.

module sf_hamming_table (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               start_i,
  input  logic        [39:0] weight_sf_vec_i,
  input  logic signed  [4:0] activation_sf_i,
  input  logic signed  [4:0] bias_i,
  output logic               done_o,
  output logic        [83:0] dist_table_o
);

  localparam int NUM_LANES   = 8;
  localparam int NUM_SAMPLES = 8;
  localparam int VEC_W       = 16;
  localparam int TABLE_W     = 84;

  logic               running_d;
  logic               running_q;
  logic         [2:0] sample_cnt_d;
  logic         [2:0] sample_cnt_q;
  logic               done_d;
  logic               done_q;
  logic        [TABLE_W-1:0] dist_table_d;
  logic        [TABLE_W-1:0] dist_table_q;
  logic        [VEC_W-1:0]   curr_vec;
  logic        [VEC_W-1:0]   vec_store_d [0:NUM_SAMPLES-1];
  logic        [VEC_W-1:0]   vec_store_q [0:NUM_SAMPLES-1];

  function automatic logic [VEC_W-1:0] build_vec_from_inputs(
    input logic [39:0]        packed_sf,
    input logic signed [4:0]  activation_sf,
    input logic signed [4:0]  bias_sf
  );
    logic signed [4:0] weight_sf;
    logic signed [6:0] norm_sf;
    logic        [VEC_W-1:0] packed_vec;
    integer             lane_idx;
    begin
      packed_vec = '0;

      for (lane_idx = 0; lane_idx < NUM_LANES; lane_idx++) begin
        weight_sf = $signed(packed_sf[lane_idx * 5 +: 5]);
        norm_sf   = $signed({{2{weight_sf[4]}}, weight_sf})
                  + $signed({{2{activation_sf[4]}}, activation_sf})
                  - $signed({{2{bias_sf[4]}}, bias_sf});
        packed_vec[lane_idx * 2 +: 2] = norm_sf[2:1];
      end

      build_vec_from_inputs = packed_vec;
    end
  endfunction

  function automatic logic [2:0] hamming_distance_sat(
    input logic [VEC_W-1:0] prev_vec,
    input logic [VEC_W-1:0] next_vec
  );
    logic   [3:0] raw_dist;
    integer       lane_idx;
    begin
      raw_dist = 4'd0;

      for (lane_idx = 0; lane_idx < NUM_LANES; lane_idx++) begin
        if (prev_vec[lane_idx * 2 +: 2] != next_vec[lane_idx * 2 +: 2]) begin
          raw_dist = raw_dist + 4'd1;
        end
      end

      if (raw_dist == 4'd8) begin
        hamming_distance_sat = 3'd7;
      end else begin
        hamming_distance_sat = raw_dist[2:0];
      end
    end
  endfunction

  function automatic integer pair_index(
    input integer older_idx,
    input integer newer_idx
  );
    integer row_idx;
    integer pair_idx;
    begin
      pair_idx = 0;

      for (row_idx = 0; row_idx < NUM_SAMPLES; row_idx++) begin
        if (row_idx < older_idx) begin
          pair_idx = pair_idx + (NUM_SAMPLES - 1 - row_idx);
        end
      end

      pair_idx = pair_idx + (newer_idx - older_idx - 1);
      pair_index = pair_idx;
    end
  endfunction

  always_comb begin
    integer vec_idx;
    integer older_idx;
    integer newer_idx;
    integer slot_idx;

    curr_vec      = build_vec_from_inputs(weight_sf_vec_i, activation_sf_i, bias_i);
    running_d     = running_q;
    sample_cnt_d  = sample_cnt_q;
    done_d        = 1'b0;
    dist_table_d  = dist_table_q;

    for (vec_idx = 0; vec_idx < NUM_SAMPLES; vec_idx++) begin
      vec_store_d[vec_idx] = vec_store_q[vec_idx];
    end

    if (!running_q) begin
      if (start_i) begin
        running_d    = 1'b1;
        sample_cnt_d = 3'd1;
        dist_table_d = '0;

        for (vec_idx = 0; vec_idx < NUM_SAMPLES; vec_idx++) begin
          vec_store_d[vec_idx] = '0;
        end

        vec_store_d[0] = curr_vec;
      end
    end else begin
      newer_idx = sample_cnt_q;
      vec_store_d[newer_idx] = curr_vec;

      for (older_idx = 0; older_idx < NUM_SAMPLES; older_idx++) begin
        if (older_idx < newer_idx) begin
          slot_idx = pair_index(older_idx, newer_idx);
          dist_table_d[slot_idx * 3 +: 3] = hamming_distance_sat(vec_store_q[older_idx], curr_vec);
        end
      end

      if (sample_cnt_q == 3'd7) begin
        running_d    = 1'b0;
        sample_cnt_d = 3'd0;
        done_d       = 1'b1;
      end else begin
        sample_cnt_d = sample_cnt_q + 3'd1;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    integer vec_idx;
    if (!rst_ni) begin
      running_q    <= 1'b0;
      sample_cnt_q <= 3'd0;
      done_q       <= 1'b0;
      dist_table_q <= '0;

      for (vec_idx = 0; vec_idx < NUM_SAMPLES; vec_idx++) begin
        vec_store_q[vec_idx] <= '0;
      end
    end else begin
      running_q    <= running_d;
      sample_cnt_q <= sample_cnt_d;
      done_q       <= done_d;
      dist_table_q <= dist_table_d;

      for (vec_idx = 0; vec_idx < NUM_SAMPLES; vec_idx++) begin
        vec_store_q[vec_idx] <= vec_store_d[vec_idx];
      end
    end
  end

  assign done_o       = done_q;
  assign dist_table_o = dist_table_q;

endmodule
