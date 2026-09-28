// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Track the minimum signed `(min(weight_sf[0:7]) + activation_sf)` value across one
// fixed 8-cycle processing window.
//
// Key I/O semantics:
// `start_i` is a one-cycle pulse that both starts a new window and captures the first sample.
// While a window is active, the module consumes one input group per cycle for exactly 8 cycles
// and ignores any overlapping `start_i` pulses.
//
// Timing / latency:
// The 8-lane signed minimum compare and signed add are combinational within each sample cycle.
// The running minimum, active-window state, and completion pulse update on `posedge clk_i`.
// `done_o` pulses for one cycle when the 8th sample is captured.

module sf_min_tracker (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               start_i,
  input  logic        [39:0] weight_sf_vec_i,
  input  logic signed  [4:0] activation_sf_i,
  output logic               done_o,
  output logic signed  [4:0] min_sf_o
);

  logic               running_d;
  logic               running_q;
  logic         [2:0] sample_cnt_d;
  logic         [2:0] sample_cnt_q;
  logic               done_d;
  logic               done_q;
  logic signed  [4:0] min_sf_d;
  logic signed  [4:0] min_sf_q;
  logic signed  [4:0] weight_min;
  logic signed  [5:0] candidate_sf_ext;
  logic signed  [4:0] candidate_sf;

  function automatic logic signed [4:0] min_weight_from_vec(input logic [39:0] packed_sf);
    logic signed [4:0] current_min;
    logic signed [4:0] lane_value;
    integer            lane_idx;
    begin
      current_min = $signed(packed_sf[4:0]);

      for (lane_idx = 1; lane_idx < 8; lane_idx++) begin
        lane_value = $signed(packed_sf[lane_idx * 5 +: 5]);
        if (lane_value < current_min) begin
          current_min = lane_value;
        end
      end

      min_weight_from_vec = current_min;
    end
  endfunction

  always_comb begin
    weight_min        = min_weight_from_vec(weight_sf_vec_i);
    candidate_sf_ext  = $signed({weight_min[4], weight_min}) + $signed({activation_sf_i[4], activation_sf_i});
    candidate_sf      = candidate_sf_ext[4:0];

    running_d         = running_q;
    sample_cnt_d      = sample_cnt_q;
    done_d            = 1'b0;
    min_sf_d          = min_sf_q;

    if (!running_q) begin
      if (start_i) begin
        running_d    = 1'b1;
        sample_cnt_d = 3'd1;
        min_sf_d     = candidate_sf;
      end
    end else begin
      if (candidate_sf < min_sf_q) begin
        min_sf_d = candidate_sf;
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
    if (!rst_ni) begin
      running_q    <= 1'b0;
      sample_cnt_q <= 3'd0;
      done_q       <= 1'b0;
      min_sf_q     <= 5'sd0;
    end else begin
      running_q    <= running_d;
      sample_cnt_q <= sample_cnt_d;
      done_q       <= done_d;
      min_sf_q     <= min_sf_d;
    end
  end

  assign done_o   = done_q;
  assign min_sf_o = min_sf_q;

endmodule
