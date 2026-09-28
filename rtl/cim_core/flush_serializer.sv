// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Compress packed flush batches into a sparse lane-request queue and emit one request per cycle.
//
// Key I/O semantics:
// `enqueue_i` samples one packed flush snapshot, but only lanes with `flush_mask_i[lane] == 1`
// are stored in the internal queue.
// `flush_valid_o` exposes the oldest queued lane request in FIFO order, with lane ordering
// preserved as `lane0 -> lane7` within each enqueued batch.
//
// Timing / latency:
// New requests are queued on `posedge clk_i`. Outputs reflect the registered queue head after
// each rising edge, so downstream logic can sample one serialized request per cycle.

module flush_serializer #(
  parameter int REQUEST_QUEUE_DEPTH = 8
) (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               clear_i,
  input  logic               enqueue_i,
  input  logic         [7:0] flush_mask_i,
  input  logic        [95:0] flush_result_vec_i,
  input  logic        [39:0] flush_sf_vec_i,
  output logic               flush_valid_o,
  output logic         [2:0] flush_lane_idx_o,
  output logic signed [11:0] flush_result_o,
  output logic         [4:0] flush_sf_o
);

  localparam int PTR_W = (REQUEST_QUEUE_DEPTH <= 1) ? 1 : $clog2(REQUEST_QUEUE_DEPTH);

  logic         [2:0] queue_lane_idx_d [0:REQUEST_QUEUE_DEPTH-1];
  logic         [2:0] queue_lane_idx_q [0:REQUEST_QUEUE_DEPTH-1];
  logic signed [11:0] queue_result_d   [0:REQUEST_QUEUE_DEPTH-1];
  logic signed [11:0] queue_result_q   [0:REQUEST_QUEUE_DEPTH-1];
  logic         [4:0] queue_sf_d       [0:REQUEST_QUEUE_DEPTH-1];
  logic         [4:0] queue_sf_q       [0:REQUEST_QUEUE_DEPTH-1];

  logic [PTR_W-1:0]   queue_head_d;
  logic [PTR_W-1:0]   queue_head_q;
  logic [PTR_W-1:0]   queue_tail_d;
  logic [PTR_W-1:0]   queue_tail_q;
  logic [PTR_W:0]     queue_count_d;
  logic [PTR_W:0]     queue_count_q;

  integer             queue_idx_comb;
  integer             queue_idx_ff;
  integer             lane_idx_comb;

  function automatic logic [PTR_W-1:0] inc_ptr(input logic [PTR_W-1:0] ptr);
    begin
      if (ptr == (REQUEST_QUEUE_DEPTH - 1)) begin
        inc_ptr = '0;
      end else begin
        inc_ptr = ptr + 1'b1;
      end
    end
  endfunction

  always_comb begin
    queue_head_d  = queue_head_q;
    queue_tail_d  = queue_tail_q;
    queue_count_d = queue_count_q;

    for (queue_idx_comb = 0; queue_idx_comb < REQUEST_QUEUE_DEPTH; queue_idx_comb++) begin
      queue_lane_idx_d[queue_idx_comb] = queue_lane_idx_q[queue_idx_comb];
      queue_result_d[queue_idx_comb]   = queue_result_q[queue_idx_comb];
      queue_sf_d[queue_idx_comb]       = queue_sf_q[queue_idx_comb];
    end

    if (clear_i) begin
      queue_head_d  = '0;
      queue_tail_d  = '0;
      queue_count_d = '0;

      for (queue_idx_comb = 0; queue_idx_comb < REQUEST_QUEUE_DEPTH; queue_idx_comb++) begin
        queue_lane_idx_d[queue_idx_comb] = 3'd0;
        queue_result_d[queue_idx_comb]   = 12'sd0;
        queue_sf_d[queue_idx_comb]       = 5'd0;
      end
    end else begin
      if (queue_count_q != '0) begin
        queue_lane_idx_d[queue_head_d] = 3'd0;
        queue_result_d[queue_head_d]   = 12'sd0;
        queue_sf_d[queue_head_d]       = 5'd0;
        queue_head_d                   = inc_ptr(queue_head_d);
        queue_count_d                  = queue_count_d - 1'b1;
      end

      if (enqueue_i) begin
        for (lane_idx_comb = 0; lane_idx_comb < 8; lane_idx_comb++) begin
          if (flush_mask_i[lane_idx_comb]) begin
            queue_lane_idx_d[queue_tail_d] = lane_idx_comb[2:0];
            queue_result_d[queue_tail_d]   = $signed(flush_result_vec_i[lane_idx_comb * 12 +: 12]);
            queue_sf_d[queue_tail_d]       = flush_sf_vec_i[lane_idx_comb * 5 +: 5];
            queue_tail_d                   = inc_ptr(queue_tail_d);
            queue_count_d                  = queue_count_d + 1'b1;
          end
        end
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      queue_head_q  <= '0;
      queue_tail_q  <= '0;
      queue_count_q <= '0;
    end else begin
      queue_head_q  <= queue_head_d;
      queue_tail_q  <= queue_tail_d;
      queue_count_q <= queue_count_d;

      for (queue_idx_ff = 0; queue_idx_ff < REQUEST_QUEUE_DEPTH; queue_idx_ff++) begin
        queue_lane_idx_q[queue_idx_ff] <= queue_lane_idx_d[queue_idx_ff];
        queue_result_q[queue_idx_ff]   <= queue_result_d[queue_idx_ff];
        queue_sf_q[queue_idx_ff]       <= queue_sf_d[queue_idx_ff];
      end
    end
  end

  assign flush_valid_o    = (queue_count_q != '0);
  assign flush_lane_idx_o = flush_valid_o ? queue_lane_idx_q[queue_head_q] : 3'd0;
  assign flush_result_o   = flush_valid_o ? queue_result_q[queue_head_q] : 12'sd0;
  assign flush_sf_o       = flush_valid_o ? queue_sf_q[queue_head_q] : 5'd0;

endmodule
