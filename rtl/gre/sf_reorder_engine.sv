// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Iteratively reorder one fixed 8-element index sequence using anchor detection
// and conditional reversal of non-anchor segments derived from an external
// upper-triangular Hamming-distance table.
//
// Key I/O semantics:
// `start_i` is a one-cycle pulse accepted only while idle.
// Each accepted start initializes the working sequence to `0,1,2,3,4,5,6,7`.
// The accepted start cycle also performs the first `ANCHOR` phase immediately.
// While a run is active, overlapping `start_i` pulses are ignored.
//
// Timing / latency:
// Each iteration uses three registered phases: `ANCHOR`, `SEGMENT`, and `DECIDE`.
// The module terminates early if no anchors are found, if no segment wants to
// reverse, or if the completed-iteration count reaches `max_iter_i`.
// `done_o` pulses for one cycle when the final sequence is committed.
//
// Reset behavior:
// `rst_ni` is an asynchronous active-low reset that clears the active state,
// masks, segment boundaries, completion pulse, and output sequence immediately.

module sf_reorder_engine (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               start_i,
  input  logic        [83:0] dist_table_i,
  input  logic         [2:0] threshold_i,
  input  logic         [2:0] max_iter_i,
  output logic               done_o,
  output logic        [23:0] seq_o
);

  localparam int NUM_POS        = 8;
  localparam int IDX_W          = 3;
  localparam int TABLE_W        = 84;
  localparam logic [1:0] PHASE_ANCHOR  = 2'd0;
  localparam logic [1:0] PHASE_SEGMENT = 2'd1;
  localparam logic [1:0] PHASE_DECIDE  = 2'd2;

  logic               busy_d;
  logic               busy_q;
  logic         [1:0] phase_d;
  logic         [1:0] phase_q;
  logic         [2:0] iter_cnt_d;
  logic         [2:0] iter_cnt_q;
  logic               done_d;
  logic               done_q;
  logic         [7:0] anchor_mask_d;
  logic         [7:0] anchor_mask_q;
  logic         [7:0] segment_mask_d;
  logic         [7:0] segment_mask_q;
  logic         [2:0] seq_d [0:NUM_POS-1];
  logic         [2:0] seq_q [0:NUM_POS-1];
  logic         [2:0] seg_start_d [0:NUM_POS-1];
  logic         [2:0] seg_start_q [0:NUM_POS-1];
  logic         [2:0] seg_end_d [0:NUM_POS-1];
  logic         [2:0] seg_end_q [0:NUM_POS-1];
  logic         [2:0] seq_src [0:NUM_POS-1];
  logic         [7:0] anchor_mask_calc;
  logic         [7:0] segment_mask_calc;
  logic         [7:0] reverse_mask_calc;
  logic         [2:0] seg_start_calc [0:NUM_POS-1];
  logic         [2:0] seg_end_calc [0:NUM_POS-1];

  function automatic integer pair_index(
    input integer lower_idx,
    input integer upper_idx
  );
    integer base_idx;
    begin
      case (lower_idx)
        0: base_idx = 0;
        1: base_idx = 7;
        2: base_idx = 13;
        3: base_idx = 18;
        4: base_idx = 22;
        5: base_idx = 25;
        6: base_idx = 27;
        default: base_idx = 0;
      endcase

      pair_index = base_idx + (upper_idx - lower_idx - 1);
    end
  endfunction

  function automatic logic [IDX_W-1:0] dist_lookup(
    input logic [TABLE_W-1:0] dist_table,
    input logic [IDX_W-1:0]   idx_a,
    input logic [IDX_W-1:0]   idx_b
  );
    integer lower_idx;
    integer upper_idx;
    integer slot_idx;
    begin
      if (idx_a == idx_b) begin
        dist_lookup = 3'd0;
      end else begin
        if (idx_a < idx_b) begin
          lower_idx = idx_a;
          upper_idx = idx_b;
        end else begin
          lower_idx = idx_b;
          upper_idx = idx_a;
        end

        slot_idx    = pair_index(lower_idx, upper_idx);
        dist_lookup = dist_table[slot_idx * IDX_W +: IDX_W];
      end
    end
  endfunction

  always_comb begin
    integer pos_idx;
    integer start_idx;
    integer end_idx;
    integer mirror_idx;
    logic [3:0] anchor_sum;
    logic [3:0] current_cost;
    logic [3:0] reversed_cost;
    logic       any_reverse;

    busy_d         = busy_q;
    phase_d        = phase_q;
    iter_cnt_d     = iter_cnt_q;
    done_d         = 1'b0;
    anchor_mask_d  = anchor_mask_q;
    segment_mask_d = segment_mask_q;
    any_reverse    = 1'b0;

    for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
      seq_d[pos_idx]          = seq_q[pos_idx];
      seg_start_d[pos_idx]    = seg_start_q[pos_idx];
      seg_end_d[pos_idx]      = seg_end_q[pos_idx];
      seq_src[pos_idx]        = seq_q[pos_idx];
      anchor_mask_calc[pos_idx]  = 1'b0;
      segment_mask_calc[pos_idx] = 1'b0;
      reverse_mask_calc[pos_idx] = 1'b0;
      seg_start_calc[pos_idx]    = 3'd0;
      seg_end_calc[pos_idx]      = 3'd0;
    end

    if (!busy_q) begin
      if (start_i) begin
        phase_d        = PHASE_ANCHOR;
        iter_cnt_d     = 3'd0;
        anchor_mask_d  = 8'd0;
        segment_mask_d = 8'd0;

        for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
          seq_src[pos_idx]     = pos_idx;
          seq_d[pos_idx]       = pos_idx;
          seg_start_d[pos_idx] = pos_idx;
          seg_end_d[pos_idx]   = pos_idx;
        end

        if (max_iter_i == 3'd0) begin
          busy_d = 1'b0;
          done_d = 1'b1;
        end else begin
          anchor_mask_calc[0] = (dist_lookup(dist_table_i, seq_src[0], seq_src[1]) > threshold_i);
          anchor_mask_calc[NUM_POS-1] =
            (dist_lookup(dist_table_i, seq_src[NUM_POS-2], seq_src[NUM_POS-1]) > threshold_i);

          for (pos_idx = 1; pos_idx < NUM_POS - 1; pos_idx++) begin
            anchor_sum = {1'b0, dist_lookup(dist_table_i, seq_src[pos_idx - 1], seq_src[pos_idx])}
                       + {1'b0, dist_lookup(dist_table_i, seq_src[pos_idx], seq_src[pos_idx + 1])};
            anchor_mask_calc[pos_idx] = (anchor_sum > threshold_i);
          end

          anchor_mask_d = anchor_mask_calc;

          if (anchor_mask_calc == 8'd0) begin
            busy_d = 1'b0;
            done_d = 1'b1;
          end else begin
            busy_d  = 1'b1;
            phase_d = PHASE_SEGMENT;
          end
        end
      end
    end else begin
      case (phase_q)
        PHASE_ANCHOR: begin
          anchor_mask_calc[0] = (dist_lookup(dist_table_i, seq_q[0], seq_q[1]) > threshold_i);
          anchor_mask_calc[NUM_POS-1] =
            (dist_lookup(dist_table_i, seq_q[NUM_POS-2], seq_q[NUM_POS-1]) > threshold_i);

          for (pos_idx = 1; pos_idx < NUM_POS - 1; pos_idx++) begin
            anchor_sum = {1'b0, dist_lookup(dist_table_i, seq_q[pos_idx - 1], seq_q[pos_idx])}
                       + {1'b0, dist_lookup(dist_table_i, seq_q[pos_idx], seq_q[pos_idx + 1])};
            anchor_mask_calc[pos_idx] = (anchor_sum > threshold_i);
          end

          anchor_mask_d = anchor_mask_calc;

          if (anchor_mask_calc == 8'd0) begin
            busy_d  = 1'b0;
            phase_d = PHASE_ANCHOR;
            done_d  = 1'b1;
          end else begin
            busy_d  = 1'b1;
            phase_d = PHASE_SEGMENT;
          end
        end

        PHASE_SEGMENT: begin
          for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
            segment_mask_calc[pos_idx] = ~anchor_mask_q[pos_idx]
                                       & (((pos_idx > 0) ? ~anchor_mask_q[pos_idx - 1] : 1'b0)
                                       | ((pos_idx < (NUM_POS - 1)) ? ~anchor_mask_q[pos_idx + 1] : 1'b0));
          end

          seg_start_calc[0] = 3'd0;
          for (pos_idx = 1; pos_idx < NUM_POS; pos_idx++) begin
            if (anchor_mask_q[pos_idx - 1]) begin
              seg_start_calc[pos_idx] = pos_idx;
            end else begin
              seg_start_calc[pos_idx] = seg_start_calc[pos_idx - 1];
            end
          end

          seg_end_calc[NUM_POS - 1] = NUM_POS - 1;
          for (pos_idx = NUM_POS - 2; pos_idx >= 0; pos_idx--) begin
            if (anchor_mask_q[pos_idx + 1]) begin
              seg_end_calc[pos_idx] = pos_idx;
            end else begin
              seg_end_calc[pos_idx] = seg_end_calc[pos_idx + 1];
            end
          end

          segment_mask_d = segment_mask_calc;
          for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
            seg_start_d[pos_idx] = seg_start_calc[pos_idx];
            seg_end_d[pos_idx]   = seg_end_calc[pos_idx];
          end

          busy_d  = 1'b1;
          phase_d = PHASE_DECIDE;
        end

        PHASE_DECIDE: begin
          for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
            reverse_mask_calc[pos_idx] = 1'b0;
            current_cost               = 4'd0;
            reversed_cost              = 4'd0;

            if (segment_mask_q[pos_idx]) begin
              start_idx = seg_start_q[pos_idx];
              end_idx   = seg_end_q[pos_idx];

              if ((start_idx == 0) && (end_idx < (NUM_POS - 1))) begin
                current_cost  = {1'b0, dist_lookup(dist_table_i, seq_q[end_idx], seq_q[end_idx + 1])};
                reversed_cost = {1'b0, dist_lookup(dist_table_i, seq_q[0], seq_q[end_idx + 1])};
              end else if ((start_idx > 0) && (end_idx == (NUM_POS - 1))) begin
                current_cost  = {1'b0, dist_lookup(dist_table_i, seq_q[start_idx - 1], seq_q[start_idx])};
                reversed_cost = {1'b0, dist_lookup(dist_table_i, seq_q[start_idx - 1], seq_q[NUM_POS - 1])};
              end else if ((start_idx > 0) && (end_idx < (NUM_POS - 1))) begin
                current_cost  = {1'b0, dist_lookup(dist_table_i, seq_q[start_idx - 1], seq_q[start_idx])}
                              + {1'b0, dist_lookup(dist_table_i, seq_q[end_idx], seq_q[end_idx + 1])};
                reversed_cost = {1'b0, dist_lookup(dist_table_i, seq_q[start_idx - 1], seq_q[end_idx])}
                              + {1'b0, dist_lookup(dist_table_i, seq_q[start_idx], seq_q[end_idx + 1])};
              end

              reverse_mask_calc[pos_idx] = (reversed_cost < current_cost);
              any_reverse = any_reverse | reverse_mask_calc[pos_idx];
            end
          end

          if (!any_reverse) begin
            busy_d  = 1'b0;
            phase_d = PHASE_ANCHOR;
            done_d  = 1'b1;
          end else begin
            for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
              if (segment_mask_q[pos_idx] && reverse_mask_calc[pos_idx]) begin
                mirror_idx    = seg_start_q[pos_idx] + seg_end_q[pos_idx] - pos_idx;
                seq_d[pos_idx] = seq_q[mirror_idx];
              end
            end

            iter_cnt_d = iter_cnt_q + 3'd1;

            if ((iter_cnt_q + 3'd1) == max_iter_i) begin
              busy_d  = 1'b0;
              phase_d = PHASE_ANCHOR;
              done_d  = 1'b1;
            end else begin
              busy_d  = 1'b1;
              phase_d = PHASE_ANCHOR;
            end
          end
        end

        default: begin
          busy_d  = 1'b0;
          phase_d = PHASE_ANCHOR;
        end
      endcase
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    integer pos_idx;
    if (!rst_ni) begin
      busy_q         <= 1'b0;
      phase_q        <= PHASE_ANCHOR;
      iter_cnt_q     <= 3'd0;
      done_q         <= 1'b0;
      anchor_mask_q  <= 8'd0;
      segment_mask_q <= 8'd0;

      for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
        seq_q[pos_idx]       <= 3'd0;
        seg_start_q[pos_idx] <= 3'd0;
        seg_end_q[pos_idx]   <= 3'd0;
      end
    end else begin
      busy_q         <= busy_d;
      phase_q        <= phase_d;
      iter_cnt_q     <= iter_cnt_d;
      done_q         <= done_d;
      anchor_mask_q  <= anchor_mask_d;
      segment_mask_q <= segment_mask_d;

      for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
        seq_q[pos_idx]       <= seq_d[pos_idx];
        seg_start_q[pos_idx] <= seg_start_d[pos_idx];
        seg_end_q[pos_idx]   <= seg_end_d[pos_idx];
      end
    end
  end

  always_comb begin
    integer pos_idx;
    seq_o = '0;
    for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
      seq_o[pos_idx * IDX_W +: IDX_W] = seq_q[pos_idx];
    end
  end

  assign done_o = done_q;

endmodule
