// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module sf_reorder_top_tb;

  localparam int NUM_SAMPLES = 8;
  localparam int NUM_LANES   = 8;
  localparam int IDX_W       = 3;
  localparam int TABLE_W     = 84;
  localparam int SEQ_W       = 24;
  localparam logic [1:0] STATE_IDLE        = 2'd0;
  localparam logic [1:0] STATE_RUN_MIN     = 2'd1;
  localparam logic [1:0] STATE_RUN_TABLE   = 2'd2;
  localparam logic [1:0] STATE_RUN_REORDER = 2'd3;
  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic               clk_i;
  logic               rst_ni;
  logic               start_i;
  logic        [39:0] weight_sf_vec_i;
  logic signed  [4:0] activation_sf_i;
  logic         [2:0] threshold_i;
  logic         [2:0] max_iter_i;
  logic               done_o;
  logic        [23:0] seq_o;

  logic        [39:0] sample_weight_vec [0:NUM_SAMPLES-1];
  logic signed  [4:0] sample_activation [0:NUM_SAMPLES-1];
  logic        [39:0] second_pass_weight_vec [0:NUM_SAMPLES-1];
  logic signed  [4:0] second_pass_activation [0:NUM_SAMPLES-1];

  sf_reorder_top dut (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .start_i        (start_i),
    .weight_sf_vec_i(weight_sf_vec_i),
    .activation_sf_i(activation_sf_i),
    .threshold_i    (threshold_i),
    .max_iter_i     (max_iter_i),
    .done_o         (done_o),
    .seq_o          (seq_o)
  );

  initial begin
    clk_i = 1'b0;
    forever #1000 clk_i = ~clk_i;
  end

`ifdef FSDB
  initial begin
    $fsdbDumpfile("waveform.fsdb");
    $fsdbDumpvars("+all");
    $fsdbDumpMDA();
  end
`endif

  task automatic print_info(input string message);
    begin
      $display("%s[INFO]%s %s", C_INFO, C_RESET, message);
    end
  endtask

  task automatic print_pass(input string message);
    begin
      $display("%s[PASS]%s %s", C_PASS, C_RESET, message);
    end
  endtask

  task automatic print_fail(input string message);
    begin
      $display("%s[FAIL]%s %s", C_FAIL, C_RESET, message);
    end
  endtask

  function automatic logic [39:0] pack_sf5x8(
    input logic signed [4:0] lane0,
    input logic signed [4:0] lane1,
    input logic signed [4:0] lane2,
    input logic signed [4:0] lane3,
    input logic signed [4:0] lane4,
    input logic signed [4:0] lane5,
    input logic signed [4:0] lane6,
    input logic signed [4:0] lane7
  );
    pack_sf5x8 = {
      lane7[4:0], lane6[4:0], lane5[4:0], lane4[4:0],
      lane3[4:0], lane2[4:0], lane1[4:0], lane0[4:0]
    };
  endfunction

  function automatic logic [15:0] pack_codes8(
    input logic [1:0] lane0,
    input logic [1:0] lane1,
    input logic [1:0] lane2,
    input logic [1:0] lane3,
    input logic [1:0] lane4,
    input logic [1:0] lane5,
    input logic [1:0] lane6,
    input logic [1:0] lane7
  );
    pack_codes8 = {
      lane7, lane6, lane5, lane4,
      lane3, lane2, lane1, lane0
    };
  endfunction

  function automatic logic [39:0] pack_codes_to_weights(
    input logic [15:0]       packed_codes,
    input logic signed [4:0] bias_sf,
    input logic signed [4:0] activation_sf
  );
    logic signed [5:0] norm_sf_ext;
    logic signed [5:0] weight_sf_ext;
    logic        [39:0] packed_sf;
    integer             lane_idx;
    begin
      packed_sf = '0;
      for (lane_idx = 0; lane_idx < NUM_LANES; lane_idx++) begin
        norm_sf_ext   = $signed({1'b0, 2'b00, packed_codes[lane_idx * 2 +: 2], 1'b0});
        weight_sf_ext = norm_sf_ext - $signed({activation_sf[4], activation_sf})
                      + $signed({bias_sf[4], bias_sf});
        packed_sf[lane_idx * 5 +: 5] = weight_sf_ext[4:0];
      end
      pack_codes_to_weights = packed_sf;
    end
  endfunction

  function automatic logic signed [4:0] min_weight_from_vec(input logic [39:0] packed_sf);
    logic signed [4:0] current_min;
    logic signed [4:0] lane_value;
    integer            lane_idx;
    begin
      current_min = $signed(packed_sf[4:0]);
      for (lane_idx = 1; lane_idx < NUM_LANES; lane_idx++) begin
        lane_value = $signed(packed_sf[lane_idx * 5 +: 5]);
        if (lane_value < current_min) begin
          current_min = lane_value;
        end
      end
      min_weight_from_vec = current_min;
    end
  endfunction

  function automatic logic signed [4:0] candidate_from_inputs(
    input logic [39:0]       packed_sf,
    input logic signed [4:0] activation_sf
  );
    logic signed [4:0] weight_min;
    logic signed [5:0] candidate_ext;
    begin
      weight_min    = min_weight_from_vec(packed_sf);
      candidate_ext = $signed({weight_min[4], weight_min}) + $signed({activation_sf[4], activation_sf});
      candidate_from_inputs = candidate_ext[4:0];
    end
  endfunction

  function automatic logic signed [4:0] expected_window_min;
    logic signed [4:0] current_min;
    logic signed [4:0] candidate_sf;
    integer            sample_idx;
    begin
      current_min = candidate_from_inputs(sample_weight_vec[0], sample_activation[0]);
      for (sample_idx = 1; sample_idx < NUM_SAMPLES; sample_idx++) begin
        candidate_sf = candidate_from_inputs(sample_weight_vec[sample_idx], sample_activation[sample_idx]);
        if (candidate_sf < current_min) begin
          current_min = candidate_sf;
        end
      end
      expected_window_min = current_min;
    end
  endfunction

  function automatic logic [39:0] select_weight_vec(
    input logic   use_second_pass,
    input integer sample_idx
  );
    begin
      if (use_second_pass) begin
        select_weight_vec = second_pass_weight_vec[sample_idx];
      end else begin
        select_weight_vec = sample_weight_vec[sample_idx];
      end
    end
  endfunction

  function automatic logic signed [4:0] select_activation_sf(
    input logic   use_second_pass,
    input integer sample_idx
  );
    begin
      if (use_second_pass) begin
        select_activation_sf = second_pass_activation[sample_idx];
      end else begin
        select_activation_sf = sample_activation[sample_idx];
      end
    end
  endfunction

  function automatic logic signed [6:0] lane_norm_value(
    input logic [39:0]       packed_sf,
    input integer            lane_idx,
    input logic signed [4:0] activation_sf,
    input logic signed [4:0] bias_sf
  );
    logic signed [4:0] weight_sf;
    logic signed [6:0] norm_sf;
    begin
      weight_sf = $signed(packed_sf[lane_idx * 5 +: 5]);
      norm_sf   = $signed({{2{weight_sf[4]}}, weight_sf})
                + $signed({{2{activation_sf[4]}}, activation_sf})
                - $signed({{2{bias_sf[4]}}, bias_sf});
      lane_norm_value = norm_sf;
    end
  endfunction

  function automatic logic [1:0] lane_code_from_inputs(
    input logic [39:0]       packed_sf,
    input integer            lane_idx,
    input logic signed [4:0] activation_sf,
    input logic signed [4:0] bias_sf
  );
    logic signed [6:0] norm_sf;
    begin
      norm_sf = lane_norm_value(packed_sf, lane_idx, activation_sf, bias_sf);
      lane_code_from_inputs = norm_sf[2:1];
    end
  endfunction

  function automatic logic [15:0] build_vec_from_inputs(
    input logic [39:0]       packed_sf,
    input logic signed [4:0] activation_sf,
    input logic signed [4:0] bias_sf
  );
    logic [15:0] packed_vec;
    integer      lane_idx;
    begin
      packed_vec = '0;
      for (lane_idx = 0; lane_idx < NUM_LANES; lane_idx++) begin
        packed_vec[lane_idx * 2 +: 2] = lane_code_from_inputs(
          packed_sf,
          lane_idx,
          activation_sf,
          bias_sf
        );
      end
      build_vec_from_inputs = packed_vec;
    end
  endfunction

  function automatic logic [2:0] hamming_distance_sat(
    input logic [15:0] vec_a,
    input logic [15:0] vec_b
  );
    logic   [3:0] raw_dist;
    integer       lane_idx;
    begin
      raw_dist = 4'd0;
      for (lane_idx = 0; lane_idx < NUM_LANES; lane_idx++) begin
        if (vec_a[lane_idx * 2 +: 2] != vec_b[lane_idx * 2 +: 2]) begin
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
      for (row_idx = 0; row_idx < older_idx; row_idx++) begin
        pair_idx = pair_idx + (NUM_SAMPLES - 1 - row_idx);
      end
      pair_idx = pair_idx + (newer_idx - older_idx - 1);
      pair_index = pair_idx;
    end
  endfunction

  function automatic logic [83:0] build_full_dist_table(
    input logic signed [4:0] bias_sf,
    input logic              use_second_pass
  );
    logic [83:0] expected_table;
    logic [15:0] older_vec;
    logic [15:0] newer_vec;
    integer      older_idx;
    integer      newer_idx;
    integer      slot_idx;
    begin
      expected_table = '0;
      for (newer_idx = 1; newer_idx < NUM_SAMPLES; newer_idx++) begin
        newer_vec = build_vec_from_inputs(
          select_weight_vec(use_second_pass, newer_idx),
          select_activation_sf(use_second_pass, newer_idx),
          bias_sf
        );
        for (older_idx = 0; older_idx < newer_idx; older_idx++) begin
          older_vec = build_vec_from_inputs(
            select_weight_vec(use_second_pass, older_idx),
            select_activation_sf(use_second_pass, older_idx),
            bias_sf
          );
          slot_idx = pair_index(older_idx, newer_idx);
          expected_table[slot_idx * IDX_W +: IDX_W] = hamming_distance_sat(older_vec, newer_vec);
        end
      end
      build_full_dist_table = expected_table;
    end
  endfunction

  function automatic logic [IDX_W-1:0] get_dist(
    input logic [TABLE_W-1:0] dist_tbl,
    input integer             idx_a,
    input integer             idx_b
  );
    integer lower_idx;
    integer upper_idx;
    integer slot_idx;
    begin
      if (idx_a == idx_b) begin
        get_dist = 3'd0;
      end else begin
        if (idx_a < idx_b) begin
          lower_idx = idx_a;
          upper_idx = idx_b;
        end else begin
          lower_idx = idx_b;
          upper_idx = idx_a;
        end
        slot_idx = pair_index(lower_idx, upper_idx);
        get_dist = dist_tbl[slot_idx * IDX_W +: IDX_W];
      end
    end
  endfunction

  function automatic logic [SEQ_W-1:0] pack_seq8(
    input logic [IDX_W-1:0] seq0,
    input logic [IDX_W-1:0] seq1,
    input logic [IDX_W-1:0] seq2,
    input logic [IDX_W-1:0] seq3,
    input logic [IDX_W-1:0] seq4,
    input logic [IDX_W-1:0] seq5,
    input logic [IDX_W-1:0] seq6,
    input logic [IDX_W-1:0] seq7
  );
    pack_seq8 = {seq7, seq6, seq5, seq4, seq3, seq2, seq1, seq0};
  endfunction

  function automatic logic [SEQ_W-1:0] pack_seq_array(
    input logic [IDX_W-1:0] seq_arr [0:NUM_SAMPLES-1]
  );
    logic [SEQ_W-1:0] packed_seq;
    integer           pos_idx;
    begin
      packed_seq = '0;
      for (pos_idx = 0; pos_idx < NUM_SAMPLES; pos_idx++) begin
        packed_seq[pos_idx * IDX_W +: IDX_W] = seq_arr[pos_idx];
      end
      pack_seq_array = packed_seq;
    end
  endfunction

  task automatic init_identity_seq(output logic [IDX_W-1:0] seq_arr [0:NUM_SAMPLES-1]);
    integer pos_idx;
    begin
      for (pos_idx = 0; pos_idx < NUM_SAMPLES; pos_idx++) begin
        seq_arr[pos_idx] = pos_idx;
      end
    end
  endtask

  function automatic logic [NUM_SAMPLES-1:0] build_anchor_mask(
    input logic [TABLE_W-1:0] dist_tbl,
    input logic        [2:0]  threshold,
    input logic [IDX_W-1:0]   seq_arr [0:NUM_SAMPLES-1]
  );
    logic [NUM_SAMPLES-1:0] anchor_mask;
    logic             [3:0] anchor_sum;
    integer                 pos_idx;
    begin
      anchor_mask = '0;
      anchor_mask[0] = (get_dist(dist_tbl, seq_arr[0], seq_arr[1]) > threshold);
      anchor_mask[NUM_SAMPLES-1] =
        (get_dist(dist_tbl, seq_arr[NUM_SAMPLES-2], seq_arr[NUM_SAMPLES-1]) > threshold);
      for (pos_idx = 1; pos_idx < NUM_SAMPLES - 1; pos_idx++) begin
        anchor_sum = {1'b0, get_dist(dist_tbl, seq_arr[pos_idx - 1], seq_arr[pos_idx])}
                   + {1'b0, get_dist(dist_tbl, seq_arr[pos_idx], seq_arr[pos_idx + 1])};
        anchor_mask[pos_idx] = (anchor_sum > threshold);
      end
      build_anchor_mask = anchor_mask;
    end
  endfunction

  function automatic logic [NUM_SAMPLES-1:0] build_segment_mask(
    input logic [NUM_SAMPLES-1:0] anchor_mask
  );
    logic [NUM_SAMPLES-1:0] segment_mask;
    logic                   left_non_anchor;
    logic                   right_non_anchor;
    integer                 pos_idx;
    begin
      segment_mask = '0;
      for (pos_idx = 0; pos_idx < NUM_SAMPLES; pos_idx++) begin
        left_non_anchor  = (pos_idx > 0) ? ~anchor_mask[pos_idx - 1] : 1'b0;
        right_non_anchor = (pos_idx < (NUM_SAMPLES - 1)) ? ~anchor_mask[pos_idx + 1] : 1'b0;
        segment_mask[pos_idx] = ~anchor_mask[pos_idx] & (left_non_anchor | right_non_anchor);
      end
      build_segment_mask = segment_mask;
    end
  endfunction

  task automatic build_seg_start(
    input  logic [NUM_SAMPLES-1:0] anchor_mask,
    output logic [IDX_W-1:0]       seg_start [0:NUM_SAMPLES-1]
  );
    integer pos_idx;
    begin
      seg_start[0] = 3'd0;
      for (pos_idx = 1; pos_idx < NUM_SAMPLES; pos_idx++) begin
        if (anchor_mask[pos_idx - 1]) begin
          seg_start[pos_idx] = pos_idx;
        end else begin
          seg_start[pos_idx] = seg_start[pos_idx - 1];
        end
      end
    end
  endtask

  task automatic build_seg_end(
    input  logic [NUM_SAMPLES-1:0] anchor_mask,
    output logic [IDX_W-1:0]       seg_end [0:NUM_SAMPLES-1]
  );
    integer pos_idx;
    begin
      seg_end[NUM_SAMPLES - 1] = NUM_SAMPLES - 1;
      for (pos_idx = NUM_SAMPLES - 2; pos_idx >= 0; pos_idx--) begin
        if (anchor_mask[pos_idx + 1]) begin
          seg_end[pos_idx] = pos_idx;
        end else begin
          seg_end[pos_idx] = seg_end[pos_idx + 1];
        end
      end
    end
  endtask

  function automatic logic segment_should_reverse(
    input logic [TABLE_W-1:0] dist_tbl,
    input logic [IDX_W-1:0]   seq_arr [0:NUM_SAMPLES-1],
    input logic [IDX_W-1:0]   start_idx,
    input logic [IDX_W-1:0]   end_idx
  );
    logic [3:0] current_cost;
    logic [3:0] reversed_cost;
    begin
      current_cost  = 4'd0;
      reversed_cost = 4'd0;

      if ((start_idx == 3'd0) && (end_idx < (NUM_SAMPLES - 1))) begin
        current_cost  = {1'b0, get_dist(dist_tbl, seq_arr[end_idx], seq_arr[end_idx + 1])};
        reversed_cost = {1'b0, get_dist(dist_tbl, seq_arr[0], seq_arr[end_idx + 1])};
      end else if ((start_idx > 3'd0) && (end_idx == (NUM_SAMPLES - 1))) begin
        current_cost  = {1'b0, get_dist(dist_tbl, seq_arr[start_idx - 1], seq_arr[start_idx])};
        reversed_cost = {1'b0, get_dist(dist_tbl, seq_arr[start_idx - 1], seq_arr[NUM_SAMPLES - 1])};
      end else if ((start_idx > 3'd0) && (end_idx < (NUM_SAMPLES - 1))) begin
        current_cost  = {1'b0, get_dist(dist_tbl, seq_arr[start_idx - 1], seq_arr[start_idx])}
                      + {1'b0, get_dist(dist_tbl, seq_arr[end_idx], seq_arr[end_idx + 1])};
        reversed_cost = {1'b0, get_dist(dist_tbl, seq_arr[start_idx - 1], seq_arr[end_idx])}
                      + {1'b0, get_dist(dist_tbl, seq_arr[start_idx], seq_arr[end_idx + 1])};
      end

      segment_should_reverse = (reversed_cost < current_cost);
    end
  endfunction

  task automatic run_reorder_reference(
    input  logic [TABLE_W-1:0] dist_tbl,
    input  logic        [2:0]  threshold,
    input  logic        [2:0]  max_iter,
    output logic [SEQ_W-1:0]   final_seq,
    output integer             expected_done_cycle
  );
    logic [IDX_W-1:0] seq_arr [0:NUM_SAMPLES-1];
    logic [IDX_W-1:0] next_seq [0:NUM_SAMPLES-1];
    logic [IDX_W-1:0] seg_start [0:NUM_SAMPLES-1];
    logic [IDX_W-1:0] seg_end [0:NUM_SAMPLES-1];
    logic [NUM_SAMPLES-1:0] anchor_mask;
    logic [NUM_SAMPLES-1:0] segment_mask;
    logic [NUM_SAMPLES-1:0] reverse_mask;
    logic                   any_reverse;
    integer                 iter_cnt;
    integer                 pos_idx;
    integer                 mirror_idx;
    begin
      init_identity_seq(seq_arr);
      expected_done_cycle = 0;
      iter_cnt            = 0;
      final_seq           = pack_seq_array(seq_arr);

      if (max_iter == 3'd0) begin
        expected_done_cycle = 1;
        final_seq           = pack_seq_array(seq_arr);
      end else begin
        forever begin
          expected_done_cycle = expected_done_cycle + 1;
          anchor_mask = build_anchor_mask(dist_tbl, threshold, seq_arr);
          if (anchor_mask == '0) begin
            final_seq = pack_seq_array(seq_arr);
            return;
          end

          expected_done_cycle = expected_done_cycle + 1;
          segment_mask = build_segment_mask(anchor_mask);
          build_seg_start(anchor_mask, seg_start);
          build_seg_end(anchor_mask, seg_end);

          expected_done_cycle = expected_done_cycle + 1;
          any_reverse = 1'b0;
          for (pos_idx = 0; pos_idx < NUM_SAMPLES; pos_idx++) begin
            reverse_mask[pos_idx] = 1'b0;
            next_seq[pos_idx]     = seq_arr[pos_idx];
            if (segment_mask[pos_idx]) begin
              reverse_mask[pos_idx] = segment_should_reverse(
                dist_tbl,
                seq_arr,
                seg_start[pos_idx],
                seg_end[pos_idx]
              );
              any_reverse = any_reverse | reverse_mask[pos_idx];
            end
          end

          if (!any_reverse) begin
            final_seq = pack_seq_array(seq_arr);
            return;
          end

          for (pos_idx = 0; pos_idx < NUM_SAMPLES; pos_idx++) begin
            if (segment_mask[pos_idx] && reverse_mask[pos_idx]) begin
              mirror_idx     = seg_start[pos_idx] + seg_end[pos_idx] - pos_idx;
              next_seq[pos_idx] = seq_arr[mirror_idx];
            end
          end

          for (pos_idx = 0; pos_idx < NUM_SAMPLES; pos_idx++) begin
            seq_arr[pos_idx] = next_seq[pos_idx];
          end

          iter_cnt = iter_cnt + 1;
          if (iter_cnt == max_iter) begin
            final_seq = pack_seq_array(seq_arr);
            return;
          end
        end
      end
    end
  endtask

  task automatic clear_window;
    integer sample_idx;
    begin
      for (sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) begin
        sample_weight_vec[sample_idx]       = pack_sf5x8(
          5'sd0, 5'sd0, 5'sd0, 5'sd0,
          5'sd0, 5'sd0, 5'sd0, 5'sd0
        );
        sample_activation[sample_idx]       = 5'sd0;
        second_pass_weight_vec[sample_idx]  = pack_sf5x8(
          5'sd0, 5'sd0, 5'sd0, 5'sd0,
          5'sd0, 5'sd0, 5'sd0, 5'sd0
        );
        second_pass_activation[sample_idx]  = 5'sd0;
      end
    end
  endtask

  task automatic copy_first_pass_to_second_pass;
    integer sample_idx;
    begin
      for (sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) begin
        second_pass_weight_vec[sample_idx] = sample_weight_vec[sample_idx];
        second_pass_activation[sample_idx] = sample_activation[sample_idx];
      end
    end
  endtask

  task automatic set_sample(
    input integer            sample_idx,
    input logic [39:0]       packed_sf,
    input logic signed [4:0] activation_sf
  );
    begin
      sample_weight_vec[sample_idx] = packed_sf;
      sample_activation[sample_idx] = activation_sf;
    end
  endtask

  task automatic set_second_pass_sample(
    input integer            sample_idx,
    input logic [39:0]       packed_sf,
    input logic signed [4:0] activation_sf
  );
    begin
      second_pass_weight_vec[sample_idx] = packed_sf;
      second_pass_activation[sample_idx] = activation_sf;
    end
  endtask

  task automatic set_second_pass_codes(
    input integer            sample_idx,
    input logic [15:0]       packed_codes,
    input logic signed [4:0] bias_sf,
    input logic signed [4:0] activation_sf
  );
    begin
      second_pass_weight_vec[sample_idx] = pack_codes_to_weights(packed_codes, bias_sf, activation_sf);
      second_pass_activation[sample_idx] = activation_sf;
    end
  endtask

  task automatic mutate_second_pass_sample(
    input integer            sample_idx,
    input logic [39:0]       packed_sf,
    input logic signed [4:0] activation_sf
  );
    begin
      second_pass_weight_vec[sample_idx] = packed_sf;
      second_pass_activation[sample_idx] = activation_sf;
    end
  endtask

  task automatic prepare_first_pass_bias_one;
    integer sample_idx;
    begin
      for (sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) begin
        sample_weight_vec[sample_idx] = pack_sf5x8(
          5'sd1, 5'sd2, 5'sd3, 5'sd4,
          5'sd5, 5'sd6, 5'sd7, 5'sd8
        );
        sample_activation[sample_idx] = 5'sd0;
      end
    end
  endtask

  task automatic prepare_nominal_second_pass(
    input logic signed [4:0] bias_sf
  );
    begin
      set_second_pass_codes(0, pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), bias_sf, 5'sd0);
      set_second_pass_codes(1, pack_codes8(2'd1, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), bias_sf, 5'sd0);
      set_second_pass_codes(2, pack_codes8(2'd1, 2'd1, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), bias_sf, 5'sd0);
      set_second_pass_codes(3, pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), bias_sf, 5'sd0);
      set_second_pass_codes(4, pack_codes8(2'd1, 2'd1, 2'd1, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), bias_sf, 5'sd0);
      set_second_pass_codes(5, pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd0, 2'd0, 2'd0, 2'd0), bias_sf, 5'sd0);
      set_second_pass_codes(6, pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd0), bias_sf, 5'sd0);
      set_second_pass_codes(7, pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1), bias_sf, 5'sd0);
    end
  endtask

  task automatic prepare_mutated_second_pass(
    input logic signed [4:0] bias_sf
  );
    begin
      prepare_nominal_second_pass(bias_sf);
      mutate_second_pass_sample(
        3,
        pack_codes_to_weights(
          pack_codes8(2'd2, 2'd2, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0),
          bias_sf,
          5'sd0
        ),
        5'sd0
      );
    end
  endtask

  task automatic check_outputs(
    input logic             expected_done,
    input logic [SEQ_W-1:0] expected_seq,
    input string            test_name
  );
    begin
      if (done_o !== expected_done) begin
        print_fail($sformatf(
          "%s done mismatch: expected %0b got %0b",
          test_name,
          expected_done,
          done_o
        ));
        $fatal(1, "%s done mismatch: expected %0b got %0b", test_name, expected_done, done_o);
      end

      if (seq_o !== expected_seq) begin
        print_fail($sformatf(
          "%s sequence mismatch: expected 0x%0h got 0x%0h",
          test_name,
          expected_seq,
          seq_o
        ));
        $fatal(1, "%s sequence mismatch: expected 0x%0h got 0x%0h", test_name, expected_seq, seq_o);
      end

      print_pass($sformatf("Check passed: %s", test_name));
    end
  endtask

  task automatic check_state(
    input logic [1:0] expected_state,
    input logic [2:0] expected_pass_cnt,
    input string      test_name
  );
    begin
      if (dut.state_q !== expected_state) begin
        print_fail($sformatf(
          "%s state mismatch: expected %0d got %0d",
          test_name,
          expected_state,
          dut.state_q
        ));
        $fatal(1, "%s state mismatch: expected %0d got %0d", test_name, expected_state, dut.state_q);
      end

      if (dut.pass_cnt_q !== expected_pass_cnt) begin
        print_fail($sformatf(
          "%s pass_cnt mismatch: expected %0d got %0d",
          test_name,
          expected_pass_cnt,
          dut.pass_cnt_q
        ));
        $fatal(1, "%s pass_cnt mismatch: expected %0d got %0d", test_name, expected_pass_cnt, dut.pass_cnt_q);
      end

      print_pass($sformatf("State matched: %s", test_name));
    end
  endtask

  task automatic check_min_output(
    input logic signed [4:0] expected_min,
    input string             test_name
  );
    begin
      if (dut.u_min_tracker.min_sf_o !== expected_min) begin
        print_fail($sformatf(
          "%s min mismatch: expected %0d got %0d",
          test_name,
          expected_min,
          dut.u_min_tracker.min_sf_o
        ));
        $fatal(1, "%s min mismatch: expected %0d got %0d", test_name, expected_min, dut.u_min_tracker.min_sf_o);
      end
      print_pass($sformatf("Min matched: %s", test_name));
    end
  endtask

  task automatic check_bias_input(
    input logic signed [4:0] expected_bias,
    input string             test_name
  );
    begin
      if (dut.u_hamming_table.bias_i !== expected_bias) begin
        print_fail($sformatf(
          "%s bias mismatch: expected %0d got %0d",
          test_name,
          expected_bias,
          dut.u_hamming_table.bias_i
        ));
        $fatal(1, "%s bias mismatch: expected %0d got %0d", test_name, expected_bias, dut.u_hamming_table.bias_i);
      end
      print_pass($sformatf("Bias matched: %s", test_name));
    end
  endtask

  task automatic check_table_output(
    input logic [TABLE_W-1:0] expected_table,
    input string              test_name
  );
    begin
      if (dut.u_hamming_table.dist_table_o !== expected_table) begin
        print_fail($sformatf(
          "%s dist_tbl mismatch: expected 0x%021h got 0x%021h",
          test_name,
          expected_table,
          dut.u_hamming_table.dist_table_o
        ));
        $fatal(
          1,
          "%s dist_tbl mismatch: expected 0x%021h got 0x%021h",
          test_name,
          expected_table,
          dut.u_hamming_table.dist_table_o
        );
      end
      print_pass($sformatf("Table matched: %s", test_name));
    end
  endtask

  task automatic apply_reset;
    begin
      print_info("Apply reset");
      rst_ni          = 1'b0;
      start_i         = 1'b0;
      weight_sf_vec_i = pack_sf5x8(
        5'sd0, 5'sd0, 5'sd0, 5'sd0,
        5'sd0, 5'sd0, 5'sd0, 5'sd0
      );
      activation_sf_i = 5'sd0;
      threshold_i     = 3'd0;
      max_iter_i      = 3'd0;

      repeat (2) @(posedge clk_i);
      #1;
      check_outputs(1'b0, 24'd0, "reset outputs");
      check_state(STATE_IDLE, 3'd0, "reset state");

      @(negedge clk_i);
      rst_ni = 1'b1;
      print_pass("Reset released");
    end
  endtask

  task automatic compute_expected_results(
    input  logic [2:0]        threshold,
    input  logic [2:0]        max_iter,
    output logic signed [4:0] expected_min,
    output logic [TABLE_W-1:0] expected_table,
    output logic [SEQ_W-1:0]  expected_seq,
    output integer            expected_reorder_done_cycle
  );
    begin
      expected_min   = expected_window_min();
      expected_table = build_full_dist_table(expected_min, 1'b1);
      run_reorder_reference(expected_table, threshold, max_iter, expected_seq, expected_reorder_done_cycle);
    end
  endtask

  task automatic start_two_pass_operation(
    input logic [2:0] threshold,
    input logic [2:0] max_iter,
    input string      test_name
  );
    begin
      print_info($sformatf("Start operation: %s", test_name));
      @(negedge clk_i);
      threshold_i     = threshold;
      max_iter_i      = max_iter;
      weight_sf_vec_i = sample_weight_vec[0];
      activation_sf_i = sample_activation[0];
      start_i         = 1'b1;
      #1;

      if (dut.state_q !== STATE_IDLE) begin
        print_fail($sformatf("%s expected idle before launch, got state %0d", test_name, dut.state_q));
        $fatal(1, "%s expected idle before launch", test_name);
      end

      if (dut.min_start !== 1'b1) begin
        print_fail($sformatf("%s did not assert min_start on launch", test_name));
        $fatal(1, "%s did not assert min_start on launch", test_name);
      end

      print_pass($sformatf("Launch accepted: %s", test_name));
    end
  endtask

  task automatic drive_first_pass(
    input integer            overlap_start_sample,
    input logic signed [4:0] expected_min,
    input string             test_name
  );
    integer sample_idx;
    begin
      for (sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) begin
        @(posedge clk_i);
        #1;

        if (sample_idx < (NUM_SAMPLES - 1)) begin
          check_state(
            STATE_RUN_MIN,
            sample_idx + 1,
            $sformatf("%s first-pass sample %0d state", test_name, sample_idx)
          );
        end else begin
          check_state(
            STATE_RUN_TABLE,
            3'd0,
            $sformatf("%s first-pass completion state", test_name)
          );
        end

        if (done_o !== 1'b0) begin
          print_fail($sformatf("%s asserted top done during first pass", test_name));
          $fatal(1, "%s asserted top done during first pass", test_name);
        end

        start_i = 1'b0;

        if (sample_idx < (NUM_SAMPLES - 1)) begin
          @(negedge clk_i);
          weight_sf_vec_i = sample_weight_vec[sample_idx + 1];
          activation_sf_i = sample_activation[sample_idx + 1];
          start_i         = ((sample_idx + 1) == overlap_start_sample);
        end
      end

      check_min_output(expected_min, $sformatf("%s first-pass minimum", test_name));
    end
  endtask

  task automatic drive_second_pass(
    input integer             overlap_start_sample,
    input logic signed  [4:0] expected_min,
    input logic [TABLE_W-1:0] expected_table,
    input string              test_name
  );
    integer sample_idx;
    begin
      @(negedge clk_i);
      weight_sf_vec_i = second_pass_weight_vec[0];
      activation_sf_i = second_pass_activation[0];
      start_i         = (overlap_start_sample == 0);
      #1;

      if (dut.table_start !== 1'b1) begin
        print_fail($sformatf("%s did not assert table_start on replay sample 0", test_name));
        $fatal(1, "%s did not assert table_start on replay sample 0", test_name);
      end

      check_bias_input(expected_min, $sformatf("%s replay bias", test_name));

      for (sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) begin
        @(posedge clk_i);
        #1;

        if (sample_idx < (NUM_SAMPLES - 1)) begin
          check_state(
            STATE_RUN_TABLE,
            sample_idx + 1,
            $sformatf("%s second-pass sample %0d state", test_name, sample_idx)
          );
          if (dut.u_hamming_table.done_o !== 1'b0) begin
            print_fail($sformatf("%s hamming done asserted early at replay sample %0d", test_name, sample_idx));
            $fatal(1, "%s hamming done asserted early at replay sample %0d", test_name, sample_idx);
          end
        end else begin
          check_state(
            STATE_RUN_TABLE,
            3'd7,
            $sformatf("%s second-pass completion state", test_name)
          );
          if (dut.u_hamming_table.done_o !== 1'b1) begin
            print_fail($sformatf("%s hamming done missing on replay sample 7", test_name));
            $fatal(1, "%s hamming done missing on replay sample 7", test_name);
          end
        end

        if (done_o !== 1'b0) begin
          print_fail($sformatf("%s asserted top done during second pass", test_name));
          $fatal(1, "%s asserted top done during second pass", test_name);
        end

        start_i = 1'b0;

        if (sample_idx < (NUM_SAMPLES - 1)) begin
          @(negedge clk_i);
          weight_sf_vec_i = second_pass_weight_vec[sample_idx + 1];
          activation_sf_i = second_pass_activation[sample_idx + 1];
          start_i         = ((sample_idx + 1) == overlap_start_sample);
        end
      end

      check_table_output(expected_table, $sformatf("%s second-pass dist_tbl", test_name));
    end
  endtask

  task automatic wait_for_done_and_check(
    input logic [SEQ_W-1:0] expected_seq,
    input integer           max_cycles,
    input integer           overlap_start_cycle,
    input string            test_name
  );
    integer wait_cycle;
    begin
      for (wait_cycle = 0; wait_cycle < max_cycles; wait_cycle++) begin
        @(negedge clk_i);
        weight_sf_vec_i = pack_sf5x8(
          5'sd0, 5'sd0, 5'sd0, 5'sd0,
          5'sd0, 5'sd0, 5'sd0, 5'sd0
        );
        activation_sf_i = 5'sd0;
        start_i         = (wait_cycle == overlap_start_cycle);

        @(posedge clk_i);
        #1;

        if (done_o === 1'b1) begin
          check_outputs(1'b1, expected_seq, $sformatf("%s done cycle", test_name));
          @(posedge clk_i);
          #1;
          check_outputs(1'b0, expected_seq, $sformatf("%s post-done hold", test_name));
          check_state(STATE_IDLE, 3'd0, $sformatf("%s idle return", test_name));
          return;
        end
      end

      print_fail($sformatf("%s timed out waiting for top done", test_name));
      $fatal(1, "%s timed out waiting for top done", test_name);
    end
  endtask

  task automatic run_end_to_end_case(
    input string       test_name,
    input logic [2:0]  threshold,
    input logic [2:0]  max_iter,
    input integer      first_pass_overlap,
    input integer      second_pass_overlap,
    input integer      reorder_overlap
  );
    logic signed [4:0] expected_min;
    logic [TABLE_W-1:0] expected_table;
    logic [SEQ_W-1:0] expected_seq;
    integer expected_reorder_done_cycle;
    begin
      compute_expected_results(threshold, max_iter, expected_min, expected_table, expected_seq, expected_reorder_done_cycle);

      start_two_pass_operation(threshold, max_iter, test_name);
      drive_first_pass(first_pass_overlap, expected_min, test_name);
      drive_second_pass(second_pass_overlap, expected_min, expected_table, test_name);
      wait_for_done_and_check(expected_seq, expected_reorder_done_cycle + 4, reorder_overlap, test_name);
    end
  endtask

  task automatic abort_with_reset(input string test_name);
    begin
      print_info($sformatf("Abort with reset: %s", test_name));
      rst_ni  = 1'b0;
      start_i = 1'b1;
      #1;
      check_outputs(1'b0, 24'd0, $sformatf("%s reset outputs", test_name));
      check_state(STATE_IDLE, 3'd0, $sformatf("%s reset state", test_name));
      repeat (2) @(posedge clk_i);
      #1;
      check_outputs(1'b0, 24'd0, $sformatf("%s reset hold", test_name));
      @(negedge clk_i);
      rst_ni          = 1'b1;
      start_i         = 1'b0;
      weight_sf_vec_i = pack_sf5x8(
        5'sd0, 5'sd0, 5'sd0, 5'sd0,
        5'sd0, 5'sd0, 5'sd0, 5'sd0
      );
      activation_sf_i = 5'sd0;
      threshold_i     = 3'd0;
      max_iter_i      = 3'd0;
      print_pass($sformatf("Reset abort finished: %s", test_name));
    end
  endtask

  initial begin
    logic signed [4:0] expected_min;
    logic [TABLE_W-1:0] expected_table;
    logic [TABLE_W-1:0] first_pass_table;
    logic [SEQ_W-1:0] expected_seq;
    logic [SEQ_W-1:0] identity_seq;
    integer expected_reorder_done_cycle;

    identity_seq = pack_seq8(3'd0, 3'd1, 3'd2, 3'd3, 3'd4, 3'd5, 3'd6, 3'd7);

    apply_reset();

    clear_window();
    prepare_first_pass_bias_one();
    prepare_nominal_second_pass(5'sd1);
    compute_expected_results(3'd3, 3'd2, expected_min, expected_table, expected_seq, expected_reorder_done_cycle);
    if (expected_min != 5'sd1) begin
      print_fail($sformatf("nominal setup expected min 1 got %0d", expected_min));
      $fatal(1, "nominal setup expected min 1 got %0d", expected_min);
    end
    if (expected_seq == identity_seq) begin
      print_fail("nominal setup did not create a non-identity reorder result");
      $fatal(1, "nominal setup did not create a non-identity reorder result");
    end
    run_end_to_end_case("nominal end-to-end", 3'd3, 3'd2, -1, -1, -1);

    clear_window();
    prepare_first_pass_bias_one();
    prepare_mutated_second_pass(5'sd1);
    compute_expected_results(3'd3, 3'd2, expected_min, expected_table, expected_seq, expected_reorder_done_cycle);
    first_pass_table = build_full_dist_table(expected_min, 1'b0);
    if (expected_table == first_pass_table) begin
      print_fail("replay sensitivity setup did not change the replayed dist_tbl");
      $fatal(1, "replay sensitivity setup did not change the replayed dist_tbl");
    end
    run_end_to_end_case("mutated replay drives dist_tbl", 3'd3, 3'd2, -1, -1, -1);

    clear_window();
    prepare_first_pass_bias_one();
    prepare_nominal_second_pass(5'sd1);
    compute_expected_results(3'd3, 3'd0, expected_min, expected_table, expected_seq, expected_reorder_done_cycle);
    if (expected_seq != identity_seq) begin
      print_fail("max_iter zero setup did not stay identity");
      $fatal(1, "max_iter zero setup did not stay identity");
    end
    start_two_pass_operation(3'd3, 3'd0, "max_iter zero still needs two passes");
    drive_first_pass(-1, expected_min, "max_iter zero still needs two passes");
    drive_second_pass(-1, expected_min, expected_table, "max_iter zero still needs two passes");
    if (done_o !== 1'b0) begin
      print_fail("max_iter zero asserted top done before reorder launch");
      $fatal(1, "max_iter zero asserted top done before reorder launch");
    end
    wait_for_done_and_check(expected_seq, expected_reorder_done_cycle + 4, -1, "max_iter zero still needs two passes");

    clear_window();
    prepare_first_pass_bias_one();
    prepare_nominal_second_pass(5'sd1);
    run_end_to_end_case("overlap start ignored in RUN_MIN", 3'd3, 3'd2, 3, -1, -1);

    clear_window();
    prepare_first_pass_bias_one();
    prepare_nominal_second_pass(5'sd1);
    run_end_to_end_case("overlap start ignored in RUN_TABLE", 3'd3, 3'd2, -1, 3, -1);

    clear_window();
    prepare_first_pass_bias_one();
    prepare_nominal_second_pass(5'sd1);
    run_end_to_end_case("overlap start ignored in RUN_REORDER", 3'd3, 3'd2, -1, -1, 1);

    clear_window();
    prepare_first_pass_bias_one();
    prepare_nominal_second_pass(5'sd1);
    start_two_pass_operation(3'd3, 3'd2, "reset abort in RUN_MIN");
    @(posedge clk_i);
    #1;
    check_state(STATE_RUN_MIN, 3'd1, "reset abort in RUN_MIN entered");
    @(negedge clk_i);
    weight_sf_vec_i = sample_weight_vec[1];
    activation_sf_i = sample_activation[1];
    start_i         = 1'b0;
    @(posedge clk_i);
    #1;
    check_state(STATE_RUN_MIN, 3'd2, "reset abort in RUN_MIN advanced");
    abort_with_reset("reset abort in RUN_MIN");

    clear_window();
    prepare_first_pass_bias_one();
    prepare_nominal_second_pass(5'sd1);
    compute_expected_results(3'd3, 3'd2, expected_min, expected_table, expected_seq, expected_reorder_done_cycle);
    start_two_pass_operation(3'd3, 3'd2, "reset abort in RUN_TABLE");
    drive_first_pass(-1, expected_min, "reset abort in RUN_TABLE");
    @(negedge clk_i);
    weight_sf_vec_i = second_pass_weight_vec[0];
    activation_sf_i = second_pass_activation[0];
    start_i         = 1'b0;
    @(posedge clk_i);
    #1;
    check_state(STATE_RUN_TABLE, 3'd1, "reset abort in RUN_TABLE replay started");
    abort_with_reset("reset abort in RUN_TABLE");

    clear_window();
    prepare_first_pass_bias_one();
    prepare_nominal_second_pass(5'sd1);
    compute_expected_results(3'd3, 3'd2, expected_min, expected_table, expected_seq, expected_reorder_done_cycle);
    start_two_pass_operation(3'd3, 3'd2, "reset abort in RUN_REORDER");
    drive_first_pass(-1, expected_min, "reset abort in RUN_REORDER");
    drive_second_pass(-1, expected_min, expected_table, "reset abort in RUN_REORDER");
    @(negedge clk_i);
    start_i = 1'b0;
    @(posedge clk_i);
    #1;
    check_state(STATE_RUN_REORDER, 3'd0, "reset abort in RUN_REORDER entered");
    abort_with_reset("reset abort in RUN_REORDER");

    clear_window();
    prepare_first_pass_bias_one();
    prepare_nominal_second_pass(5'sd1);
    run_end_to_end_case("back-to-back operation A", 3'd3, 3'd2, -1, -1, -1);

    clear_window();
    prepare_first_pass_bias_one();
    prepare_mutated_second_pass(5'sd1);
    run_end_to_end_case("back-to-back operation B", 3'd2, 3'd1, -1, -1, -1);

    print_pass("sf_reorder_top_tb PASS");
    $finish;
  end

endmodule
