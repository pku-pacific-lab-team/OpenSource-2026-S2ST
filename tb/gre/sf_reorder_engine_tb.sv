// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module sf_reorder_engine_tb;

  localparam int NUM_POS  = 8;
  localparam int IDX_W    = 3;
  localparam int TABLE_W  = 84;
  localparam int SEQ_W    = 24;
  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic               clk_i;
  logic               rst_ni;
  logic               start_i;
  logic        [83:0] dist_table_i;
  logic         [2:0] threshold_i;
  logic         [2:0] max_iter_i;
  logic               done_o;
  logic        [23:0] seq_o;

  sf_reorder_engine dut (
    .clk_i       (clk_i),
    .rst_ni      (rst_ni),
    .start_i     (start_i),
    .dist_table_i(dist_table_i),
    .threshold_i (threshold_i),
    .max_iter_i  (max_iter_i),
    .done_o      (done_o),
    .seq_o       (seq_o)
  );

  initial begin
    clk_i = 1'b0;
    forever #5 clk_i = ~clk_i;
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

  function automatic integer pair_index(
    input integer lower_idx,
    input integer upper_idx
  );
    integer row_idx;
    integer slot_idx;
    begin
      slot_idx = 0;
      for (row_idx = 0; row_idx < lower_idx; row_idx++) begin
        slot_idx = slot_idx + (NUM_POS - 1 - row_idx);
      end
      slot_idx = slot_idx + (upper_idx - lower_idx - 1);
      pair_index = slot_idx;
    end
  endfunction

  task automatic set_dist(
    inout logic [TABLE_W-1:0] dist_tbl,
    input integer             idx_a,
    input integer             idx_b,
    input logic         [2:0] dist_value
  );
    integer lower_idx;
    integer upper_idx;
    integer slot_idx;
    begin
      if (idx_a == idx_b) begin
        print_fail($sformatf("set_dist requires distinct indices, got %0d and %0d", idx_a, idx_b));
        $fatal(1, "set_dist requires distinct indices, got %0d and %0d", idx_a, idx_b);
      end

      if (idx_a < idx_b) begin
        lower_idx = idx_a;
        upper_idx = idx_b;
      end else begin
        lower_idx = idx_b;
        upper_idx = idx_a;
      end

      slot_idx = pair_index(lower_idx, upper_idx);
      dist_tbl[slot_idx * IDX_W +: IDX_W] = dist_value;
    end
  endtask

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
    input logic [IDX_W-1:0] seq_arr [0:NUM_POS-1]
  );
    logic [SEQ_W-1:0] packed_seq;
    integer           pos_idx;
    begin
      packed_seq = '0;
      for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
        packed_seq[pos_idx * IDX_W +: IDX_W] = seq_arr[pos_idx];
      end
      pack_seq_array = packed_seq;
    end
  endfunction

  task automatic init_identity_seq(output logic [IDX_W-1:0] seq_arr [0:NUM_POS-1]);
    integer pos_idx;
    begin
      for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
        seq_arr[pos_idx] = pos_idx;
      end
    end
  endtask

  function automatic logic [NUM_POS-1:0] build_anchor_mask(
    input logic [TABLE_W-1:0] dist_tbl,
    input logic        [2:0]  threshold,
    input logic [IDX_W-1:0]   seq_arr [0:NUM_POS-1]
  );
    logic [NUM_POS-1:0] anchor_mask;
    logic         [3:0] anchor_sum;
    integer             pos_idx;
    begin
      anchor_mask          = '0;
      anchor_mask[0]       = (get_dist(dist_tbl, seq_arr[0], seq_arr[1]) > threshold);
      anchor_mask[NUM_POS-1] = (get_dist(dist_tbl, seq_arr[NUM_POS-2], seq_arr[NUM_POS-1]) > threshold);

      for (pos_idx = 1; pos_idx < NUM_POS - 1; pos_idx++) begin
        anchor_sum = {1'b0, get_dist(dist_tbl, seq_arr[pos_idx - 1], seq_arr[pos_idx])}
                   + {1'b0, get_dist(dist_tbl, seq_arr[pos_idx], seq_arr[pos_idx + 1])};
        anchor_mask[pos_idx] = (anchor_sum > threshold);
      end

      build_anchor_mask = anchor_mask;
    end
  endfunction

  function automatic logic [NUM_POS-1:0] build_segment_mask(
    input logic [NUM_POS-1:0] anchor_mask
  );
    logic [NUM_POS-1:0] segment_mask;
    logic               left_non_anchor;
    logic               right_non_anchor;
    integer             pos_idx;
    begin
      segment_mask = '0;
      for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
        left_non_anchor  = (pos_idx > 0) ? ~anchor_mask[pos_idx - 1] : 1'b0;
        right_non_anchor = (pos_idx < (NUM_POS - 1)) ? ~anchor_mask[pos_idx + 1] : 1'b0;
        segment_mask[pos_idx] = ~anchor_mask[pos_idx] & (left_non_anchor | right_non_anchor);
      end
      build_segment_mask = segment_mask;
    end
  endfunction

  task automatic build_seg_start(
    input  logic [NUM_POS-1:0] anchor_mask,
    output logic [IDX_W-1:0]   seg_start [0:NUM_POS-1]
  );
    integer pos_idx;
    begin
      seg_start[0] = 3'd0;
      for (pos_idx = 1; pos_idx < NUM_POS; pos_idx++) begin
        if (anchor_mask[pos_idx - 1]) begin
          seg_start[pos_idx] = pos_idx;
        end else begin
          seg_start[pos_idx] = seg_start[pos_idx - 1];
        end
      end
    end
  endtask

  task automatic build_seg_end(
    input  logic [NUM_POS-1:0] anchor_mask,
    output logic [IDX_W-1:0]   seg_end [0:NUM_POS-1]
  );
    integer pos_idx;
    begin
      seg_end[NUM_POS - 1] = NUM_POS - 1;
      for (pos_idx = NUM_POS - 2; pos_idx >= 0; pos_idx--) begin
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
    input logic [IDX_W-1:0]   seq_arr [0:NUM_POS-1],
    input logic [IDX_W-1:0]   start_idx,
    input logic [IDX_W-1:0]   end_idx
  );
    logic [3:0] current_cost;
    logic [3:0] reversed_cost;
    begin
      current_cost  = 4'd0;
      reversed_cost = 4'd0;

      if ((start_idx == 3'd0) && (end_idx < (NUM_POS - 1))) begin
        current_cost  = {1'b0, get_dist(dist_tbl, seq_arr[end_idx], seq_arr[end_idx + 1])};
        reversed_cost = {1'b0, get_dist(dist_tbl, seq_arr[0], seq_arr[end_idx + 1])};
      end else if ((start_idx > 3'd0) && (end_idx == (NUM_POS - 1))) begin
        current_cost  = {1'b0, get_dist(dist_tbl, seq_arr[start_idx - 1], seq_arr[start_idx])};
        reversed_cost = {1'b0, get_dist(dist_tbl, seq_arr[start_idx - 1], seq_arr[NUM_POS - 1])};
      end else if ((start_idx > 3'd0) && (end_idx < (NUM_POS - 1))) begin
        current_cost  = {1'b0, get_dist(dist_tbl, seq_arr[start_idx - 1], seq_arr[start_idx])}
                      + {1'b0, get_dist(dist_tbl, seq_arr[end_idx], seq_arr[end_idx + 1])};
        reversed_cost = {1'b0, get_dist(dist_tbl, seq_arr[start_idx - 1], seq_arr[end_idx])}
                      + {1'b0, get_dist(dist_tbl, seq_arr[start_idx], seq_arr[end_idx + 1])};
      end

      segment_should_reverse = (reversed_cost < current_cost);
    end
  endfunction

  task automatic run_reference(
    input  logic [TABLE_W-1:0] dist_tbl,
    input  logic        [2:0]  threshold,
    input  logic        [2:0]  max_iter,
    output logic [SEQ_W-1:0]   final_seq,
    output integer             expected_done_cycle
  );
    logic [IDX_W-1:0] seq_arr [0:NUM_POS-1];
    logic [IDX_W-1:0] next_seq [0:NUM_POS-1];
    logic [IDX_W-1:0] seg_start [0:NUM_POS-1];
    logic [IDX_W-1:0] seg_end [0:NUM_POS-1];
    logic [NUM_POS-1:0] anchor_mask;
    logic [NUM_POS-1:0] segment_mask;
    logic [NUM_POS-1:0] reverse_mask;
    logic               any_reverse;
    integer             iter_cnt;
    integer             pos_idx;
    integer             mirror_idx;
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
          for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
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

          for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
            if (segment_mask[pos_idx] && reverse_mask[pos_idx]) begin
              mirror_idx    = seg_start[pos_idx] + seg_end[pos_idx] - pos_idx;
              next_seq[pos_idx] = seq_arr[mirror_idx];
            end
          end

          for (pos_idx = 0; pos_idx < NUM_POS; pos_idx++) begin
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

  task automatic check_outputs(
    input logic              expected_done,
    input logic [SEQ_W-1:0]  expected_seq,
    input string             test_name
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

  task automatic apply_reset;
    begin
      print_info("Apply reset");
      rst_ni      = 1'b0;
      start_i     = 1'b0;
      dist_table_i = '0;
      threshold_i = 3'd0;
      max_iter_i  = 3'd0;

      repeat (2) @(posedge clk_i);
      #1;
      check_outputs(1'b0, 24'd0, "reset outputs");

      @(negedge clk_i);
      rst_ni = 1'b1;
      print_pass("Reset released");
    end
  endtask

  task automatic start_operation(
    input logic [TABLE_W-1:0] dist_table,
    input logic        [2:0]  threshold,
    input logic        [2:0]  max_iter
  );
    begin
      @(negedge clk_i);
      dist_table_i = dist_table;
      threshold_i  = threshold;
      max_iter_i   = max_iter;
      start_i      = 1'b1;
    end
  endtask

  task automatic wait_for_done_and_check(
    input string             test_name,
    input logic [SEQ_W-1:0]  expected_seq,
    input integer            expected_done_cycle,
    input integer            overlap_cycle
  );
    integer cycle_idx;
    begin
      for (cycle_idx = 1; cycle_idx <= (expected_done_cycle + 1); cycle_idx++) begin
        @(posedge clk_i);
        #1;

        if (cycle_idx < expected_done_cycle) begin
          if (done_o !== 1'b0) begin
            print_fail($sformatf("%s asserted done early at cycle %0d", test_name, cycle_idx));
            $fatal(1, "%s asserted done early at cycle %0d", test_name, cycle_idx);
          end
        end else if (cycle_idx == expected_done_cycle) begin
          if (done_o !== 1'b1) begin
            print_fail($sformatf("%s did not assert done at cycle %0d", test_name, expected_done_cycle));
            $fatal(1, "%s did not assert done at cycle %0d", test_name, expected_done_cycle);
          end

          if (seq_o !== expected_seq) begin
            print_fail($sformatf(
              "%s final sequence mismatch at done: expected 0x%0h got 0x%0h",
              test_name,
              expected_seq,
              seq_o
            ));
            $fatal(1, "%s final sequence mismatch at done", test_name);
          end
        end else begin
          if (done_o !== 1'b0) begin
            print_fail($sformatf("%s held done high beyond one cycle", test_name));
            $fatal(1, "%s held done high beyond one cycle", test_name);
          end

          if (seq_o !== expected_seq) begin
            print_fail($sformatf(
              "%s did not hold final sequence: expected 0x%0h got 0x%0h",
              test_name,
              expected_seq,
              seq_o
            ));
            $fatal(1, "%s did not hold final sequence", test_name);
          end
        end

        @(negedge clk_i);
        if (cycle_idx == 1) begin
          start_i = 1'b0;
        end else if (cycle_idx == overlap_cycle) begin
          start_i = 1'b1;
        end else begin
          start_i = 1'b0;
        end
      end

      print_pass($sformatf(
        "%s completed at cycle %0d with sequence 0x%0h",
        test_name,
        expected_done_cycle,
        expected_seq
      ));
    end
  endtask

  task automatic run_case(
    input string             test_name,
    input logic [TABLE_W-1:0] dist_table,
    input logic        [2:0]  threshold,
    input logic        [2:0]  max_iter,
    input logic [SEQ_W-1:0]   directed_expected_seq,
    input integer             directed_expected_done_cycle,
    input integer             overlap_cycle
  );
    logic [SEQ_W-1:0] reference_seq;
    integer           reference_done_cycle;
    begin
      run_reference(dist_table, threshold, max_iter, reference_seq, reference_done_cycle);

      if (reference_seq !== directed_expected_seq) begin
        print_fail($sformatf(
          "%s reference sequence mismatch: expected 0x%0h got 0x%0h",
          test_name,
          directed_expected_seq,
          reference_seq
        ));
        $fatal(1, "%s reference sequence mismatch", test_name);
      end

      if (reference_done_cycle != directed_expected_done_cycle) begin
        print_fail($sformatf(
          "%s reference done-cycle mismatch: expected %0d got %0d",
          test_name,
          directed_expected_done_cycle,
          reference_done_cycle
        ));
        $fatal(1, "%s reference done-cycle mismatch", test_name);
      end

      print_pass($sformatf("%s reference model matched the directed expectation", test_name));
      print_info($sformatf("Run case: %s", test_name));

      start_operation(dist_table, threshold, max_iter);
      wait_for_done_and_check(test_name, reference_seq, reference_done_cycle, overlap_cycle);
    end
  endtask

  initial begin
    logic [TABLE_W-1:0] dist_table_case;
    logic [SEQ_W-1:0]   reference_seq_limit2;
    integer             reference_done_cycle_limit2;

    rst_ni      = 1'b1;
    start_i     = 1'b0;
    dist_table_i = '0;
    threshold_i = 3'd0;
    max_iter_i  = 3'd0;

    apply_reset();

    dist_table_case = '0;
    run_case(
      "max_iter zero returns identity immediately",
      dist_table_case,
      3'd0,
      3'd0,
      pack_seq8(3'd0, 3'd1, 3'd2, 3'd3, 3'd4, 3'd5, 3'd6, 3'd7),
      1,
      -1
    );

    dist_table_case = '0;
    run_case(
      "initial no-anchor exits in anchor phase",
      dist_table_case,
      3'd0,
      3'd3,
      pack_seq8(3'd0, 3'd1, 3'd2, 3'd3, 3'd4, 3'd5, 3'd6, 3'd7),
      1,
      -1
    );

    dist_table_case = '0;
    set_dist(dist_table_case, 0, 1, 3'd3);
    set_dist(dist_table_case, 1, 2, 3'd1);
    set_dist(dist_table_case, 2, 3, 3'd0);
    set_dist(dist_table_case, 3, 4, 3'd1);
    set_dist(dist_table_case, 4, 5, 3'd2);
    set_dist(dist_table_case, 5, 6, 3'd3);
    set_dist(dist_table_case, 6, 7, 3'd3);
    set_dist(dist_table_case, 1, 3, 3'd1);
    set_dist(dist_table_case, 2, 4, 3'd2);
    run_case(
      "anchors without any beneficial reversal",
      dist_table_case,
      3'd2,
      3'd3,
      pack_seq8(3'd0, 3'd1, 3'd2, 3'd3, 3'd4, 3'd5, 3'd6, 3'd7),
      3,
      -1
    );

    dist_table_case = '0;
    set_dist(dist_table_case, 0, 1, 3'd3);
    set_dist(dist_table_case, 1, 2, 3'd1);
    set_dist(dist_table_case, 2, 3, 3'd0);
    set_dist(dist_table_case, 3, 4, 3'd0);
    set_dist(dist_table_case, 4, 5, 3'd0);
    set_dist(dist_table_case, 5, 6, 3'd1);
    set_dist(dist_table_case, 6, 7, 3'd3);
    set_dist(dist_table_case, 1, 5, 3'd0);
    set_dist(dist_table_case, 2, 6, 3'd0);
    run_case(
      "interior segment reversal",
      dist_table_case,
      3'd2,
      3'd1,
      pack_seq8(3'd0, 3'd1, 3'd5, 3'd4, 3'd3, 3'd2, 3'd6, 3'd7),
      3,
      -1
    );

    dist_table_case = '0;
    set_dist(dist_table_case, 0, 1, 3'd1);
    set_dist(dist_table_case, 1, 2, 3'd0);
    set_dist(dist_table_case, 2, 3, 3'd1);
    set_dist(dist_table_case, 3, 4, 3'd2);
    set_dist(dist_table_case, 4, 5, 3'd3);
    set_dist(dist_table_case, 5, 6, 3'd3);
    set_dist(dist_table_case, 6, 7, 3'd3);
    set_dist(dist_table_case, 0, 3, 3'd0);
    run_case(
      "prefix segment reversal uses only the right boundary",
      dist_table_case,
      3'd2,
      3'd1,
      pack_seq8(3'd2, 3'd1, 3'd0, 3'd3, 3'd4, 3'd5, 3'd6, 3'd7),
      3,
      -1
    );

    dist_table_case = '0;
    set_dist(dist_table_case, 0, 1, 3'd3);
    set_dist(dist_table_case, 1, 2, 3'd3);
    set_dist(dist_table_case, 2, 3, 3'd3);
    set_dist(dist_table_case, 3, 4, 3'd2);
    set_dist(dist_table_case, 4, 5, 3'd1);
    set_dist(dist_table_case, 5, 6, 3'd0);
    set_dist(dist_table_case, 6, 7, 3'd1);
    set_dist(dist_table_case, 4, 7, 3'd0);
    run_case(
      "suffix segment reversal uses only the left boundary",
      dist_table_case,
      3'd2,
      3'd1,
      pack_seq8(3'd0, 3'd1, 3'd2, 3'd3, 3'd4, 3'd7, 3'd6, 3'd5),
      3,
      -1
    );

    dist_table_case = '0;
    set_dist(dist_table_case, 0, 1, 3'd1);
    set_dist(dist_table_case, 1, 2, 3'd1);
    set_dist(dist_table_case, 2, 3, 3'd2);
    set_dist(dist_table_case, 3, 4, 3'd1);
    set_dist(dist_table_case, 4, 5, 3'd1);
    set_dist(dist_table_case, 5, 6, 3'd1);
    set_dist(dist_table_case, 6, 7, 3'd2);
    set_dist(dist_table_case, 0, 2, 3'd0);
    set_dist(dist_table_case, 3, 5, 3'd0);
    set_dist(dist_table_case, 4, 6, 3'd0);
    run_case(
      "parallel segment decisions in one iteration",
      dist_table_case,
      3'd2,
      3'd1,
      pack_seq8(3'd1, 3'd0, 3'd2, 3'd3, 3'd5, 3'd4, 3'd6, 3'd7),
      3,
      -1
    );

    dist_table_case = '0;
    set_dist(dist_table_case, 4, 7, 3'd1);
    set_dist(dist_table_case, 1, 3, 3'd1);
    set_dist(dist_table_case, 5, 6, 3'd1);
    set_dist(dist_table_case, 0, 7, 3'd1);
    set_dist(dist_table_case, 1, 6, 3'd1);
    set_dist(dist_table_case, 3, 7, 3'd2);
    set_dist(dist_table_case, 2, 5, 3'd3);
    set_dist(dist_table_case, 0, 3, 3'd0);
    set_dist(dist_table_case, 1, 2, 3'd1);
    set_dist(dist_table_case, 6, 7, 3'd3);
    set_dist(dist_table_case, 1, 5, 3'd2);
    set_dist(dist_table_case, 3, 6, 3'd3);
    set_dist(dist_table_case, 0, 4, 3'd1);
    set_dist(dist_table_case, 2, 7, 3'd1);
    set_dist(dist_table_case, 2, 6, 3'd2);
    set_dist(dist_table_case, 4, 5, 3'd3);
    set_dist(dist_table_case, 1, 4, 3'd2);
    set_dist(dist_table_case, 0, 5, 3'd2);
    set_dist(dist_table_case, 3, 5, 3'd1);
    set_dist(dist_table_case, 0, 1, 3'd2);
    set_dist(dist_table_case, 4, 6, 3'd2);
    set_dist(dist_table_case, 5, 7, 3'd0);
    set_dist(dist_table_case, 0, 2, 3'd0);
    set_dist(dist_table_case, 0, 6, 3'd1);
    set_dist(dist_table_case, 1, 7, 3'd1);
    set_dist(dist_table_case, 2, 3, 3'd3);
    set_dist(dist_table_case, 3, 4, 3'd2);
    set_dist(dist_table_case, 2, 4, 3'd2);

    run_reference(dist_table_case, 3'd3, 3'd2, reference_seq_limit2, reference_done_cycle_limit2);
    if (reference_seq_limit2 !== pack_seq8(3'd2, 3'd0, 3'd1, 3'd3, 3'd4, 3'd5, 3'd6, 3'd7)) begin
      print_fail($sformatf(
        "iteration-limit reference for max_iter=2 mismatch: expected 0x%0h got 0x%0h",
        pack_seq8(3'd2, 3'd0, 3'd1, 3'd3, 3'd4, 3'd5, 3'd6, 3'd7),
        reference_seq_limit2
      ));
      $fatal(1, "iteration-limit reference for max_iter=2 mismatch");
    end

    if (reference_done_cycle_limit2 != 6) begin
      print_fail($sformatf(
        "iteration-limit reference for max_iter=2 done-cycle mismatch: expected 6 got %0d",
        reference_done_cycle_limit2
      ));
      $fatal(1, "iteration-limit reference for max_iter=2 done-cycle mismatch");
    end

    print_pass("Iteration-limit setup proves a second update exists when max_iter increases");
    run_case(
      "iteration limit stops after the first completed update",
      dist_table_case,
      3'd3,
      3'd1,
      pack_seq8(3'd1, 3'd0, 3'd2, 3'd3, 3'd4, 3'd5, 3'd6, 3'd7),
      3,
      -1
    );

    dist_table_case = '0;
    set_dist(dist_table_case, 0, 1, 3'd3);
    set_dist(dist_table_case, 1, 2, 3'd1);
    set_dist(dist_table_case, 2, 3, 3'd0);
    set_dist(dist_table_case, 3, 4, 3'd0);
    set_dist(dist_table_case, 4, 5, 3'd0);
    set_dist(dist_table_case, 5, 6, 3'd1);
    set_dist(dist_table_case, 6, 7, 3'd3);
    set_dist(dist_table_case, 1, 5, 3'd0);
    set_dist(dist_table_case, 2, 6, 3'd0);
    run_case(
      "ignore overlapping start while busy",
      dist_table_case,
      3'd2,
      3'd1,
      pack_seq8(3'd0, 3'd1, 3'd5, 3'd4, 3'd3, 3'd2, 3'd6, 3'd7),
      3,
      2
    );

    print_pass("sf_reorder_engine_tb PASS");
    $finish;
  end

endmodule
