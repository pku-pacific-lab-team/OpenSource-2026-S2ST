// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module pe_array_hybrid_accumulator_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  localparam int EXPECTED_QUEUE_DEPTH = 8;
  localparam logic [15:0] E2E_GOLDEN_LANE0 = 16'h40e0;
  localparam logic [15:0] E2E_GOLDEN_LANE1 = 16'h4100;
  localparam logic [15:0] E2E_GOLDEN_LANE2 = 16'h4110;
  localparam logic [15:0] E2E_GOLDEN_LANE3 = 16'h4120;
  localparam logic [15:0] E2E_GOLDEN_LANE4 = 16'h4130;
  localparam logic [15:0] E2E_GOLDEN_LANE5 = 16'h4140;
  localparam logic [15:0] E2E_GOLDEN_LANE6 = 16'h4150;
  localparam logic [15:0] E2E_GOLDEN_LANE7 = 16'h4160;

  logic        clk_i;
  logic        rst_ni;
  logic        clear_i;
  logic        weight_we_i;
  logic  [4:0] weight_addr_i;
  logic [31:0] weight_row_i;
  logic [15:0] row_weight_rsel_i;
  logic [31:0] activation_vec_i;
  logic [39:0] sf_vec_i;
  logic  [7:0] fp_flush_flag_i;
  logic        compute_accumulate_i;
  logic        finalize_i;
  logic [7:0][15:0] fp_result_o;

  logic [87:0] ref_result_vec_o;

  logic signed [11:0] ref_int12_i;
  logic signed  [4:0] ref_sf_i;
  logic        [15:0] ref_bf16_o;
  logic        [15:0] ref_acc_prev_i;
  logic        [15:0] ref_acc_sum_o;

  logic               expected_bank_valid;
  logic signed [11:0] expected_int_lane [0:7];
  logic         [4:0] expected_sf_lane  [0:7];
  logic        [15:0] expected_fp_lane  [0:7];

  logic         [2:0] expected_req_lane   [0:EXPECTED_QUEUE_DEPTH-1];
  logic signed [11:0] expected_req_result [0:EXPECTED_QUEUE_DEPTH-1];
  logic         [4:0] expected_req_sf     [0:EXPECTED_QUEUE_DEPTH-1];
  integer             expected_req_head;
  integer             expected_req_tail;
  integer             expected_req_count;

  pe_array_hybrid_accumulator dut (
    .clk_i             (clk_i),
    .rst_ni            (rst_ni),
    .clear_i           (clear_i),
    .weight_we_i       (weight_we_i),
    .weight_addr_i     (weight_addr_i),
    .weight_row_i      (weight_row_i),
    .row_weight_rsel_i (row_weight_rsel_i),
    .activation_vec_i  (activation_vec_i),
    .sf_vec_i          (sf_vec_i),
    .fp_flush_flag_i   (fp_flush_flag_i),
    .compute_accumulate_i(compute_accumulate_i),
    .finalize_i        (finalize_i),
    .fp_result_o       (fp_result_o)
  );

  pe_array ref_pe_array (
    .clk_i             (clk_i),
    .weight_we_i       (weight_we_i),
    .weight_addr_i     (weight_addr_i),
    .weight_row_i      (weight_row_i),
    .activation_vec_i  (activation_vec_i),
    .row_weight_rsel_i (row_weight_rsel_i),
    .result_vec_o      (ref_result_vec_o)
  );

  int12_scale_to_bf16 ref_converter (
    .int12_i          (ref_int12_i),
    .scaling_factor_i (ref_sf_i),
    .bf16_o           (ref_bf16_o)
  );

  bf16_add ref_adder (
    .a_i   (ref_acc_prev_i),
    .b_i   (ref_bf16_o),
    .sum_o (ref_acc_sum_o)
  );

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b1;
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

  function automatic logic [31:0] pack_int4x8(
    input logic signed [3:0] lane0,
    input logic signed [3:0] lane1,
    input logic signed [3:0] lane2,
    input logic signed [3:0] lane3,
    input logic signed [3:0] lane4,
    input logic signed [3:0] lane5,
    input logic signed [3:0] lane6,
    input logic signed [3:0] lane7
  );
    pack_int4x8 = {
      lane7[3:0], lane6[3:0], lane5[3:0], lane4[3:0],
      lane3[3:0], lane2[3:0], lane1[3:0], lane0[3:0]
    };
  endfunction

  function automatic logic [39:0] pack_sf5x8(
    input logic [4:0] lane0,
    input logic [4:0] lane1,
    input logic [4:0] lane2,
    input logic [4:0] lane3,
    input logic [4:0] lane4,
    input logic [4:0] lane5,
    input logic [4:0] lane6,
    input logic [4:0] lane7
  );
    pack_sf5x8 = {
      lane7, lane6, lane5, lane4,
      lane3, lane2, lane1, lane0
    };
  endfunction

  function automatic logic [15:0] pack_rsel8(
    input logic [1:0] row0,
    input logic [1:0] row1,
    input logic [1:0] row2,
    input logic [1:0] row3,
    input logic [1:0] row4,
    input logic [1:0] row5,
    input logic [1:0] row6,
    input logic [1:0] row7
  );
    pack_rsel8 = {
      row7, row6, row5, row4,
      row3, row2, row1, row0
    };
  endfunction

  function automatic logic signed [10:0] int11_lane(
    input logic [87:0] packed_results,
    input int          lane_idx
  );
    int11_lane = $signed(packed_results[lane_idx * 11 +: 11]);
  endfunction

  task automatic reset_expected_model;
    integer lane_idx;
    begin
      expected_bank_valid = 1'b0;
      expected_req_head   = 0;
      expected_req_tail   = 0;
      expected_req_count  = 0;

      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        expected_int_lane[lane_idx] = 12'sd0;
        expected_sf_lane[lane_idx]  = 5'd0;
        expected_fp_lane[lane_idx]  = 16'h0000;
      end

      for (lane_idx = 0; lane_idx < EXPECTED_QUEUE_DEPTH; lane_idx++) begin
        expected_req_lane[lane_idx]   = 3'd0;
        expected_req_result[lane_idx] = 12'sd0;
        expected_req_sf[lane_idx]     = 5'd0;
      end
    end
  endtask

  task automatic load_fixed_golden_e2e_expectations;
    begin
      // Hand-derived final totals for the dedicated end-to-end INT-accumulate test:
      // [7, 8, 9, 10, 11, 12, 13, 14] with sf = 0, encoded as BF16 constants.
      expected_fp_lane[0] = E2E_GOLDEN_LANE0;
      expected_fp_lane[1] = E2E_GOLDEN_LANE1;
      expected_fp_lane[2] = E2E_GOLDEN_LANE2;
      expected_fp_lane[3] = E2E_GOLDEN_LANE3;
      expected_fp_lane[4] = E2E_GOLDEN_LANE4;
      expected_fp_lane[5] = E2E_GOLDEN_LANE5;
      expected_fp_lane[6] = E2E_GOLDEN_LANE6;
      expected_fp_lane[7] = E2E_GOLDEN_LANE7;
    end
  endtask

  task automatic queue_push(
    input logic         [2:0] push_lane_idx,
    input logic signed [11:0] push_result,
    input logic         [4:0] push_sf
  );
    begin
      if (expected_req_count >= EXPECTED_QUEUE_DEPTH) begin
        print_fail("Reference request queue overflow");
        $fatal(1, "Reference request queue overflow");
      end

      expected_req_lane[expected_req_tail]   = push_lane_idx;
      expected_req_result[expected_req_tail] = push_result;
      expected_req_sf[expected_req_tail]     = push_sf;
      expected_req_tail                      = (expected_req_tail + 1) % EXPECTED_QUEUE_DEPTH;
      expected_req_count                     = expected_req_count + 1;
    end
  endtask

  task automatic check_serializer_head(input string test_name);
    logic               expected_valid;
    logic         [2:0] expected_lane_idx;
    logic signed [11:0] expected_result;
    logic         [4:0] expected_sf;
    begin
      expected_valid    = (expected_req_count != 0);
      expected_lane_idx = expected_valid ? expected_req_lane[expected_req_head] : 3'd0;
      expected_result   = expected_valid ? expected_req_result[expected_req_head] : 12'sd0;
      expected_sf       = expected_valid ? expected_req_sf[expected_req_head] : 5'd0;

      if (dut.serializer_flush_valid !== expected_valid) begin
        print_fail($sformatf(
          "%s serializer valid mismatch: expected %0b got %0b",
          test_name,
          expected_valid,
          dut.serializer_flush_valid
        ));
        $fatal(
          1,
          "%s serializer valid mismatch: expected %0b got %0b",
          test_name,
          expected_valid,
          dut.serializer_flush_valid
        );
      end

      if (dut.serializer_flush_lane_idx !== expected_lane_idx) begin
        print_fail($sformatf(
          "%s serializer lane mismatch: expected %0d got %0d",
          test_name,
          expected_lane_idx,
          dut.serializer_flush_lane_idx
        ));
        $fatal(
          1,
          "%s serializer lane mismatch: expected %0d got %0d",
          test_name,
          expected_lane_idx,
          dut.serializer_flush_lane_idx
        );
      end

      if (dut.serializer_flush_result !== expected_result) begin
        print_fail($sformatf(
          "%s serializer result mismatch: expected %0d got %0d",
          test_name,
          expected_result,
          dut.serializer_flush_result
        ));
        $fatal(
          1,
          "%s serializer result mismatch: expected %0d got %0d",
          test_name,
          expected_result,
          dut.serializer_flush_result
        );
      end

      if (dut.serializer_flush_sf !== expected_sf) begin
        print_fail($sformatf(
          "%s serializer sf mismatch: expected %0d got %0d",
          test_name,
          expected_sf,
          dut.serializer_flush_sf
        ));
        $fatal(
          1,
          "%s serializer sf mismatch: expected %0d got %0d",
          test_name,
          expected_sf,
          dut.serializer_flush_sf
        );
      end
    end
  endtask

  task automatic check_fp_output(input string test_name);
    integer fp_lane_idx;
    begin
      for (fp_lane_idx = 0; fp_lane_idx < 8; fp_lane_idx++) begin
        if (fp_result_o[fp_lane_idx] !== expected_fp_lane[fp_lane_idx]) begin
          print_fail($sformatf(
            "%s fp_result lane%0d mismatch: expected 0x%04h got 0x%04h",
            test_name,
            fp_lane_idx,
            expected_fp_lane[fp_lane_idx],
            fp_result_o[fp_lane_idx]
          ));
          $fatal(
            1,
            "%s fp_result lane%0d mismatch: expected 0x%04h got 0x%04h",
            test_name,
            fp_lane_idx,
            expected_fp_lane[fp_lane_idx],
            fp_result_o[fp_lane_idx]
          );
        end
      end
    end
  endtask

  task automatic advance_cycle(input string test_name);
    logic               consumed_valid;
    logic         [2:0] consumed_lane_idx;
    logic signed [11:0] consumed_result;
    logic         [4:0] consumed_sf;
    logic signed [10:0] pe_lane_value;
    logic signed [11:0] input_value;
    logic signed [11:0] aligned_input;
    logic signed [11:0] aligned_stored;
    logic         [4:0] input_sf;
    integer             lane_idx;
    begin
      print_info($sformatf("Advance cycle: %s", test_name));

      #1;

      if (clear_i) begin
        reset_expected_model();
      end else begin
        consumed_valid    = 1'b0;
        consumed_lane_idx = 3'd0;
        consumed_result   = 12'sd0;
        consumed_sf       = 5'd0;

        if (expected_req_count != 0) begin
          consumed_valid    = 1'b1;
          consumed_lane_idx = expected_req_lane[expected_req_head];
          consumed_result   = expected_req_result[expected_req_head];
          consumed_sf       = expected_req_sf[expected_req_head];
          expected_req_head = (expected_req_head + 1) % EXPECTED_QUEUE_DEPTH;
          expected_req_count = expected_req_count - 1;
        end

        if (compute_accumulate_i) begin
          if (!expected_bank_valid) begin
            expected_bank_valid = 1'b1;

            for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
              pe_lane_value            = int11_lane(ref_result_vec_o, lane_idx);
              input_value              = $signed({pe_lane_value[10], pe_lane_value});
              expected_int_lane[lane_idx] = input_value;
              expected_sf_lane[lane_idx]  = sf_vec_i[lane_idx * 5 +: 5];
            end
          end else begin
            for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
              if (fp_flush_flag_i[lane_idx]) begin
                queue_push(
                  lane_idx[2:0],
                  expected_int_lane[lane_idx],
                  expected_sf_lane[lane_idx]
                );
              end
            end

            for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
              pe_lane_value  = int11_lane(ref_result_vec_o, lane_idx);
              input_value    = $signed({pe_lane_value[10], pe_lane_value});
              input_sf       = sf_vec_i[lane_idx * 5 +: 5];
              aligned_input  = input_value;
              aligned_stored = expected_int_lane[lane_idx];

              if (fp_flush_flag_i[lane_idx]) begin
                expected_int_lane[lane_idx] = input_value;
                expected_sf_lane[lane_idx]  = input_sf;
              end else if (input_sf == expected_sf_lane[lane_idx]) begin
                expected_int_lane[lane_idx] = expected_int_lane[lane_idx] + input_value;
              end else if (input_sf == (expected_sf_lane[lane_idx] + 5'd1)) begin
                aligned_input               = input_value <<< 1;
                expected_int_lane[lane_idx] = expected_int_lane[lane_idx] + aligned_input;
              end else if (expected_sf_lane[lane_idx] == (input_sf + 5'd1)) begin
                aligned_stored              = expected_int_lane[lane_idx] <<< 1;
                expected_int_lane[lane_idx] = aligned_stored + input_value;
                expected_sf_lane[lane_idx]  = input_sf;
              end else begin
                expected_int_lane[lane_idx] = expected_int_lane[lane_idx] + input_value;
              end
            end
          end
        end

        if (consumed_valid) begin
          ref_int12_i    = consumed_result;
          ref_sf_i       = $signed(consumed_sf);
          ref_acc_prev_i = expected_fp_lane[consumed_lane_idx];
          #1;
          expected_fp_lane[consumed_lane_idx] = ref_acc_sum_o;
        end
      end

      @(posedge clk_i);
      #1;
      check_serializer_head(test_name);
      check_fp_output(test_name);
      print_pass($sformatf(
        "%s matched expected serializer/fp state (pending_reqs=%0d)",
        test_name,
        expected_req_count
      ));
    end
  endtask

  task automatic write_row(
    input logic  [2:0] row_idx,
    input logic  [1:0] slot_idx,
    input logic [31:0] packed_row
  );
    begin
      weight_we_i          = 1'b1;
      weight_addr_i        = {row_idx, slot_idx};
      weight_row_i         = packed_row;
      compute_accumulate_i = 1'b0;
      fp_flush_flag_i      = 8'h00;
      sf_vec_i             = '0;
      advance_cycle($sformatf("write_row%0d_slot%0d", row_idx, slot_idx));
      weight_we_i          = 1'b0;
      weight_addr_i        = '0;
      weight_row_i         = '0;
    end
  endtask

  integer row_idx;
  integer slot_idx;
  integer lane_idx;

  initial begin
    reset_expected_model();
    ref_int12_i          = 12'sd0;
    ref_sf_i             = 5'sd0;
    ref_acc_prev_i       = 16'h0000;
    clear_i              = 1'b1;
    weight_we_i          = 1'b0;
    weight_addr_i        = '0;
    weight_row_i         = '0;
    row_weight_rsel_i    = pack_rsel8(
      2'd0, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    activation_vec_i     = pack_int4x8(
      4'sd0, 4'sd0, 4'sd0, 4'sd0,
      4'sd0, 4'sd0, 4'sd0, 4'sd0
    );
    sf_vec_i             = '0;
    fp_flush_flag_i      = 8'h00;
    compute_accumulate_i = 1'b0;
    finalize_i           = 1'b0;

    advance_cycle("synchronous_clear_initializes_top_level_state");
    clear_i = 1'b0;

    for (row_idx = 0; row_idx < 8; row_idx++) begin
      for (slot_idx = 0; slot_idx < 4; slot_idx++) begin
        write_row(
          row_idx[2:0],
          slot_idx[1:0],
          pack_int4x8(
            4'sd0, 4'sd0, 4'sd0, 4'sd0,
            4'sd0, 4'sd0, 4'sd0, 4'sd0
          )
        );
      end
    end

    write_row(
      3'd0,
      2'd0,
      pack_int4x8(
        4'sd1, -4'sd2,  4'sd3, -4'sd4,
        4'sd5, -4'sd6,  4'sd7, -4'sd7
      )
    );
    write_row(
      3'd0,
      2'd1,
      pack_int4x8(
         4'sd2,  4'sd1, -4'sd1,  4'sd4,
        -4'sd3,  4'sd2, -4'sd2,  4'sd1
      )
    );
    write_row(
      3'd0,
      2'd2,
      pack_int4x8(
        -4'sd3,  4'sd4,  4'sd2, -4'sd1,
         4'sd1, -4'sd2,  4'sd3, -4'sd4
      )
    );

    activation_vec_i = pack_int4x8(
      4'sd1, 4'sd0, 4'sd0, 4'sd0,
      4'sd0, 4'sd0, 4'sd0, 4'sd0
    );

    row_weight_rsel_i = pack_rsel8(
      2'd0, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    sf_vec_i = pack_sf5x8(
      5'd0, 5'd1, 5'd2, 5'd3,
      5'd4, 5'd0, 5'd1, 5'd2
    );
    fp_flush_flag_i      = 8'hA5;
    compute_accumulate_i = 1'b1;
    advance_cycle("first_post_clear_compute_ignores_flush_flags");

    row_weight_rsel_i = pack_rsel8(
      2'd1, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    sf_vec_i = pack_sf5x8(
      5'd0, 5'd2, 5'd1, 5'd3,
      5'd5, 5'd1, 5'd0, 5'd2
    );
    fp_flush_flag_i = 8'h52;
    advance_cycle("mixed_int_accumulate_and_flush_cycle0");

    row_weight_rsel_i = pack_rsel8(
      2'd2, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    sf_vec_i = pack_sf5x8(
      5'd1, 5'd2, 5'd1, 5'd4,
      5'd4, 5'd0, 5'd1, 5'd3
    );
    fp_flush_flag_i = 8'h89;
    advance_cycle("mixed_int_accumulate_and_flush_cycle1");

    compute_accumulate_i = 1'b0;
    fp_flush_flag_i      = 8'h00;
    activation_vec_i     = pack_int4x8(
      4'sd0, 4'sd0, 4'sd0, 4'sd0,
      4'sd0, 4'sd0, 4'sd0, 4'sd0
    );
    row_weight_rsel_i    = pack_rsel8(
      2'd0, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    sf_vec_i             = '0;
    // Drain down to one pending request so the next all-lane flush fits in the 8-entry serializer queue.
    for (int drain_idx = 0; drain_idx < 4; drain_idx++) begin
      advance_cycle($sformatf("mixed_sequence_pre_final_flush_drain_%0d", drain_idx));
    end

    compute_accumulate_i = 1'b1;
    row_weight_rsel_i = pack_rsel8(
      2'd3, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    sf_vec_i = pack_sf5x8(
      5'd0, 5'd0, 5'd0, 5'd0,
      5'd0, 5'd0, 5'd0, 5'd0
    );
    fp_flush_flag_i = 8'hFF;
    advance_cycle("final_explicit_flush_all_lanes");

    compute_accumulate_i = 1'b0;
    fp_flush_flag_i      = 8'h00;
    activation_vec_i     = pack_int4x8(
      4'sd0, 4'sd0, 4'sd0, 4'sd0,
      4'sd0, 4'sd0, 4'sd0, 4'sd0
    );
    row_weight_rsel_i    = pack_rsel8(
      2'd0, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    sf_vec_i             = '0;

    while (expected_req_count != 0) begin
      advance_cycle($sformatf("drain_pending_flushes_%0d", expected_req_count));
    end

    advance_cycle("parallel_fp_output_hold_after_drain");

    clear_i = 1'b1;
    advance_cycle("fixed_golden_e2e_clear_state");
    clear_i = 1'b0;

    write_row(
      3'd0,
      2'd0,
      pack_int4x8(
        4'sd1, 4'sd2, 4'sd3, 4'sd4,
        4'sd5, 4'sd6, 4'sd7, 4'sd7
      )
    );
    write_row(
      3'd0,
      2'd1,
      pack_int4x8(
        4'sd1, 4'sd1, 4'sd1, 4'sd1,
        4'sd1, 4'sd1, 4'sd1, 4'sd1
      )
    );
    write_row(
      3'd0,
      2'd2,
      pack_int4x8(
        4'sd2, 4'sd2, 4'sd2, 4'sd2,
        4'sd2, 4'sd2, 4'sd2, 4'sd2
      )
    );
    write_row(
      3'd0,
      2'd3,
      pack_int4x8(
        4'sd3, 4'sd3, 4'sd3, 4'sd3,
        4'sd3, 4'sd3, 4'sd3, 4'sd3
      )
    );
    write_row(
      3'd1,
      2'd0,
      pack_int4x8(
        4'sd0, 4'sd0, 4'sd0, 4'sd0,
        4'sd0, 4'sd0, 4'sd0, 4'sd1
      )
    );

    activation_vec_i = pack_int4x8(
      4'sd1, 4'sd1, 4'sd0, 4'sd0,
      4'sd0, 4'sd0, 4'sd0, 4'sd0
    );
    sf_vec_i = pack_sf5x8(
      5'd0, 5'd0, 5'd0, 5'd0,
      5'd0, 5'd0, 5'd0, 5'd0
    );
    fp_flush_flag_i      = 8'h00;
    compute_accumulate_i = 1'b1;

    row_weight_rsel_i = pack_rsel8(
      2'd0, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    advance_cycle("fixed_golden_e2e_int_init_slot0");

    activation_vec_i = pack_int4x8(
      4'sd1, 4'sd0, 4'sd0, 4'sd0,
      4'sd0, 4'sd0, 4'sd0, 4'sd0
    );
    row_weight_rsel_i = pack_rsel8(
      2'd1, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    advance_cycle("fixed_golden_e2e_int_accumulate_slot1");

    row_weight_rsel_i = pack_rsel8(
      2'd2, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    advance_cycle("fixed_golden_e2e_int_accumulate_slot2");

    row_weight_rsel_i = pack_rsel8(
      2'd3, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    fp_flush_flag_i = 8'hFF;
    advance_cycle("fixed_golden_e2e_flush_accumulated_int");

    compute_accumulate_i = 1'b0;
    fp_flush_flag_i      = 8'h00;
    activation_vec_i = pack_int4x8(
      4'sd0, 4'sd0, 4'sd0, 4'sd0,
      4'sd0, 4'sd0, 4'sd0, 4'sd0
    );
    row_weight_rsel_i = pack_rsel8(
      2'd0, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    // Keep the residual INT bank contents intact while draining down to one pending serialized request.
    for (int drain_idx = 0; drain_idx < 7; drain_idx++) begin
      advance_cycle($sformatf("fixed_golden_e2e_pre_residual_flush_drain_%0d", drain_idx));
    end

    compute_accumulate_i = 1'b1;
    fp_flush_flag_i      = 8'hFF;
    advance_cycle("fixed_golden_e2e_flush_residual_int");

    compute_accumulate_i = 1'b0;
    fp_flush_flag_i      = 8'h00;
    while (expected_req_count != 0) begin
      advance_cycle($sformatf("fixed_golden_e2e_drain_pending_%0d", expected_req_count));
    end

    load_fixed_golden_e2e_expectations();
    advance_cycle("fixed_golden_e2e_parallel_output_hold");

    print_pass("pe_array_hybrid_accumulator_tb PASS");
    $finish;
  end

endmodule
