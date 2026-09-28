// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module fp_accumulate_bank_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic               clk_i;
  logic               clear_i;
  logic               flush_valid_i;
  logic         [2:0] flush_lane_idx_i;
  logic signed [11:0] flush_result_i;
  logic         [4:0] flush_sf_i;
  logic [7:0][15:0]   fp_result_o;

  logic signed [11:0] ref_int12_i;
  logic signed  [4:0] ref_sf_i;
  logic        [15:0] ref_bf16_o;
  logic        [15:0] ref_acc_prev_i;
  logic        [15:0] ref_acc_sum_o;

  logic [15:0] expected_fp_lane [0:7];

  fp_accumulate_bank dut (
    .clk_i           (clk_i),
    .clear_i         (clear_i),
    .flush_valid_i   (flush_valid_i),
    .flush_lane_idx_i(flush_lane_idx_i),
    .flush_result_i  (flush_result_i),
    .flush_sf_i      (flush_sf_i),
    .fp_result_o     (fp_result_o)
  );

  int12_scale_to_bf16 ref_converter (
    .int12_i         (ref_int12_i),
    .scaling_factor_i(ref_sf_i),
    .bf16_o          (ref_bf16_o)
  );

  bf16_add ref_adder (
    .a_i  (ref_acc_prev_i),
    .b_i  (ref_bf16_o),
    .sum_o(ref_acc_sum_o)
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

  task automatic reset_expected_model;
    integer lane_idx;
    begin
      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        expected_fp_lane[lane_idx] = 16'h0000;
      end
    end
  endtask

  task automatic drive_inputs(
    input logic               next_clear,
    input logic               next_flush_valid,
    input logic         [2:0] next_flush_lane_idx,
    input logic signed [11:0] next_flush_result,
    input logic         [4:0] next_flush_sf
  );
    begin
      clear_i          = next_clear;
      flush_valid_i    = next_flush_valid;
      flush_lane_idx_i = next_flush_lane_idx;
      flush_result_i   = next_flush_result;
      flush_sf_i       = next_flush_sf;
    end
  endtask

  task automatic check_fp_results(
    input string test_name
  );
    integer lane_idx;
    begin
      print_info($sformatf("Run parallel output check: %s", test_name));

      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        if (fp_result_o[lane_idx] !== expected_fp_lane[lane_idx]) begin
          print_fail($sformatf(
            "%s lane%0d mismatch: expected 0x%04h got 0x%04h",
            test_name,
            lane_idx,
            expected_fp_lane[lane_idx],
            fp_result_o[lane_idx]
          ));
          $fatal(
            1,
            "%s lane%0d mismatch: expected 0x%04h got 0x%04h",
            test_name,
            lane_idx,
            expected_fp_lane[lane_idx],
            fp_result_o[lane_idx]
          );
        end
      end

      print_pass($sformatf("%s matched all 8 FP lanes", test_name));
    end
  endtask

  task automatic apply_flush_request(
    input logic         [2:0] request_lane_idx,
    input logic signed [11:0] request_result,
    input logic         [4:0] request_sf,
    input string              test_name
  );
    logic [15:0] expected_after;
    begin
      print_info($sformatf(
        "Apply flush request: %s (lane=%0d, result=%0d, sf=%0d)",
        test_name,
        request_lane_idx,
        request_result,
        request_sf
      ));

      ref_int12_i    = request_result;
      ref_sf_i       = $signed(request_sf);
      ref_acc_prev_i = expected_fp_lane[request_lane_idx];
      #1;
      expected_after = ref_acc_sum_o;

      drive_inputs(1'b0, 1'b1, request_lane_idx, request_result, request_sf);
      @(posedge clk_i);
      #1;
      expected_fp_lane[request_lane_idx] = expected_after;
      check_fp_results(test_name);
    end
  endtask

  task automatic hold_and_check(
    input string test_name
  );
    begin
      drive_inputs(1'b0, 1'b0, 3'd0, 12'sd0, 5'd0);
      @(posedge clk_i);
      #1;
      check_fp_results(test_name);
    end
  endtask

  initial begin
    reset_expected_model();
    ref_int12_i    = 12'sd0;
    ref_sf_i       = 5'sd0;
    ref_acc_prev_i = 16'h0000;
    drive_inputs(1'b1, 1'b0, 3'd0, 12'sd0, 5'd0);

    print_info("Check synchronous clear initializes parallel outputs");
    @(posedge clk_i);
    #1;
    check_fp_results("clear_initializes_fp_results");

    drive_inputs(1'b0, 1'b0, 3'd0, 12'sd0, 5'd0);

    apply_flush_request(3'd1, 12'sd1, 5'd0, "single_request_lane1");
    hold_and_check("single_request_hold_outputs");

    apply_flush_request(3'd1, 12'sd1, 5'd0, "same_lane_accumulate_second_request");
    hold_and_check("same_lane_accumulate_hold_outputs");

    apply_flush_request(3'd0, -12'sd3, 5'd2, "independent_lane0_request");
    apply_flush_request(3'd6, 12'sd3, 5'd0, "independent_lane6_request");
    hold_and_check("multi_lane_independence_hold_outputs");

    drive_inputs(1'b1, 1'b0, 3'd0, 12'sd0, 5'd0);
    @(posedge clk_i);
    #1;
    reset_expected_model();
    check_fp_results("clear_after_activity_resets_fp_results");

    drive_inputs(1'b0, 1'b0, 3'd0, 12'sd0, 5'd0);
    apply_flush_request(3'd0, 12'sd1, 5'd0, "post_clear_lane0_request");
    apply_flush_request(3'd1, 12'sd3, 5'd0, "post_clear_lane1_request");
    hold_and_check("post_clear_resets_state_and_outputs");

    print_pass("fp_accumulate_bank_tb PASS");
    $finish;
  end

endmodule
