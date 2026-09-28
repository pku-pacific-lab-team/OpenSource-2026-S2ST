// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module int_acc_flush_bank_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic        clk_i;
  logic        rst_ni;
  logic        clear_i;
  logic        update_i;
  logic [87:0] result_vec_i;
  logic [39:0] sf_vec_i;
  logic  [7:0] fp_flush_flag_i;

  logic        bank_valid_o;
  logic [95:0] int_result_vec_o;
  logic [39:0] int_sf_vec_o;
  logic  [7:0] flush_mask_o;
  logic [95:0] flush_result_vec_o;
  logic [39:0] flush_sf_vec_o;

  int_acc_flush_bank dut (
    .clk_i           (clk_i),
    .rst_ni          (rst_ni),
    .clear_i         (clear_i),
    .update_i        (update_i),
    .result_vec_i    (result_vec_i),
    .sf_vec_i        (sf_vec_i),
    .fp_flush_flag_i (fp_flush_flag_i),
    .bank_valid_o    (bank_valid_o),
    .int_result_vec_o(int_result_vec_o),
    .int_sf_vec_o    (int_sf_vec_o),
    .flush_mask_o    (flush_mask_o),
    .flush_result_vec_o(flush_result_vec_o),
    .flush_sf_vec_o  (flush_sf_vec_o)
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

  function automatic logic [87:0] pack_int11x8(
    input logic signed [10:0] lane0,
    input logic signed [10:0] lane1,
    input logic signed [10:0] lane2,
    input logic signed [10:0] lane3,
    input logic signed [10:0] lane4,
    input logic signed [10:0] lane5,
    input logic signed [10:0] lane6,
    input logic signed [10:0] lane7
  );
    pack_int11x8 = {
      lane7[10:0], lane6[10:0], lane5[10:0], lane4[10:0],
      lane3[10:0], lane2[10:0], lane1[10:0], lane0[10:0]
    };
  endfunction

  function automatic logic [95:0] pack_int12x8(
    input logic signed [11:0] lane0,
    input logic signed [11:0] lane1,
    input logic signed [11:0] lane2,
    input logic signed [11:0] lane3,
    input logic signed [11:0] lane4,
    input logic signed [11:0] lane5,
    input logic signed [11:0] lane6,
    input logic signed [11:0] lane7
  );
    pack_int12x8 = {
      lane7[11:0], lane6[11:0], lane5[11:0], lane4[11:0],
      lane3[11:0], lane2[11:0], lane1[11:0], lane0[11:0]
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

  function automatic logic signed [11:0] int12_lane(
    input logic [95:0] packed_results,
    input int          lane_idx
  );
    int12_lane = $signed(packed_results[lane_idx * 12 +: 12]);
  endfunction

  function automatic logic [4:0] sf5_lane(
    input logic [39:0] packed_sf,
    input int          lane_idx
  );
    sf5_lane = packed_sf[lane_idx * 5 +: 5];
  endfunction

  task automatic check_bank_state(
    input logic        expected_valid,
    input logic [95:0] expected_result_vec,
    input logic [39:0] expected_sf_vec,
    input string       test_name
  );
    logic signed [11:0] observed_result;
    logic signed [11:0] expected_result;
    logic        [4:0]  observed_sf;
    logic        [4:0]  expected_sf;
    integer             lane_idx;
    begin
      print_info($sformatf("Run state check: %s", test_name));

      if (bank_valid_o !== expected_valid) begin
        print_fail($sformatf(
          "%s valid mismatch: expected %0b got %0b",
          test_name,
          expected_valid,
          bank_valid_o
        ));
        $fatal(1, "%s valid mismatch: expected %0b got %0b", test_name, expected_valid, bank_valid_o);
      end

      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        observed_result = int12_lane(int_result_vec_o, lane_idx);
        expected_result = int12_lane(expected_result_vec, lane_idx);
        observed_sf     = sf5_lane(int_sf_vec_o, lane_idx);
        expected_sf     = sf5_lane(expected_sf_vec, lane_idx);

        if (observed_result !== expected_result) begin
          print_fail($sformatf(
            "%s lane %0d result mismatch: expected %0d got %0d",
            test_name,
            lane_idx,
            expected_result,
            observed_result
          ));
          $fatal(
            1,
            "%s lane %0d result mismatch: expected %0d got %0d",
            test_name,
            lane_idx,
            expected_result,
            observed_result
          );
        end

        if (observed_sf !== expected_sf) begin
          print_fail($sformatf(
            "%s lane %0d sf mismatch: expected %0d got %0d",
            test_name,
            lane_idx,
            expected_sf,
            observed_sf
          ));
          $fatal(
            1,
            "%s lane %0d sf mismatch: expected %0d got %0d",
            test_name,
            lane_idx,
            expected_sf,
            observed_sf
          );
        end
      end

      print_pass($sformatf("State check passed: %s", test_name));
    end
  endtask

  task automatic check_flush_batch(
    input logic  [7:0] expected_mask,
    input logic [95:0] expected_result_vec,
    input logic [39:0] expected_sf_vec,
    input string       test_name
  );
    logic signed [11:0] observed_result;
    logic signed [11:0] expected_result;
    logic        [4:0]  observed_sf;
    logic        [4:0]  expected_sf;
    integer             lane_idx;
    begin
      print_info($sformatf("Run flush check: %s", test_name));

      if (flush_mask_o !== expected_mask) begin
        print_fail($sformatf(
          "%s mask mismatch: expected 0x%02h got 0x%02h",
          test_name,
          expected_mask,
          flush_mask_o
        ));
        $fatal(1, "%s mask mismatch: expected 0x%02h got 0x%02h", test_name, expected_mask, flush_mask_o);
      end

      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        observed_result = int12_lane(flush_result_vec_o, lane_idx);
        expected_result = int12_lane(expected_result_vec, lane_idx);
        observed_sf     = sf5_lane(flush_sf_vec_o, lane_idx);
        expected_sf     = sf5_lane(expected_sf_vec, lane_idx);

        if (observed_result !== expected_result) begin
          print_fail($sformatf(
            "%s lane %0d flush result mismatch: expected %0d got %0d",
            test_name,
            lane_idx,
            expected_result,
            observed_result
          ));
          $fatal(
            1,
            "%s lane %0d flush result mismatch: expected %0d got %0d",
            test_name,
            lane_idx,
            expected_result,
            observed_result
          );
        end

        if (observed_sf !== expected_sf) begin
          print_fail($sformatf(
            "%s lane %0d flush sf mismatch: expected %0d got %0d",
            test_name,
            lane_idx,
            expected_sf,
            observed_sf
          ));
          $fatal(
            1,
            "%s lane %0d flush sf mismatch: expected %0d got %0d",
            test_name,
            lane_idx,
            expected_sf,
            observed_sf
          );
        end
      end

      print_pass($sformatf("Flush check passed: %s", test_name));
    end
  endtask

  task automatic drive_inputs(
    input logic        next_clear,
    input logic        next_update,
    input logic  [7:0] next_fp_flush_flag,
    input logic [87:0] next_result_vec,
    input logic [39:0] next_sf_vec
  );
    begin
      clear_i         = next_clear;
      update_i        = next_update;
      fp_flush_flag_i = next_fp_flush_flag;
      result_vec_i    = next_result_vec;
      sf_vec_i        = next_sf_vec;
    end
  endtask

  initial begin
    drive_inputs(1'b1, 1'b0, 8'h00, '0, '0);

    print_info("Check synchronous clear initializes the bank");
    @(posedge clk_i);
    #1;
    check_bank_state(1'b0, 96'd0, 40'd0, "clear_initializes_state");
    check_flush_batch(8'h00, 96'd0, 40'd0, "clear_initializes_flush_outputs");

    drive_inputs(
      1'b0,
      1'b1,
      8'hff,
      pack_int11x8(
        11'sd5,  -11'sd6, 11'sd7,  -11'sd8,
        11'sd9,  -11'sd10, 11'sd11, -11'sd12
      ),
      pack_sf5x8(
        5'd3, 5'd4, 5'd5, 5'd6,
        5'd7, 5'd8, 5'd9, 5'd10
      )
    );
    #1;
    check_flush_batch(8'h00, 96'd0, 40'd0, "first_use_ignores_flush_flags");
    @(posedge clk_i);
    #1;
    check_bank_state(
      1'b1,
      pack_int12x8(
        12'sd5,  -12'sd6, 12'sd7,  -12'sd8,
        12'sd9,  -12'sd10, 12'sd11, -12'sd12
      ),
      pack_sf5x8(
        5'd3, 5'd4, 5'd5, 5'd6,
        5'd7, 5'd8, 5'd9, 5'd10
      ),
      "first_use_captures_inputs"
    );

    drive_inputs(
      1'b0,
      1'b1,
      8'h00,
      pack_int11x8(
        11'sd10, 11'sd20, -11'sd3,  11'sd4,
        -11'sd5, 11'sd6,  -11'sd7,  11'sd8
      ),
      pack_sf5x8(
        5'd3, 5'd4, 5'd5, 5'd6,
        5'd7, 5'd8, 5'd9, 5'd10
      )
    );
    #1;
    check_flush_batch(8'h00, 96'd0, 40'd0, "equal_sf_has_no_flush");
    @(posedge clk_i);
    #1;
    check_bank_state(
      1'b1,
      pack_int12x8(
        12'sd15, 12'sd14, 12'sd4,  -12'sd4,
        12'sd4,  -12'sd4, 12'sd4,  -12'sd4
      ),
      pack_sf5x8(
        5'd3, 5'd4, 5'd5, 5'd6,
        5'd7, 5'd8, 5'd9, 5'd10
      ),
      "equal_sf_accumulate"
    );

    drive_inputs(
      1'b0,
      1'b1,
      8'h00,
      pack_int11x8(
        11'sd1,  -11'sd1, 11'sd2,  -11'sd2,
        11'sd3,  -11'sd3, 11'sd4,  -11'sd4
      ),
      pack_sf5x8(
        5'd4, 5'd5, 5'd6, 5'd7,
        5'd8, 5'd9, 5'd10, 5'd11
      )
    );
    @(posedge clk_i);
    #1;
    check_bank_state(
      1'b1,
      pack_int12x8(
        12'sd17, 12'sd12, 12'sd8,  -12'sd8,
        12'sd10, -12'sd10, 12'sd12, -12'sd12
      ),
      pack_sf5x8(
        5'd3, 5'd4, 5'd5, 5'd6,
        5'd7, 5'd8, 5'd9, 5'd10
      ),
      "input_sf_one_larger"
    );

    drive_inputs(
      1'b0,
      1'b1,
      8'h00,
      pack_int11x8(
        -11'sd1, 11'sd2,  -11'sd3, 11'sd4,
        -11'sd5, 11'sd6,  -11'sd7, 11'sd8
      ),
      pack_sf5x8(
        5'd2, 5'd3, 5'd4, 5'd5,
        5'd6, 5'd7, 5'd8, 5'd9
      )
    );
    @(posedge clk_i);
    #1;
    check_bank_state(
      1'b1,
      pack_int12x8(
        12'sd33, 12'sd26, 12'sd13, -12'sd12,
        12'sd15, -12'sd14, 12'sd17, -12'sd16
      ),
      pack_sf5x8(
        5'd2, 5'd3, 5'd4, 5'd5,
        5'd6, 5'd7, 5'd8, 5'd9
      ),
      "stored_sf_one_larger"
    );

    drive_inputs(
      1'b0,
      1'b1,
      8'b10010011,
      pack_int11x8(
        11'sd100, -11'sd101, 11'sd3,   -11'sd4,
        11'sd104, 11'sd5,    -11'sd6,  -11'sd107
      ),
      pack_sf5x8(
        5'd12, 5'd13, 5'd4,  5'd5,
        5'd14, 5'd7,  5'd8,  5'd15
      )
    );
    #1;
    check_flush_batch(
      8'b10010011,
      pack_int12x8(
        12'sd33, 12'sd26, 12'sd13, -12'sd12,
        12'sd15, -12'sd14, 12'sd17, -12'sd16
      ),
      pack_sf5x8(
        5'd2, 5'd3, 5'd4, 5'd5,
        5'd6, 5'd7, 5'd8, 5'd9
      ),
      "flush_snapshot_before_overwrite"
    );
    @(posedge clk_i);
    #1;
    check_bank_state(
      1'b1,
      pack_int12x8(
        12'sd100, -12'sd101, 12'sd16, -12'sd16,
        12'sd104, -12'sd9,   12'sd11, -12'sd107
      ),
      pack_sf5x8(
        5'd12, 5'd13, 5'd4, 5'd5,
        5'd14, 5'd7,  5'd8, 5'd15
      ),
      "flush_overwrite_and_accumulate"
    );

    drive_inputs(
      1'b0,
      1'b0,
      8'hff,
      pack_int11x8(
        11'sd1, 11'sd1, 11'sd1, 11'sd1,
        11'sd1, 11'sd1, 11'sd1, 11'sd1
      ),
      pack_sf5x8(
        5'd1, 5'd1, 5'd1, 5'd1,
        5'd1, 5'd1, 5'd1, 5'd1
      )
    );
    #1;
    check_flush_batch(8'h00, 96'd0, 40'd0, "idle_cycle_has_no_flush");
    @(posedge clk_i);
    #1;
    check_bank_state(
      1'b1,
      pack_int12x8(
        12'sd100, -12'sd101, 12'sd16, -12'sd16,
        12'sd104, -12'sd9,   12'sd11, -12'sd107
      ),
      pack_sf5x8(
        5'd12, 5'd13, 5'd4, 5'd5,
        5'd14, 5'd7,  5'd8, 5'd15
      ),
      "idle_cycle_holds_state"
    );

    drive_inputs(
      1'b1,
      1'b1,
      8'hff,
      pack_int11x8(
        11'sd7, 11'sd7, 11'sd7, 11'sd7,
        11'sd7, 11'sd7, 11'sd7, 11'sd7
      ),
      pack_sf5x8(
        5'd7, 5'd7, 5'd7, 5'd7,
        5'd7, 5'd7, 5'd7, 5'd7
      )
    );
    #1;
    check_flush_batch(8'h00, 96'd0, 40'd0, "clear_blocks_flush_outputs");
    @(posedge clk_i);
    #1;
    check_bank_state(1'b0, 96'd0, 40'd0, "clear_resets_bank");
    check_flush_batch(8'h00, 96'd0, 40'd0, "clear_resets_flush_outputs");

    print_pass("int_acc_flush_bank_tb PASS");
    $finish;
  end

endmodule
