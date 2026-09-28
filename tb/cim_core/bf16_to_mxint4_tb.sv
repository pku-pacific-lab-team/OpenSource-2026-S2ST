// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module bf16_to_mxint4_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic        valid_i;
  logic [127:0] bf16_vec_i;
  logic [31:0] mxint4_vec_o;
  logic signed [4:0] scaling_factor_o;
  logic [127:0] lane_packing_order_bf16;
  logic [31:0]  lane_packing_order_expected;

  bf16_to_mxint4 dut (
    .valid_i        (valid_i),
    .bf16_vec_i     (bf16_vec_i),
    .mxint4_vec_o   (mxint4_vec_o),
    .scaling_factor_o(scaling_factor_o)
  );

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

  function automatic logic [127:0] pack_bf16x8(
    input logic [15:0] lane0,
    input logic [15:0] lane1,
    input logic [15:0] lane2,
    input logic [15:0] lane3,
    input logic [15:0] lane4,
    input logic [15:0] lane5,
    input logic [15:0] lane6,
    input logic [15:0] lane7
  );
    pack_bf16x8 = {
      lane7, lane6, lane5, lane4,
      lane3, lane2, lane1, lane0
    };
  endfunction

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

  task automatic check_case(
    input string         test_name,
    input logic          valid_value,
    input logic [127:0]  bf16_value,
    input logic [31:0]   expected_mxint4,
    input logic signed [4:0] expected_scale
  );
    begin
      print_info($sformatf("Run test: %s", test_name));
      valid_i    = valid_value;
      bf16_vec_i = bf16_value;
      #1;

      if (mxint4_vec_o !== expected_mxint4) begin
        print_fail($sformatf(
          "%s mxint4 mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_mxint4,
          mxint4_vec_o
        ));
        $fatal(
          1,
          "%s mxint4 mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_mxint4,
          mxint4_vec_o
        );
      end

      if (scaling_factor_o !== expected_scale) begin
        print_fail($sformatf(
          "%s scaling factor mismatch: expected %0d got %0d",
          test_name,
          expected_scale,
          scaling_factor_o
        ));
        $fatal(
          1,
          "%s scaling factor mismatch: expected %0d got %0d",
          test_name,
          expected_scale,
          scaling_factor_o
        );
      end

      print_pass($sformatf("%s passed", test_name));
    end
  endtask

  initial begin
    valid_i        = 1'b0;
    bf16_vec_i     = '0;

    check_case(
      "invalid_forces_zero",
      1'b0,
      pack_bf16x8(
        16'h3f80, 16'hbf80, 16'h3fe0, 16'h0000,
        16'h0000, 16'h0000, 16'h0000, 16'h0000
      ),
      32'h00000000,
      5'sd0
    );

    check_case(
      "all_zero_valid_vector",
      1'b1,
      '0,
      '0,
      5'sd0
    );

    check_case(
      "single_positive_lane",
      1'b1,
      pack_bf16x8(16'h3f80, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000),
      pack_int4x8(4'sd4, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0),
      -5'sd2
    );

    check_case(
      "single_negative_lane",
      1'b1,
      pack_bf16x8(16'hbf80, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000),
      pack_int4x8(-4'sd4, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0),
      -5'sd2
    );

    check_case(
      "mixed_sign_shared_scale",
      1'b1,
      pack_bf16x8(
        16'h3fe0, 16'hbf60, 16'h3f60, 16'h0000,
        16'h0000, 16'h0000, 16'h0000, 16'h0000
      ),
      pack_int4x8(4'sd7, -4'sd3, 4'sd3, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0),
      -5'sd2
    );

    check_case(
      "lower_exponent_lane_quantizes_to_zero",
      1'b1,
      pack_bf16x8(16'h3f80, 16'h3d80, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000),
      pack_int4x8(4'sd4, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0),
      -5'sd2
    );

    check_case(
      "truncation_toward_zero",
      1'b1,
      pack_bf16x8(16'h3fc0, 16'hc108, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000),
      pack_int4x8(4'sd0, -4'sd4, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0),
      5'sd1
    );

    check_case(
      "negative_small_lane_truncates_to_zero",
      1'b1,
      pack_bf16x8(16'h3f80, 16'hbd80, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000, 16'h0000),
      pack_int4x8(4'sd4, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0),
      -5'sd2
    );

    lane_packing_order_bf16 = pack_bf16x8(
      16'h3fe0, 16'h0000, 16'h0000, 16'h0000,
      16'h0000, 16'h0000, 16'h0000, 16'hbf80
    );
    lane_packing_order_expected = pack_int4x8(
      4'sd7, 4'sd0, 4'sd0, 4'sd0,
      4'sd0, 4'sd0, 4'sd0, -4'sd4
    );

    check_case(
      "lane_packing_order",
      1'b1,
      lane_packing_order_bf16,
      lane_packing_order_expected,
      -5'sd2
    );

    valid_i    = 1'b1;
    bf16_vec_i = lane_packing_order_bf16;
    #1;

    print_info("Verify lane_packing_order packing order slices");
    if ($signed(mxint4_vec_o[3:0]) !== 4'sd7) begin
      print_fail($sformatf(
        "lane_packing_order lane0 mismatch: expected %0d got %0d",
        4'sd7,
        $signed(mxint4_vec_o[3:0])
      ));
      $fatal(
        1,
        "lane_packing_order lane0 mismatch: expected %0d got %0d",
        4'sd7,
        $signed(mxint4_vec_o[3:0])
      );
    end

    if ($signed(mxint4_vec_o[31:28]) !== -4'sd4) begin
      print_fail($sformatf(
        "lane_packing_order lane7 mismatch: expected %0d got %0d",
        -4'sd4,
        $signed(mxint4_vec_o[31:28])
      ));
      $fatal(
        1,
        "lane_packing_order lane7 mismatch: expected %0d got %0d",
        -4'sd4,
        $signed(mxint4_vec_o[31:28])
      );
    end

    print_pass("lane_packing_order slices verified");

    print_pass("bf16_to_mxint4_tb PASS");
    $finish;
  end

endmodule
