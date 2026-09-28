// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module int12_scale_to_bf16_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic signed [11:0] int12_i;
  logic signed  [4:0] scaling_factor_i;
  logic        [15:0] bf16_o;

  int12_scale_to_bf16 dut (
    .int12_i         (int12_i),
    .scaling_factor_i(scaling_factor_i),
    .bf16_o          (bf16_o)
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

  task automatic check_case(
    input logic signed [11:0] test_int12,
    input logic signed  [4:0] test_scale,
    input logic        [15:0] expected_bf16,
    input string               test_name
  );
    begin
      print_info($sformatf(
        "Run test: %s (int12=%0d, scale=%0d)",
        test_name,
        test_int12,
        test_scale
      ));

      int12_i          = test_int12;
      scaling_factor_i = test_scale;
      #1;

      if (bf16_o !== expected_bf16) begin
        print_fail($sformatf(
          "%s mismatch: expected 0x%04h got 0x%04h",
          test_name,
          expected_bf16,
          bf16_o
        ));
        $fatal(
          1,
          "%s mismatch: expected 0x%04h got 0x%04h",
          test_name,
          expected_bf16,
          bf16_o
        );
      end

      print_pass($sformatf("%s matched 0x%04h", test_name, expected_bf16));
    end
  endtask

  initial begin
    int12_i          = '0;
    scaling_factor_i = '0;

    check_case(12'sd0,      5'sd0,  16'h0000, "zero_maps_to_zero");
    check_case(12'sd1,      5'sd0,  16'h3f80, "positive_one");
    check_case(-12'sd1,     5'sd0,  16'hbf80, "negative_one");
    check_case(12'sd1,      5'sd5,  16'h4200, "positive_scaling_factor");
    check_case(12'sd1,     -5'sd3,  16'h3e00, "negative_scaling_factor");
    check_case(12'sd3,      5'sd0,  16'h4040, "fraction_from_int12");
    check_case(-12'sd3,     5'sd2,  16'hc140, "negative_with_scaling");
    check_case(12'sd2047,   5'sd0,  16'h44ff, "truncate_without_rounding");
    check_case(-12'sd2048,  5'sd0,  16'hc500, "minimum_negative_value");

    print_pass("int12_scale_to_bf16_tb PASS");
    $finish;
  end

endmodule
