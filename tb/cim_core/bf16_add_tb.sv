// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module bf16_add_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic [15:0] a_i;
  logic [15:0] b_i;
  logic [15:0] sum_o;

  bf16_add dut (
    .a_i  (a_i),
    .b_i  (b_i),
    .sum_o(sum_o)
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
    input logic [15:0] test_a,
    input logic [15:0] test_b,
    input logic [15:0] expected_sum,
    input string       test_name
  );
    begin
      print_info($sformatf(
        "Run test: %s (a=0x%04h, b=0x%04h)",
        test_name,
        test_a,
        test_b
      ));

      a_i = test_a;
      b_i = test_b;
      #1;

      if (sum_o !== expected_sum) begin
        print_fail($sformatf(
          "%s mismatch: expected 0x%04h got 0x%04h",
          test_name,
          expected_sum,
          sum_o
        ));
        $fatal(
          1,
          "%s mismatch: expected 0x%04h got 0x%04h",
          test_name,
          expected_sum,
          sum_o
        );
      end

      print_pass($sformatf("%s matched 0x%04h", test_name, expected_sum));
    end
  endtask

  initial begin
    a_i = '0;
    b_i = '0;

    check_case(16'h0000, 16'h3f80, 16'h3f80, "zero_plus_one");
    check_case(16'h3f80, 16'h4000, 16'h4040, "one_plus_two");
    check_case(16'hbf80, 16'hbf80, 16'hc000, "negative_one_plus_negative_one");
    check_case(16'h4000, 16'hbf80, 16'h3f80, "two_plus_negative_one");
    check_case(16'h3f80, 16'hbf80, 16'h0000, "cancels_to_zero");
    check_case(16'h3f80, 16'h3c00, 16'h3f81, "align_small_operand");
    check_case(16'h4040, 16'h3f80, 16'h4080, "carry_out_normalizes");
    check_case(16'h4020, 16'hbfc0, 16'h3f80, "subtract_then_left_normalize");

    print_pass("bf16_add_tb PASS");
    $finish;
  end

endmodule
