// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module signed_4bit_pe_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic              clk_i;
  logic signed [3:0] activation_i;
  logic              weight_we_i;
  logic        [1:0] weight_wsel_i;
  logic signed [3:0] weight_d_i;
  logic        [1:0] weight_rsel_i;
  logic signed [7:0] product_o;

  signed_4bit_pe dut (
    .clk_i        (clk_i),
    .activation_i (activation_i),
    .weight_we_i  (weight_we_i),
    .weight_wsel_i(weight_wsel_i),
    .weight_d_i   (weight_d_i),
    .weight_rsel_i(weight_rsel_i),
    .product_o    (product_o)
  );

  initial begin
    clk_i = 1'b0;
    forever #5 clk_i = ~clk_i;
  end

  task automatic write_weight(input logic [1:0] sel, input logic signed [3:0] value);
    begin
      print_info($sformatf("Write slot %0d with value %0d", sel, value));
      weight_we_i   = 1'b1;
      weight_wsel_i = sel;
      weight_d_i    = value;
      @(posedge clk_i);
      #1;
      weight_we_i = 1'b0;
      print_pass($sformatf("Slot %0d write completed", sel));
    end
  endtask

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

  initial begin
    activation_i  = '0;
    weight_we_i   = 1'b0;
    weight_wsel_i = '0;
    weight_d_i    = '0;
    weight_rsel_i = '0;

    print_info("Check product starts unknown before any weight is written");
    #1;
    if (product_o !== 8'sbx) begin
      print_fail($sformatf("product_o should start as x before any weight is written: %0d", product_o));
      $fatal(1, "product_o should start as x before any weight is written: %0d", product_o);
    end
    print_pass("Initial product is unknown before programming weights");

    write_weight(2'd0, 4'sd3);
    write_weight(2'd1, -4'sd2);
    write_weight(2'd2, 4'sd5);
    write_weight(2'd3, -4'sd8);

    activation_i = 4'sd4;

    print_info("Check slot 0 product");
    weight_rsel_i = 2'd0;
    #1;
    if (product_o !== 8'sd12) begin
      print_fail($sformatf("slot 0 product mismatch: %0d", product_o));
      $fatal(1, "slot 0 product mismatch: %0d", product_o);
    end
    print_pass("Slot 0 product matches expected value");

    print_info("Check slot 1 product");
    weight_rsel_i = 2'd1;
    #1;
    if (product_o !== -8'sd8) begin
      print_fail($sformatf("slot 1 product mismatch: %0d", product_o));
      $fatal(1, "slot 1 product mismatch: %0d", product_o);
    end
    print_pass("Slot 1 product matches expected value");

    print_info("Check slot 2 product");
    weight_rsel_i = 2'd2;
    #1;
    if (product_o !== 8'sd20) begin
      print_fail($sformatf("slot 2 product mismatch: %0d", product_o));
      $fatal(1, "slot 2 product mismatch: %0d", product_o);
    end
    print_pass("Slot 2 product matches expected value");

    print_info("Check slot 3 product");
    weight_rsel_i = 2'd3;
    #1;
    if (product_o !== -8'sd32) begin
      print_fail($sformatf("slot 3 product mismatch: %0d", product_o));
      $fatal(1, "slot 3 product mismatch: %0d", product_o);
    end
    print_pass("Slot 3 product matches expected value");

    activation_i = -4'sd3;
    weight_rsel_i = 2'd2;
    print_info("Check negative activation multiply behavior");
    #1;
    if (product_o !== -8'sd15) begin
      print_fail($sformatf("negative activation multiply mismatch: %0d", product_o));
      $fatal(1, "negative activation multiply mismatch: %0d", product_o);
    end
    print_pass("Negative activation multiply matches expected value");

    activation_i = -4'sd8;
    weight_rsel_i = 2'd3;
    print_info("Check signed boundary multiply behavior");
    #1;
    if (product_o !== 8'sd64) begin
      print_fail($sformatf("boundary multiply mismatch: %0d", product_o));
      $fatal(1, "boundary multiply mismatch: %0d", product_o);
    end
    print_pass("Signed boundary multiply matches expected value");

    activation_i = 4'sd7;
    write_weight(2'd1, 4'sd1);
    weight_rsel_i = 2'd0;
    print_info("Check writing slot 1 does not disturb slot 0");
    #1;
    if (product_o !== 8'sd21) begin
      print_fail($sformatf("writing slot 1 should not disturb slot 0: %0d", product_o));
      $fatal(1, "writing slot 1 should not disturb slot 0: %0d", product_o);
    end
    print_pass("Writing slot 1 preserved slot 0 contents");

    weight_rsel_i = 2'd1;
    print_info("Check updated slot 1 product");
    #1;
    if (product_o !== 8'sd7) begin
      print_fail($sformatf("updated slot 1 product mismatch: %0d", product_o));
      $fatal(1, "updated slot 1 product mismatch: %0d", product_o);
    end
    print_pass("Updated slot 1 product matches expected value");

    print_pass("signed_4bit_pe_tb PASS");
    $finish;
  end

endmodule
