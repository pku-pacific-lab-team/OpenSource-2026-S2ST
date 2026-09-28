// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module packed_4bit_register_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic       clk_i;
  logic [3:0] d_i;
  logic [3:0] q_o;

  packed_4bit_register dut (
    .clk_i (clk_i),
    .d_i   (d_i),
    .q_o   (q_o)
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

  initial begin
    d_i = 4'h3;

    print_info("Check output stays unknown before the first rising edge");
    #2;
    if (q_o !== 4'hx) begin
      print_fail($sformatf("q_o changed before the first rising edge: %h", q_o));
      $fatal(1, "q_o changed before the first rising edge: %h", q_o);
    end
    print_pass("Output stayed unknown before the first rising edge");

    print_info("Check first sample is captured on the first rising edge");
    @(posedge clk_i);
    #1;
    if (q_o !== 4'h3) begin
      print_fail($sformatf("q_o did not capture first sample: %h", q_o));
      $fatal(1, "q_o did not capture first sample: %h", q_o);
    end
    print_pass("First sample captured correctly");

    d_i = 4'ha;
    print_info("Check output does not change between rising edges");
    #2;
    if (q_o !== 4'h3) begin
      print_fail($sformatf("q_o changed without a rising edge: %h", q_o));
      $fatal(1, "q_o changed without a rising edge: %h", q_o);
    end
    print_pass("Output remained stable between rising edges");

    print_info("Check second sample is captured on the next rising edge");
    @(posedge clk_i);
    #1;
    if (q_o !== 4'ha) begin
      print_fail($sformatf("q_o did not capture second sample: %h", q_o));
      $fatal(1, "q_o did not capture second sample: %h", q_o);
    end
    print_pass("Second sample captured correctly");

    print_pass("packed_4bit_register_tb PASS");
    $finish;
  end

endmodule
