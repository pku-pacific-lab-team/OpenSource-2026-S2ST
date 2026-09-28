// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module pe_array_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic        clk_i;
  logic        weight_we_i;
  logic  [4:0] weight_addr_i;
  logic [31:0] weight_row_i;
  logic [31:0] activation_vec_i;
  logic [15:0] row_weight_rsel_i;
  logic [87:0] result_vec_o;

  pe_array dut (
    .clk_i           (clk_i),
    .weight_we_i     (weight_we_i),
    .weight_addr_i   (weight_addr_i),
    .weight_row_i    (weight_row_i),
    .activation_vec_i(activation_vec_i),
    .row_weight_rsel_i(row_weight_rsel_i),
    .result_vec_o    (result_vec_o)
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

  function automatic logic signed [10:0] result_lane(
    input logic [87:0] packed_results,
    input int          lane_idx
  );
    result_lane = $signed(packed_results[lane_idx * 11 +: 11]);
  endfunction

  task automatic write_row(
    input logic [2:0] row_idx,
    input logic [1:0] slot_idx,
    input logic [31:0] packed_row
  );
    begin
      print_info($sformatf("Write row %0d slot %0d", row_idx, slot_idx));
      weight_we_i   = 1'b1;
      weight_addr_i = {row_idx, slot_idx};
      weight_row_i  = packed_row;
      @(posedge clk_i);
      #1;
      weight_we_i = 1'b0;
      print_pass($sformatf("Row %0d slot %0d write completed", row_idx, slot_idx));
    end
  endtask

  task automatic check_lane(
    input int                     lane_idx,
    input logic signed [10:0]     expected_value,
    input string                  test_name
  );
    logic signed [10:0] observed_value;
    begin
      observed_value = result_lane(result_vec_o, lane_idx);
      if (observed_value !== expected_value) begin
        print_fail($sformatf(
          "%s lane %0d mismatch: expected %0d got %0d",
          test_name,
          lane_idx,
          expected_value,
          observed_value
        ));
        $fatal(
          1,
          "%s lane %0d mismatch: expected %0d got %0d",
          test_name,
          lane_idx,
          expected_value,
          observed_value
        );
      end
    end
  endtask

  task automatic check_all_lanes(
    input logic signed [10:0] expected0,
    input logic signed [10:0] expected1,
    input logic signed [10:0] expected2,
    input logic signed [10:0] expected3,
    input logic signed [10:0] expected4,
    input logic signed [10:0] expected5,
    input logic signed [10:0] expected6,
    input logic signed [10:0] expected7,
    input string              test_name
  );
    begin
      print_info($sformatf("Run test: %s", test_name));
      check_lane(0, expected0, test_name);
      check_lane(1, expected1, test_name);
      check_lane(2, expected2, test_name);
      check_lane(3, expected3, test_name);
      check_lane(4, expected4, test_name);
      check_lane(5, expected5, test_name);
      check_lane(6, expected6, test_name);
      check_lane(7, expected7, test_name);
      print_pass($sformatf("Test passed: %s", test_name));
    end
  endtask

  integer row_idx;

  initial begin
    weight_we_i       = 1'b0;
    weight_addr_i     = '0;
    weight_row_i      = '0;
    activation_vec_i  = '0;
    row_weight_rsel_i = '0;

    for (row_idx = 0; row_idx < 8; row_idx++) begin
      write_row(row_idx[2:0], 2'd0, pack_int4x8(
        4'sd0, 4'sd0, 4'sd0, 4'sd0,
        4'sd0, 4'sd0, 4'sd0, 4'sd0
      ));
    end

    activation_vec_i  = pack_int4x8(
      4'sd0, 4'sd0, 4'sd0, 4'sd0,
      4'sd0, 4'sd0, 4'sd0, 4'sd0
    );
    row_weight_rsel_i = pack_rsel8(
      2'd0, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    #1;
    check_all_lanes(
      11'sd0, 11'sd0, 11'sd0, 11'sd0,
      11'sd0, 11'sd0, 11'sd0, 11'sd0,
      "zero_init"
    );

    write_row(3'd3, 2'd0, pack_int4x8(
      4'sd1, -4'sd2, 4'sd3, -4'sd4,
      4'sd5, -4'sd6, 4'sd7, -4'sd8
    ));
    activation_vec_i = pack_int4x8(
      4'sd0, 4'sd0, 4'sd0, 4'sd2,
      4'sd0, 4'sd0, 4'sd0, 4'sd0
    );
    #1;
    check_all_lanes(
      11'sd2, -11'sd4, 11'sd6, -11'sd8,
      11'sd10, -11'sd12, 11'sd14, -11'sd16,
      "row_write_isolation"
    );

    write_row(3'd2, 2'd1, pack_int4x8(
      4'sd1, 4'sd1, 4'sd1, 4'sd1,
      4'sd1, 4'sd1, 4'sd1, 4'sd1
    ));
    write_row(3'd2, 2'd2, pack_int4x8(
      -4'sd1, -4'sd1, -4'sd1, -4'sd1,
      -4'sd1, -4'sd1, -4'sd1, -4'sd1
    ));
    activation_vec_i = pack_int4x8(
      4'sd0, 4'sd0, 4'sd3, 4'sd0,
      4'sd0, 4'sd0, 4'sd0, 4'sd0
    );
    row_weight_rsel_i = pack_rsel8(
      2'd0, 2'd0, 2'd1, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    #1;
    check_all_lanes(
      11'sd3, 11'sd3, 11'sd3, 11'sd3,
      11'sd3, 11'sd3, 11'sd3, 11'sd3,
      "slot_select_row2_slot1"
    );

    row_weight_rsel_i = pack_rsel8(
      2'd0, 2'd0, 2'd2, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    #1;
    check_all_lanes(
      -11'sd3, -11'sd3, -11'sd3, -11'sd3,
      -11'sd3, -11'sd3, -11'sd3, -11'sd3,
      "slot_select_row2_slot2"
    );

    write_row(3'd0, 2'd0, pack_int4x8(
      4'sd1, 4'sd0, -4'sd1, 4'sd2,
      -4'sd2, 4'sd3, -4'sd3, 4'sd4
    ));
    write_row(3'd1, 2'd0, pack_int4x8(
      4'sd0, 4'sd1, 4'sd2, -4'sd1,
      -4'sd2, 4'sd3, 4'sd4, -4'sd3
    ));
    write_row(3'd2, 2'd0, pack_int4x8(
      4'sd1, 4'sd1, 4'sd1, 4'sd1,
      4'sd1, 4'sd1, 4'sd1, 4'sd1
    ));
    write_row(3'd3, 2'd0, pack_int4x8(
      -4'sd1, 4'sd2, -4'sd3, 4'sd4,
      -4'sd5, 4'sd6, -4'sd7, 4'sd0
    ));
    write_row(3'd4, 2'd0, pack_int4x8(
      4'sd2, -4'sd2, 4'sd2, -4'sd2,
      4'sd2, -4'sd2, 4'sd2, -4'sd2
    ));
    write_row(3'd5, 2'd0, pack_int4x8(
      4'sd3, 4'sd0, -4'sd3, 4'sd0,
      4'sd3, 4'sd0, -4'sd3, 4'sd0
    ));
    write_row(3'd6, 2'd0, pack_int4x8(
      -4'sd4, 4'sd4, 4'sd0, 4'sd1,
      -4'sd1, 4'sd2, -4'sd2, 4'sd3
    ));
    write_row(3'd7, 2'd0, pack_int4x8(
      4'sd7, -4'sd6, 4'sd5, -4'sd4,
      4'sd3, -4'sd2, 4'sd1, 4'sd0
    ));
    activation_vec_i = pack_int4x8(
      4'sd1, -4'sd2, 4'sd3, -4'sd4,
      4'sd5, -4'sd6, 4'sd7, -4'sd8
    );
    row_weight_rsel_i = pack_rsel8(
      2'd0, 2'd0, 2'd0, 2'd0,
      2'd0, 2'd0, 2'd0, 2'd0
    );
    #1;
    check_all_lanes(
      -11'sd84, 11'sd59, -11'sd2, 11'sd20,
      -11'sd14, -11'sd4, 11'sd26, 11'sd24,
      "full_matrix_vector"
    );

    for (row_idx = 0; row_idx < 8; row_idx++) begin
      write_row(row_idx[2:0], 2'd3, pack_int4x8(
        -4'sd8, 4'sd0, 4'sd0, 4'sd0,
        4'sd0, 4'sd0, 4'sd0, 4'sd0
      ));
    end
    activation_vec_i = pack_int4x8(
      -4'sd8, -4'sd8, -4'sd8, -4'sd8,
      -4'sd8, -4'sd8, -4'sd8, -4'sd8
    );
    row_weight_rsel_i = pack_rsel8(
      2'd3, 2'd3, 2'd3, 2'd3,
      2'd3, 2'd3, 2'd3, 2'd3
    );
    #1;
    check_all_lanes(
      11'sd512, 11'sd0, 11'sd0, 11'sd0,
      11'sd0, 11'sd0, 11'sd0, 11'sd0,
      "positive_boundary_512"
    );

    print_pass("pe_array_tb PASS");
    $finish;
  end

endmodule
