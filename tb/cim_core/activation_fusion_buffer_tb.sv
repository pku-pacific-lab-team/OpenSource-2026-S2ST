// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module activation_fusion_buffer_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic        clk_i;
  logic        clear_i;
  logic        slots_clear_i;
  logic        shift_refill_i;
  logic [31:0] refill_activation_i;
  logic  [4:0] refill_activation_sf_i;
  logic  [7:0] refill_mask_i;

  logic [31:0] fused_activation_vec_o;
  logic  [4:0] fused_activation_sf_o;
  logic [15:0] fused_row_weight_rsel_o;

  logic [31:0] payload_act [0:3];
  logic  [4:0] payload_sf  [0:3];
  logic  [7:0] payload_mask[0:3];
  logic [15:0] expected_row_weight_rsel;
  logic  [4:0] expected_fused_sf;
  logic  [3:0] expected_lane0_nibble;
  logic  [3:0] expected_lane1_nibble;
  logic  [3:0] expected_lane2_nibble;
  logic  [3:0] expected_lane3_nibble;
  logic [31:0] expected_fused_vec;

  int error_count;

  activation_fusion_buffer dut (
    .clk_i                  (clk_i),
    .clear_i                (clear_i),
    .slots_clear_i          (slots_clear_i),
    .shift_refill_i         (shift_refill_i),
    .refill_activation_i    (refill_activation_i),
    .refill_activation_sf_i (refill_activation_sf_i),
    .refill_mask_i          (refill_mask_i),
    .fused_activation_vec_o (fused_activation_vec_o),
    .fused_activation_sf_o  (fused_activation_sf_o),
    .fused_row_weight_rsel_o(fused_row_weight_rsel_o)
  );

  initial begin
    clk_i = 1'b0;
    forever #1000 clk_i = ~clk_i;
  end

`ifdef FSDB
  initial begin
    $fsdbDumpfile("waveform_activation_fusion_buffer.fsdb");
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

  task automatic expect_eq_32(
    input string       label,
    input logic [31:0] got,
    input logic [31:0] exp
  );
    begin
      if (got !== exp) begin
        error_count = error_count + 1;
        print_fail($sformatf("%s mismatch: got=0x%08h exp=0x%08h", label, got, exp));
      end else begin
        print_pass($sformatf("%s matched: 0x%08h", label, got));
      end
    end
  endtask

  task automatic expect_eq_8(
    input string      label,
    input logic [7:0] got,
    input logic [7:0] exp
  );
    begin
      if (got !== exp) begin
        error_count = error_count + 1;
        print_fail($sformatf("%s mismatch: got=0x%02h exp=0x%02h", label, got, exp));
      end else begin
        print_pass($sformatf("%s matched: 0x%02h", label, got));
      end
    end
  endtask

  task automatic expect_eq_5(
    input string      label,
    input logic [4:0] got,
    input logic [4:0] exp
  );
    begin
      if (got !== exp) begin
        error_count = error_count + 1;
        print_fail($sformatf("%s mismatch: got=0x%02h exp=0x%02h", label, got, exp));
      end else begin
        print_pass($sformatf("%s matched: 0x%02h", label, got));
      end
    end
  endtask

  task automatic expect_eq_4(
    input string      label,
    input logic [3:0] got,
    input logic [3:0] exp
  );
    begin
      if (got !== exp) begin
        error_count = error_count + 1;
        print_fail($sformatf("%s mismatch: got=0x%01h exp=0x%01h", label, got, exp));
      end else begin
        print_pass($sformatf("%s matched: 0x%01h", label, got));
      end
    end
  endtask

  task automatic expect_eq_16(
    input string       label,
    input logic [15:0] got,
    input logic [15:0] exp
  );
    begin
      if (got !== exp) begin
        error_count = error_count + 1;
        print_fail($sformatf("%s mismatch: got=0x%04h exp=0x%04h", label, got, exp));
      end else begin
        print_pass($sformatf("%s matched: 0x%04h", label, got));
      end
    end
  endtask

  task automatic push_payload(
    input logic [31:0] act_word,
    input logic  [4:0] sf_word,
    input logic  [7:0] mask_word
  );
    begin
      refill_activation_i    = act_word;
      refill_activation_sf_i = sf_word;
      refill_mask_i          = mask_word;
      shift_refill_i         = 1'b1;
      @(posedge clk_i);
      shift_refill_i         = 1'b0;
      refill_activation_i    = '0;
      refill_activation_sf_i = '0;
      refill_mask_i          = 8'hFF;
      @(posedge clk_i);
    end
  endtask

  task automatic clear_all;
    begin
      clear_i                = 1'b1;
      slots_clear_i          = 1'b0;
      shift_refill_i         = 1'b0;
      refill_activation_i    = '0;
      refill_activation_sf_i = '0;
      refill_mask_i          = 8'hFF;
      repeat (2) @(posedge clk_i);
      clear_i = 1'b0;
      @(posedge clk_i);
    end
  endtask

  initial begin
    error_count = 0;
    clear_i = 1'b0;
    slots_clear_i = 1'b0;
    shift_refill_i = 1'b0;
    refill_activation_i = '0;
    refill_activation_sf_i = '0;
    refill_mask_i = 8'hFF;

    payload_act[0]  = pack_int4x8(4'sd3, -4'sd8, 4'sd1, -4'sd1, 4'sd0, 4'sd2, -4'sd2, 4'sd7);
    payload_sf[0]   = 5'd1;
    payload_mask[0] = 8'hFD;

    payload_act[1]  = pack_int4x8(-4'sd4, 4'sd0, 4'sd2, 4'sd4, -4'sd6, 4'sd3, 4'sd1, -4'sd1);
    payload_sf[1]   = 5'd4;
    payload_mask[1] = 8'hFF;

    payload_act[2]  = pack_int4x8(4'sd5, -4'sd1, -4'sd4, 4'sd2, 4'sd1, -4'sd3, 4'sd4, -4'sd2);
    payload_sf[2]   = 5'd3;
    payload_mask[2] = 8'hFE;

    payload_act[3]  = pack_int4x8(-4'sd7, 4'sd6, -4'sd3, 4'sd1, 4'sd5, -4'sd4, 4'sd2, 4'sd0);
    payload_sf[3]   = 5'd2;
    payload_mask[3] = 8'hFB;

    print_info("starting activation_fusion_buffer red unit tests");

    clear_all();

    push_payload(payload_act[0], payload_sf[0], payload_mask[0]);
    push_payload(payload_act[1], payload_sf[1], payload_mask[1]);
    push_payload(payload_act[2], payload_sf[2], payload_mask[2]);
    push_payload(payload_act[3], payload_sf[3], payload_mask[3]);

    expect_eq_32("preload_shift_order_slot0_from_payload0", dut.slot_activation_q[0], payload_act[0]);
    expect_eq_32("preload_shift_order_slot1_from_payload1", dut.slot_activation_q[1], payload_act[1]);
    expect_eq_32("preload_shift_order_slot2_from_payload2", dut.slot_activation_q[2], payload_act[2]);
    expect_eq_32("preload_shift_order_slot3_from_payload3", dut.slot_activation_q[3], payload_act[3]);

    expect_eq_5("preload_shift_order_slot0_sf_from_payload0", dut.slot_activation_sf_q[0], payload_sf[0]);
    expect_eq_5("preload_shift_order_slot1_sf_from_payload1", dut.slot_activation_sf_q[1], payload_sf[1]);
    expect_eq_5("preload_shift_order_slot2_sf_from_payload2", dut.slot_activation_sf_q[2], payload_sf[2]);
    expect_eq_5("preload_shift_order_slot3_sf_from_payload3", dut.slot_activation_sf_q[3], payload_sf[3]);

    expect_eq_8("preload_shift_order_slot0_mask_from_payload0", dut.slot_mask_q[0], payload_mask[0]);
    expect_eq_8("preload_shift_order_slot1_mask_from_payload1", dut.slot_mask_q[1], payload_mask[1]);
    expect_eq_8("preload_shift_order_slot2_mask_from_payload2", dut.slot_mask_q[2], payload_mask[2]);
    expect_eq_8("preload_shift_order_slot3_mask_from_payload3", dut.slot_mask_q[3], payload_mask[3]);

    expected_row_weight_rsel = 16'h0032;
    expected_fused_sf        = 5'd3;
    expected_lane0_nibble    = 4'h5;
    expected_lane1_nibble    = 4'hE;
    expected_lane2_nibble    = 4'hE;
    expected_lane3_nibble    = 4'h0;
    expected_fused_vec = {
      16'h0000,
      expected_lane3_nibble,
      expected_lane2_nibble,
      expected_lane1_nibble,
      expected_lane0_nibble
    };

    @(posedge clk_i);
    #1;

    expect_eq_16("mask_only_selection_row_slot_selects", fused_row_weight_rsel_o, expected_row_weight_rsel);
    expect_eq_5("max_sf_alignment_fused_activation_sf", fused_activation_sf_o, expected_fused_sf);
    expect_eq_32("mask_only_selection_fused_activation_vec", fused_activation_vec_o, expected_fused_vec);
    expect_eq_4("lane0_slot2_no_shift_nibble", fused_activation_vec_o[3:0], expected_lane0_nibble);
    expect_eq_4("lane1_slot0_shift_and_truncate_nibble", fused_activation_vec_o[7:4], expected_lane1_nibble);
    expect_eq_4("lane2_slot3_shift_and_truncate_nibble", fused_activation_vec_o[11:8], expected_lane2_nibble);
    expect_eq_4("lane3_invalid_defaults_to_zero_nibble", fused_activation_vec_o[15:12], expected_lane3_nibble);

    if (error_count == 0) begin
      print_pass("all red-test expectations matched");
      $finish;
    end

    print_fail($sformatf("red testbench found %0d mismatches", error_count));
    $fatal(1);
  end

endmodule
