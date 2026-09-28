// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module pe_array_buffered_hybrid_top_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  localparam logic [8:0] WEIGHT_BASE_A    = 9'd16;
  localparam logic [8:0] WEIGHT_BASE_B    = 9'd80;
  localparam logic [8:0] ACT_ADDR_A       = 9'd20;
  localparam logic [8:0] ACT_ADDR_B       = 9'd21;
  localparam logic [5:0] WEIGHT_SF_ADDR_A = 6'd3;
  localparam logic [5:0] WEIGHT_SF_ADDR_B = 6'd4;
  localparam logic [8:0] REORDER_ACT_BASE = 9'd40;
  localparam logic [5:0] REORDER_WSF_BASE = 6'd16;
  localparam logic [8:0] WRITEBACK_ADDR_A = 9'd200;
  localparam logic [8:0] WRITEBACK_ADDR_B = 9'd201;
  localparam logic [8:0] FUSION_PRELOAD_BASE = 9'd300;
  localparam logic [7:0] THRESHOLD_PARTIAL = 8'd129;
  localparam logic [7:0] THRESHOLD_NONE    = 8'd0;

  logic              clk_i;
  logic              rst_ni;
  logic              clear_i;

  logic              weight_buf_en_i;
  logic              weight_buf_wen_i;
  logic        [8:0] weight_buf_addr_i;
  logic       [31:0] weight_buf_wdata_i;

  logic              weight_sf_buf_en_i;
  logic              weight_sf_buf_wen_i;
  logic        [5:0] weight_sf_buf_addr_i;
  logic       [39:0] weight_sf_buf_wdata_i;

  logic              activation_buf_en_i;
  logic              activation_buf_wen_i;
  logic        [8:0] activation_buf_addr_i;
  logic       [31:0] activation_buf_wdata_i;

  logic              activation_sf_buf_en_i;
  logic              activation_sf_buf_wen_i;
  logic        [8:0] activation_sf_buf_addr_i;
  logic        [4:0] activation_sf_buf_wdata_i;
  logic              mask_buf_en_i;
  logic              mask_buf_wen_i;
  logic        [8:0] mask_buf_addr_i;
  logic        [7:0] mask_buf_wdata_i;

  logic              weight_load_start_i;
  logic        [8:0] weight_load_base_addr_i;
  logic              fusion_preload_start_i;
  logic              fusion_mode_en_i;

  logic              compute_start_i;
  logic        [8:0] activation_addr_i;
  logic        [5:0] weight_sf_addr_i;
  logic       [15:0] row_weight_rsel_i;
  logic        [7:0] fp_flush_flag_i;
  logic              writeback_enable_i;
  logic        [8:0] writeback_addr_i;
  logic        [7:0] threshold_i;
  logic              use_reorder_seq_i;
  logic              reorder_ptr_reset_i;
  logic              reorder_seq_we_i;
  logic       [23:0] reorder_seq_i;

  logic              busy_o;
  logic              weight_load_done_o;
  logic              compute_done_o;

  logic              ref_weight_we;
  logic        [4:0] ref_weight_addr;
  logic       [31:0] ref_weight_row;
  logic       [31:0] ref_activation_vec;
  logic       [15:0] ref_row_weight_rsel;
  logic       [87:0] ref_result_vec;

  logic signed [11:0] ref_int12_i;
  logic signed  [4:0] ref_sf_i;
  logic        [15:0] ref_bf16_o;
  logic        [15:0] ref_bf16_add_a;
  logic        [15:0] ref_bf16_add_b;
  logic        [15:0] ref_bf16_add_sum;
  logic               ref_quant_valid;
  logic       [127:0] ref_quant_bf16_vec;
  logic        [31:0] ref_quant_mxint4_vec;
  logic signed  [4:0] ref_quant_scaling_factor;

  logic       [31:0] weight_block_a [0:63];
  logic       [31:0] weight_block_b [0:63];
  logic       [31:0] activation_word_a;
  logic       [31:0] activation_word_b;
  logic       [31:0] reorder_activation_word [0:7];
  logic       [31:0] fusion_activation_word [0:7];
  logic signed  [4:0] reorder_activation_sf_word [0:7];
  logic signed  [4:0] fusion_activation_sf_word [0:7];
  logic        [7:0] fusion_mask_word [0:7];
  logic       [39:0] weight_sf_word_a;
  logic       [39:0] weight_sf_word_b;
  logic       [39:0] reorder_weight_sf_word [0:7];
  logic signed  [4:0] activation_sf_a;
  logic signed  [4:0] activation_sf_b;
  logic       [15:0] row_rsel_zero;
  logic       [39:0] expected_sf_vec;
  logic [15:0] expected_fp_lane [0:7];
  logic [15:0] expected_compute_a [0:7];
  logic [15:0] expected_compute_b [0:7];
  logic [15:0] expected_final_ab [0:7];
  logic  [3:0] pe_weight_slot_obs [0:7][0:7][0:3];
  logic [31:0] expected_writeback_word;
  logic signed [4:0] expected_writeback_sf;
  logic  [7:0] expected_writeback_mask;
  logic [31:0] expected_fused_activation_a;
  logic [31:0] expected_fused_activation_b;
  logic [39:0] expected_fused_sf_vec_a;
  logic [39:0] expected_fused_sf_vec_b;
  logic [15:0] expected_fused_rsel_a;
  logic [15:0] expected_fused_rsel_b;
  logic  [3:0] expected_fused_act_lane_a [0:7];
  logic  [3:0] expected_fused_act_lane_b [0:7];
  logic  [1:0] expected_fused_rsel_lane_a [0:7];
  logic  [1:0] expected_fused_rsel_lane_b [0:7];
  logic [31:0] activation_shadow_mem [0:511];
  logic  [4:0] activation_sf_shadow_mem [0:511];
  logic  [7:0] mask_shadow_mem [0:511];
  logic [23:0] reorder_seq_pattern;

  integer idx;

  pe_array_buffered_hybrid_top dut (
    .clk_i                   (clk_i),
    .rst_ni                  (rst_ni),
    .clear_i                 (clear_i),
    .weight_buf_en_i         (weight_buf_en_i),
    .weight_buf_wen_i        (weight_buf_wen_i),
    .weight_buf_addr_i       (weight_buf_addr_i),
    .weight_buf_wdata_i      (weight_buf_wdata_i),
    .weight_sf_buf_en_i      (weight_sf_buf_en_i),
    .weight_sf_buf_wen_i     (weight_sf_buf_wen_i),
    .weight_sf_buf_addr_i    (weight_sf_buf_addr_i),
    .weight_sf_buf_wdata_i   (weight_sf_buf_wdata_i),
    .activation_buf_en_i     (activation_buf_en_i),
    .activation_buf_wen_i    (activation_buf_wen_i),
    .activation_buf_addr_i   (activation_buf_addr_i),
    .activation_buf_wdata_i  (activation_buf_wdata_i),
    .activation_sf_buf_en_i  (activation_sf_buf_en_i),
    .activation_sf_buf_wen_i (activation_sf_buf_wen_i),
    .activation_sf_buf_addr_i(activation_sf_buf_addr_i),
    .activation_sf_buf_wdata_i(activation_sf_buf_wdata_i),
    .mask_buf_en_i           (mask_buf_en_i),
    .mask_buf_wen_i          (mask_buf_wen_i),
    .mask_buf_addr_i         (mask_buf_addr_i),
    .mask_buf_wdata_i        (mask_buf_wdata_i),
    .weight_load_start_i     (weight_load_start_i),
    .weight_load_base_addr_i (weight_load_base_addr_i),
    .fusion_preload_start_i  (fusion_preload_start_i),
    .fusion_mode_en_i        (fusion_mode_en_i),
    .compute_start_i         (compute_start_i),
    .activation_addr_i       (activation_addr_i),
    .weight_sf_addr_i        (weight_sf_addr_i),
    .row_weight_rsel_i       (row_weight_rsel_i),
    .fp_flush_flag_i         (fp_flush_flag_i),
    .writeback_enable_i      (writeback_enable_i),
    .writeback_addr_i        (writeback_addr_i),
    .threshold_i             (threshold_i),
    .use_reorder_seq_i       (use_reorder_seq_i),
    .reorder_ptr_reset_i     (reorder_ptr_reset_i),
    .reorder_seq_we_i        (reorder_seq_we_i),
    .reorder_seq_i           (reorder_seq_i),
    .busy_o                  (busy_o),
    .weight_load_done_o      (weight_load_done_o),
    .compute_done_o          (compute_done_o)
  );

  pe_array ref_pe_array (
    .clk_i             (clk_i),
    .weight_we_i       (ref_weight_we),
    .weight_addr_i     (ref_weight_addr),
    .weight_row_i      (ref_weight_row),
    .activation_vec_i  (ref_activation_vec),
    .row_weight_rsel_i (ref_row_weight_rsel),
    .result_vec_o      (ref_result_vec)
  );

  int12_scale_to_bf16 ref_converter (
    .int12_i          (ref_int12_i),
    .scaling_factor_i (ref_sf_i),
    .bf16_o           (ref_bf16_o)
  );

  bf16_add ref_bf16_adder (
    .a_i   (ref_bf16_add_a),
    .b_i   (ref_bf16_add_b),
    .sum_o (ref_bf16_add_sum)
  );

  bf16_to_mxint4 ref_quantizer (
    .valid_i          (ref_quant_valid),
    .bf16_vec_i       (ref_quant_bf16_vec),
    .mxint4_vec_o     (ref_quant_mxint4_vec),
    .scaling_factor_o (ref_quant_scaling_factor)
  );

  genvar obs_row_idx;
  genvar obs_col_idx;
  genvar obs_slot_idx;
  generate
    for (obs_row_idx = 0; obs_row_idx < 8; obs_row_idx++) begin : gen_obs_rows
      for (obs_col_idx = 0; obs_col_idx < 8; obs_col_idx++) begin : gen_obs_cols
        for (obs_slot_idx = 0; obs_slot_idx < 4; obs_slot_idx++) begin : gen_obs_slots
          assign pe_weight_slot_obs[obs_row_idx][obs_col_idx][obs_slot_idx] =
            dut.hybrid_acc_u.pe_array_u.gen_rows[obs_row_idx]
               .gen_cols[obs_col_idx]
               .pe_u.weight_slot_q[obs_slot_idx];
        end
      end
    end
  endgenerate

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b1;
    forever #1000 clk_i = ~clk_i;
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
    input logic signed [4:0] lane0,
    input logic signed [4:0] lane1,
    input logic signed [4:0] lane2,
    input logic signed [4:0] lane3,
    input logic signed [4:0] lane4,
    input logic signed [4:0] lane5,
    input logic signed [4:0] lane6,
    input logic signed [4:0] lane7
  );
    pack_sf5x8 = {
      lane7[4:0], lane6[4:0], lane5[4:0], lane4[4:0],
      lane3[4:0], lane2[4:0], lane1[4:0], lane0[4:0]
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

  function automatic logic [23:0] pack_seq8(
    input logic [2:0] seq0,
    input logic [2:0] seq1,
    input logic [2:0] seq2,
    input logic [2:0] seq3,
    input logic [2:0] seq4,
    input logic [2:0] seq5,
    input logic [2:0] seq6,
    input logic [2:0] seq7
  );
    pack_seq8 = {
      seq7, seq6, seq5, seq4,
      seq3, seq2, seq1, seq0
    };
  endfunction

  function automatic logic signed [10:0] result_lane(
    input logic [87:0] packed_results,
    input int          lane_idx
  );
    result_lane = $signed(packed_results[lane_idx * 11 +: 11]);
  endfunction

  function automatic logic signed [5:0] sf_sum_ext(
    input logic signed [4:0] weight_sf_lane,
    input logic signed [4:0] activation_sf
  );
    sf_sum_ext = $signed({weight_sf_lane[4], weight_sf_lane})
               + $signed({activation_sf[4], activation_sf});
  endfunction

  function automatic logic [39:0] build_sf_vec(
    input logic [39:0]       packed_weight_sf,
    input logic signed [4:0] activation_sf
  );
    logic signed [4:0] weight_sf_lane;
    logic signed [5:0] lane_sum;
    integer            lane_idx;
    begin
      build_sf_vec = '0;

      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        weight_sf_lane = $signed(packed_weight_sf[lane_idx * 5 +: 5]);
        lane_sum = sf_sum_ext(weight_sf_lane, activation_sf);
        build_sf_vec[lane_idx * 5 +: 5] = lane_sum[4:0];
      end
    end
  endfunction

  function automatic logic [3:0] weight_lane_nibble(
    input logic [31:0] weight_word,
    input int          lane_idx
  );
    weight_lane_nibble = weight_word[lane_idx * 4 +: 4];
  endfunction

  function automatic logic [2:0] packed_seq_elem(
    input logic [23:0] packed_seq,
    input int          seq_idx
  );
    begin
      unique case (seq_idx)
        0: packed_seq_elem = packed_seq[2:0];
        1: packed_seq_elem = packed_seq[5:3];
        2: packed_seq_elem = packed_seq[8:6];
        3: packed_seq_elem = packed_seq[11:9];
        4: packed_seq_elem = packed_seq[14:12];
        5: packed_seq_elem = packed_seq[17:15];
        6: packed_seq_elem = packed_seq[20:18];
        7: packed_seq_elem = packed_seq[23:21];
        default: packed_seq_elem = 3'bxxx;
      endcase
    end
  endfunction

  function automatic logic [31:0] selected_weight_word(
    input bit use_block_b,
    input int word_idx
  );
    selected_weight_word = use_block_b ? weight_block_b[word_idx] : weight_block_a[word_idx];
  endfunction

  function automatic logic [31:0] activation_mem_word(input logic [8:0] addr);
    activation_mem_word = activation_shadow_mem[addr];
  endfunction

  function automatic logic [4:0] activation_sf_mem_word(input logic [8:0] addr);
    activation_sf_mem_word = activation_sf_shadow_mem[addr];
  endfunction

  function automatic logic [7:0] mask_mem_word(input logic [8:0] addr);
    mask_mem_word = mask_shadow_mem[addr];
  endfunction

  task automatic check_child_address_contracts;
    begin
      if ($bits(dut.weight_buf_addr_i) != 9) begin
        $fatal(1, "weight_buf_addr_i width mismatch: expected 9, got %0d", $bits(dut.weight_buf_addr_i));
      end
      if ($bits(dut.weight_sf_buf_addr_i) != 6) begin
        $fatal(1, "weight_sf_buf_addr_i width mismatch: expected 6, got %0d", $bits(dut.weight_sf_buf_addr_i));
      end
      if ($bits(dut.activation_addr_i) != 9) begin
        $fatal(1, "activation_addr_i width mismatch: expected 9, got %0d", $bits(dut.activation_addr_i));
      end
      if ($bits(dut.writeback_addr_i) != 9) begin
        $fatal(1, "writeback_addr_i width mismatch: expected 9, got %0d", $bits(dut.writeback_addr_i));
      end
      if ($bits(dut.weight_sf_addr_i) != 6) begin
        $fatal(1, "weight_sf_addr_i width mismatch: expected 6, got %0d", $bits(dut.weight_sf_addr_i));
      end
    end
  endtask

  always_ff @(posedge clk_i) begin
    if (dut.activation_data_en && dut.activation_data_write) begin
      activation_shadow_mem[dut.activation_data_addr] <= dut.activation_data_wdata;
    end

    if (dut.activation_sf_mem_en && dut.activation_sf_mem_write) begin
      activation_sf_shadow_mem[dut.activation_sf_mem_addr] <= dut.activation_sf_mem_wdata;
    end

    if (dut.mask_mem_en && dut.mask_mem_write) begin
      mask_shadow_mem[dut.mask_mem_addr] <= dut.mask_mem_wdata;
    end
  end

  task automatic reset_inputs;
    begin
      rst_ni                     = 1'b1;
      clear_i                   = 1'b0;

      weight_buf_en_i           = 1'b0;
      weight_buf_wen_i          = 1'b0;
      weight_buf_addr_i         = '0;
      weight_buf_wdata_i        = '0;

      weight_sf_buf_en_i        = 1'b0;
      weight_sf_buf_wen_i       = 1'b0;
      weight_sf_buf_addr_i      = '0;
      weight_sf_buf_wdata_i     = '0;

      activation_buf_en_i       = 1'b0;
      activation_buf_wen_i      = 1'b0;
      activation_buf_addr_i     = '0;
      activation_buf_wdata_i    = '0;

      activation_sf_buf_en_i    = 1'b0;
      activation_sf_buf_wen_i   = 1'b0;
      activation_sf_buf_addr_i  = '0;
      activation_sf_buf_wdata_i = '0;
      mask_buf_en_i             = 1'b0;
      mask_buf_wen_i            = 1'b0;
      mask_buf_addr_i           = '0;
      mask_buf_wdata_i          = '0;

      weight_load_start_i       = 1'b0;
      weight_load_base_addr_i   = '0;
      fusion_preload_start_i    = 1'b0;
      fusion_mode_en_i          = 1'b0;

      compute_start_i           = 1'b0;
      activation_addr_i         = '0;
      weight_sf_addr_i          = '0;
      row_weight_rsel_i         = '0;
      fp_flush_flag_i           = '0;
      writeback_enable_i        = 1'b0;
      writeback_addr_i          = '0;
      threshold_i               = '0;
      use_reorder_seq_i         = 1'b0;
      reorder_ptr_reset_i       = 1'b0;
      reorder_seq_we_i          = 1'b0;
      reorder_seq_i             = '0;

      ref_weight_we             = 1'b0;
      ref_weight_addr           = '0;
      ref_weight_row            = '0;
      ref_activation_vec        = '0;
      ref_row_weight_rsel       = '0;
      ref_int12_i               = '0;
      ref_sf_i                  = '0;
      ref_bf16_add_a            = '0;
      ref_bf16_add_b            = '0;
      ref_quant_valid           = 1'b0;
      ref_quant_bf16_vec        = '0;
    end
  endtask

  task automatic apply_async_reset;
    begin
      @(negedge clk_i);
      rst_ni = 1'b0;
      @(negedge clk_i);
      rst_ni = 1'b1;
      @(posedge clk_i);
    end
  endtask

  task automatic check_reorder_seq_retention(
    input logic [23:0] expected_seq,
    input string       test_name
  );
    begin
      if (dut.reorder_seq_q !== expected_seq) begin
        print_fail($sformatf(
          "%s reorder_seq_q mismatch: expected 0x%06h got 0x%06h",
          test_name,
          expected_seq,
          dut.reorder_seq_q
        ));
        $fatal(1, "%s reorder_seq_q mismatch: expected 0x%06h got 0x%06h", test_name, expected_seq, dut.reorder_seq_q);
      end

      print_pass($sformatf("Test passed: %s", test_name));
    end
  endtask

  task automatic check_reorder_ptr(
    input logic [2:0] expected_ptr,
    input string      test_name
  );
    begin
      if (dut.reorder_ptr_q !== expected_ptr) begin
        print_fail($sformatf(
          "%s reorder_ptr_q mismatch: expected %0d got %0d",
          test_name,
          expected_ptr,
          dut.reorder_ptr_q
        ));
        $fatal(1, "%s reorder_ptr_q mismatch: expected %0d got %0d", test_name, expected_ptr, dut.reorder_ptr_q);
      end
    end
  endtask

  task automatic init_test_vectors;
    begin
      row_rsel_zero    = pack_rsel8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0);
      activation_word_a = pack_int4x8(4'sd1, 4'sd2, 4'sd3, 4'sd4, -4'sd1, -4'sd2, -4'sd3, -4'sd4);
      activation_word_b = pack_int4x8(-4'sd1, 4'sd1, -4'sd2, 4'sd2, -4'sd3, 4'sd3, -4'sd4, 4'sd4);
      weight_sf_word_a  = pack_sf5x8(5'sd0, 5'sd1, -5'sd1, 5'sd2, -5'sd2, 5'sd3, -5'sd3, 5'sd4);
      weight_sf_word_b  = pack_sf5x8(5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2);
      activation_sf_a   = 5'sd1;
      activation_sf_b   = -5'sd1;
      reorder_activation_word[0] = pack_int4x8(4'sd1, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      reorder_activation_word[1] = pack_int4x8(4'sd0, 4'sd2, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      reorder_activation_word[2] = pack_int4x8(4'sd0, 4'sd0, 4'sd3, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      reorder_activation_word[3] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd4, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      reorder_activation_word[4] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, -4'sd1, 4'sd0, 4'sd0, 4'sd0);
      reorder_activation_word[5] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, -4'sd2, 4'sd0, 4'sd0);
      reorder_activation_word[6] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, -4'sd3, 4'sd0);
      reorder_activation_word[7] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, -4'sd4);
      reorder_activation_sf_word[0] = 5'sd0;
      reorder_activation_sf_word[1] = 5'sd1;
      reorder_activation_sf_word[2] = 5'sd2;
      reorder_activation_sf_word[3] = 5'sd3;
      reorder_activation_sf_word[4] = -5'sd1;
      reorder_activation_sf_word[5] = -5'sd2;
      reorder_activation_sf_word[6] = -5'sd3;
      reorder_activation_sf_word[7] = 5'sd4;
      reorder_weight_sf_word[0] = pack_sf5x8(5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0);
      reorder_weight_sf_word[1] = pack_sf5x8(5'sd1, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0);
      reorder_weight_sf_word[2] = pack_sf5x8(5'sd1, 5'sd1, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0);
      reorder_weight_sf_word[3] = pack_sf5x8(5'sd1, 5'sd1, 5'sd1, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0);
      reorder_weight_sf_word[4] = pack_sf5x8(5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd1, 5'sd0, 5'sd0, 5'sd0);
      reorder_weight_sf_word[5] = pack_sf5x8(5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd1, 5'sd0, 5'sd0);
      reorder_weight_sf_word[6] = pack_sf5x8(5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd1, 5'sd0);
      reorder_weight_sf_word[7] = pack_sf5x8(5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd1);
      fusion_activation_word[0] = pack_int4x8(4'sd7, 4'sd0, 4'sd0, 4'sd0, -4'sd8, 4'sd0, 4'sd0, 4'sd0);
      fusion_activation_word[1] = pack_int4x8(4'sd0, 4'sd6, 4'sd0, 4'sd0, 4'sd0, 4'sd0, -4'sd4, 4'sd0);
      fusion_activation_word[2] = pack_int4x8(4'sd0, 4'sd0, 4'sd4, 4'sd0, 4'sd0, 4'sd0, 4'sd0, -4'sd2);
      fusion_activation_word[3] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd3, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      fusion_activation_word[4] = pack_int4x8(4'sd1, 4'sd2, 4'sd3, 4'sd4, 4'sd5, 4'sd6, 4'sd7, -4'sd8);
      fusion_activation_word[5] = pack_int4x8(-4'sd1, -4'sd2, -4'sd3, -4'sd4, -4'sd5, -4'sd6, -4'sd7, 4'sd7);
      fusion_activation_word[6] = pack_int4x8(4'sd2, 4'sd2, 4'sd2, 4'sd2, 4'sd2, 4'sd2, 4'sd2, 4'sd2);
      fusion_activation_word[7] = pack_int4x8(-4'sd2, -4'sd2, -4'sd2, -4'sd2, -4'sd2, -4'sd2, -4'sd2, -4'sd2);
      fusion_activation_sf_word[0] = 5'sd0;
      fusion_activation_sf_word[1] = 5'sd1;
      fusion_activation_sf_word[2] = 5'sd2;
      fusion_activation_sf_word[3] = 5'sd3;
      fusion_activation_sf_word[4] = 5'sd0;
      fusion_activation_sf_word[5] = -5'sd1;
      fusion_activation_sf_word[6] = 5'sd1;
      fusion_activation_sf_word[7] = 5'sd0;
      fusion_mask_word[0] = 8'hEE;
      fusion_mask_word[1] = 8'hBD;
      fusion_mask_word[2] = 8'h7B;
      fusion_mask_word[3] = 8'hF7;
      fusion_mask_word[4] = 8'hFF;
      fusion_mask_word[5] = 8'hFF;
      fusion_mask_word[6] = 8'h00;
      fusion_mask_word[7] = 8'h00;
      expected_fused_act_lane_a[0] = 4'sd0;
      expected_fused_act_lane_a[1] = 4'sd1;
      expected_fused_act_lane_a[2] = 4'sd2;
      expected_fused_act_lane_a[3] = 4'sd3;
      expected_fused_act_lane_a[4] = -4'sd1;
      expected_fused_act_lane_a[5] = 4'sd0;
      expected_fused_act_lane_a[6] = -4'sd1;
      expected_fused_act_lane_a[7] = -4'sd1;
      expected_fused_act_lane_b[0] = 4'sd0;
      expected_fused_act_lane_b[1] = 4'sd1;
      expected_fused_act_lane_b[2] = 4'sd2;
      expected_fused_act_lane_b[3] = 4'sd3;
      expected_fused_act_lane_b[4] = 4'sd0;
      expected_fused_act_lane_b[5] = 4'sd0;
      expected_fused_act_lane_b[6] = -4'sd1;
      expected_fused_act_lane_b[7] = -4'sd1;
      expected_fused_activation_a = pack_int4x8(
        expected_fused_act_lane_a[0], expected_fused_act_lane_a[1],
        expected_fused_act_lane_a[2], expected_fused_act_lane_a[3],
        expected_fused_act_lane_a[4], expected_fused_act_lane_a[5],
        expected_fused_act_lane_a[6], expected_fused_act_lane_a[7]
      );
      expected_fused_activation_b = pack_int4x8(
        expected_fused_act_lane_b[0], expected_fused_act_lane_b[1],
        expected_fused_act_lane_b[2], expected_fused_act_lane_b[3],
        expected_fused_act_lane_b[4], expected_fused_act_lane_b[5],
        expected_fused_act_lane_b[6], expected_fused_act_lane_b[7]
      );
      expected_fused_sf_vec_a     = build_sf_vec(weight_sf_word_a, 5'sd3);
      expected_fused_sf_vec_b     = build_sf_vec(weight_sf_word_b, 5'sd3);
      expected_fused_rsel_lane_a[0] = 2'd0;
      expected_fused_rsel_lane_a[1] = 2'd1;
      expected_fused_rsel_lane_a[2] = 2'd2;
      expected_fused_rsel_lane_a[3] = 2'd3;
      expected_fused_rsel_lane_a[4] = 2'd0;
      expected_fused_rsel_lane_a[5] = 2'd0;
      expected_fused_rsel_lane_a[6] = 2'd1;
      expected_fused_rsel_lane_a[7] = 2'd2;
      expected_fused_rsel_lane_b[0] = 2'd0;
      expected_fused_rsel_lane_b[1] = 2'd0;
      expected_fused_rsel_lane_b[2] = 2'd1;
      expected_fused_rsel_lane_b[3] = 2'd2;
      expected_fused_rsel_lane_b[4] = 2'd0;
      expected_fused_rsel_lane_b[5] = 2'd0;
      expected_fused_rsel_lane_b[6] = 2'd0;
      expected_fused_rsel_lane_b[7] = 2'd1;
      expected_fused_rsel_a       = pack_rsel8(
        expected_fused_rsel_lane_a[0], expected_fused_rsel_lane_a[1],
        expected_fused_rsel_lane_a[2], expected_fused_rsel_lane_a[3],
        expected_fused_rsel_lane_a[4], expected_fused_rsel_lane_a[5],
        expected_fused_rsel_lane_a[6], expected_fused_rsel_lane_a[7]
      );
      expected_fused_rsel_b       = pack_rsel8(
        expected_fused_rsel_lane_b[0], expected_fused_rsel_lane_b[1],
        expected_fused_rsel_lane_b[2], expected_fused_rsel_lane_b[3],
        expected_fused_rsel_lane_b[4], expected_fused_rsel_lane_b[5],
        expected_fused_rsel_lane_b[6], expected_fused_rsel_lane_b[7]
      );

      for (idx = 0; idx < 64; idx++) begin
        weight_block_a[idx] = idx * 32'h11111111;
        weight_block_b[idx] = 32'hfedcba98 - (idx * 32'h01010101);
      end

      weight_block_a[6'd0] = pack_int4x8(4'sd1, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      weight_block_a[6'd1] = pack_int4x8(4'sd0, 4'sd1, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      weight_block_a[6'd2] = pack_int4x8(4'sd0, 4'sd0, 4'sd1, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      weight_block_a[6'd3] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd1, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      weight_block_a[6'd4] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd1, 4'sd0, 4'sd0, 4'sd0);
      weight_block_a[6'd5] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd1, 4'sd0, 4'sd0);
      weight_block_a[6'd6] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd1, 4'sd0);
      weight_block_a[6'd7] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd1);

      weight_block_b[6'd0] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd1);
      weight_block_b[6'd1] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd1, 4'sd0);
      weight_block_b[6'd2] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd1, 4'sd0, 4'sd0);
      weight_block_b[6'd3] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd1, 4'sd0, 4'sd0, 4'sd0);
      weight_block_b[6'd4] = pack_int4x8(4'sd0, 4'sd0, 4'sd0, 4'sd1, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      weight_block_b[6'd5] = pack_int4x8(4'sd0, 4'sd0, 4'sd1, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      weight_block_b[6'd6] = pack_int4x8(4'sd0, 4'sd1, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
      weight_block_b[6'd7] = pack_int4x8(4'sd1, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0, 4'sd0);
    end
  endtask

  task automatic check_accumulator_cleared(input string test_name);
    integer lane_idx;
    begin
      if (dut.hybrid_acc_u.int_bank_valid !== 1'b0) begin
        print_fail($sformatf("%s int bank should be cleared", test_name));
        $fatal(1, "%s int bank should be cleared", test_name);
      end

      if (dut.hybrid_fp_path_busy !== 1'b0) begin
        print_fail($sformatf("%s fp path should be idle", test_name));
        $fatal(1, "%s fp path should be idle", test_name);
      end

      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        if (dut.hybrid_fp_result[lane_idx] !== 16'h0000) begin
          print_fail($sformatf(
            "%s lane %0d should be cleared but got 0x%04h",
            test_name,
            lane_idx,
            dut.hybrid_fp_result[lane_idx]
          ));
          $fatal(
            1,
            "%s lane %0d should be cleared but got 0x%04h",
            test_name,
            lane_idx,
            dut.hybrid_fp_result[lane_idx]
          );
        end
      end
    end
  endtask

  task automatic clear_wrapper;
    begin
      print_info("Apply synchronous clear");
      @(negedge clk_i);
      clear_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      clear_i = 1'b0;
      @(posedge clk_i);

      if (busy_o !== 1'b0) begin
        print_fail("busy_o should be low immediately after clear");
        $fatal(1, "busy_o should be low immediately after clear");
      end

      check_accumulator_cleared("clear_wrapper");
    end
  endtask

  task automatic write_weight_buf(
    input logic [8:0]  addr,
    input logic [31:0] data
  );
    begin
      @(negedge clk_i);
      weight_buf_en_i    = 1'b1;
      weight_buf_wen_i   = 1'b1;
      weight_buf_addr_i  = addr;
      weight_buf_wdata_i = data;
      @(posedge clk_i);
      @(negedge clk_i);
      weight_buf_en_i    = 1'b0;
      weight_buf_wen_i   = 1'b0;
      weight_buf_addr_i  = '0;
      weight_buf_wdata_i = '0;
    end
  endtask

  task automatic write_weight_sf_buf(
    input logic [5:0]  addr,
    input logic [39:0] data
  );
    begin
      @(negedge clk_i);
      weight_sf_buf_en_i    = 1'b1;
      weight_sf_buf_wen_i   = 1'b1;
      weight_sf_buf_addr_i  = addr;
      weight_sf_buf_wdata_i = data;
      @(posedge clk_i);
      @(negedge clk_i);
      weight_sf_buf_en_i    = 1'b0;
      weight_sf_buf_wen_i   = 1'b0;
      weight_sf_buf_addr_i  = '0;
      weight_sf_buf_wdata_i = '0;
    end
  endtask

  task automatic write_activation_buf(
    input logic [8:0]  addr,
    input logic [31:0] data
  );
    begin
      @(negedge clk_i);
      activation_buf_en_i    = 1'b1;
      activation_buf_wen_i   = 1'b1;
      activation_buf_addr_i  = addr;
      activation_buf_wdata_i = data;
      @(posedge clk_i);
      @(negedge clk_i);
      activation_buf_en_i    = 1'b0;
      activation_buf_wen_i   = 1'b0;
      activation_buf_addr_i  = '0;
      activation_buf_wdata_i = '0;
    end
  endtask

  task automatic write_activation_sf_buf(
    input logic [8:0]        addr,
    input logic signed [4:0] data
  );
    begin
      @(negedge clk_i);
      activation_sf_buf_en_i     = 1'b1;
      activation_sf_buf_wen_i    = 1'b1;
      activation_sf_buf_addr_i   = addr;
      activation_sf_buf_wdata_i  = data[4:0];
      @(posedge clk_i);
      @(negedge clk_i);
      activation_sf_buf_en_i     = 1'b0;
      activation_sf_buf_wen_i    = 1'b0;
      activation_sf_buf_addr_i   = '0;
      activation_sf_buf_wdata_i  = '0;
    end
  endtask

  task automatic write_mask_buf(
    input logic [8:0] addr,
    input logic [7:0] data
  );
    begin
      @(negedge clk_i);
      mask_buf_en_i    = 1'b1;
      mask_buf_wen_i   = 1'b1;
      mask_buf_addr_i  = addr;
      mask_buf_wdata_i = data;
      @(posedge clk_i);
      @(negedge clk_i);
      mask_buf_en_i    = 1'b0;
      mask_buf_wen_i   = 1'b0;
      mask_buf_addr_i  = '0;
      mask_buf_wdata_i = '0;
    end
  endtask

  task automatic write_ref_weight(
    input logic [4:0]  addr,
    input logic [31:0] data
  );
    begin
      @(negedge clk_i);
      ref_weight_we   = 1'b1;
      ref_weight_addr = addr;
      ref_weight_row  = data;
      @(posedge clk_i);
      @(negedge clk_i);
      ref_weight_we   = 1'b0;
      ref_weight_addr = '0;
      ref_weight_row  = '0;
    end
  endtask

  task automatic load_ref_block(input bit use_block_b);
    begin
      for (idx = 0; idx < 32; idx++) begin
        write_ref_weight({idx[2:0], idx[4:3]}, selected_weight_word(use_block_b, idx));
      end
    end
  endtask

  task automatic program_weight_block(
    input logic [8:0] base_addr,
    input bit         use_block_b
  );
    begin
      for (idx = 0; idx < 64; idx++) begin
        write_weight_buf(base_addr + idx[8:0], selected_weight_word(use_block_b, idx));
      end
    end
  endtask

  task automatic check_loaded_weight_entry(
    input int          load_entry_idx,
    input logic [31:0] expected_word,
    input string       test_name
  );
    integer row_idx;
    integer slot_idx;
    integer col_idx;
    logic [3:0] expected_nibble;
    logic [3:0] observed_nibble;
    begin
      row_idx  = load_entry_idx % 8;
      slot_idx = load_entry_idx / 8;

      for (col_idx = 0; col_idx < 8; col_idx++) begin
        expected_nibble = weight_lane_nibble(expected_word, col_idx);
        observed_nibble = pe_weight_slot_obs[row_idx][col_idx][slot_idx];

        if (observed_nibble !== expected_nibble) begin
          print_fail($sformatf(
            "%s row %0d col %0d slot %0d mismatch: expected 0x%01h got 0x%01h",
            test_name,
            row_idx,
            col_idx,
            slot_idx,
            expected_nibble,
            observed_nibble
          ));
          $fatal(
            1,
            "%s row %0d col %0d slot %0d mismatch: expected 0x%01h got 0x%01h",
            test_name,
            row_idx,
            col_idx,
            slot_idx,
            expected_nibble,
            observed_nibble
          );
        end
      end
    end
  endtask

  task automatic check_weight_load(
    input logic [8:0] base_addr,
    input bit         use_block_b,
    input string      test_name
  );
    logic [31:0] expected_word;
    begin
      print_info($sformatf("Run test: %s", test_name));

      @(negedge clk_i);
      weight_load_base_addr_i = base_addr;
      weight_load_start_i     = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      weight_load_start_i     = 1'b0;
      weight_load_base_addr_i = '0;

      for (idx = 0; idx < 32; idx++) begin
        @(posedge clk_i);

        if (busy_o !== 1'b1) begin
          print_fail($sformatf("%s busy_o should stay high during load cycle %0d", test_name, idx));
          $fatal(1, "%s busy_o should stay high during load cycle %0d", test_name, idx);
        end

        if (weight_load_done_o !== 1'b0) begin
          print_fail($sformatf("%s weight_load_done_o should stay low during load cycle %0d", test_name, idx));
          $fatal(1, "%s weight_load_done_o should stay low during load cycle %0d", test_name, idx);
        end

        if (compute_done_o !== 1'b0) begin
          print_fail($sformatf("%s compute_done_o should stay low during weight load", test_name));
          $fatal(1, "%s compute_done_o should stay low during weight load", test_name);
        end

        if (idx != 0) begin
          expected_word = use_block_b ? weight_block_b[idx - 1] : weight_block_a[idx - 1];
          check_loaded_weight_entry(idx - 1, expected_word, test_name);
        end
      end

      @(posedge clk_i);

      expected_word = use_block_b ? weight_block_b[31] : weight_block_a[31];
      check_loaded_weight_entry(31, expected_word, test_name);

      if (busy_o !== 1'b0) begin
        print_fail($sformatf("%s busy_o should be low after final write", test_name));
        $fatal(1, "%s busy_o should be low after final write", test_name);
      end

      if (weight_load_done_o !== 1'b1) begin
        print_fail($sformatf("%s weight_load_done_o should pulse after final write", test_name));
        $fatal(1, "%s weight_load_done_o should pulse after final write", test_name);
      end

      @(posedge clk_i);

      if (weight_load_done_o !== 1'b0) begin
        print_fail($sformatf("%s weight_load_done_o should deassert after one cycle", test_name));
        $fatal(1, "%s weight_load_done_o should deassert after one cycle", test_name);
      end

      print_pass($sformatf("Test passed: %s", test_name));
    end
  endtask

  task automatic write_reorder_seq(input logic [23:0] packed_seq);
    begin
      @(negedge clk_i);
      reorder_seq_i    = packed_seq;
      reorder_seq_we_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      reorder_seq_we_i = 1'b0;
      reorder_seq_i    = '0;
    end
  endtask

  task automatic pulse_reorder_ptr_reset(input string test_name);
    begin
      @(negedge clk_i);
      reorder_ptr_reset_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      reorder_ptr_reset_i = 1'b0;
      check_reorder_ptr(3'd0, test_name);
    end
  endtask

  task automatic program_reorder_compute_window(
    input logic [8:0] act_base_addr,
    input logic [5:0] wsf_base_addr
  );
    begin
      for (idx = 0; idx < 8; idx++) begin
        write_activation_buf(act_base_addr + idx[8:0], reorder_activation_word[idx]);
        write_activation_sf_buf(act_base_addr + idx[8:0], reorder_activation_sf_word[idx]);
        write_weight_sf_buf(wsf_base_addr + idx[5:0], reorder_weight_sf_word[idx]);
      end
    end
  endtask

  task automatic check_weight_load_reordered(
    input logic [8:0]  base_addr,
    input bit          use_block_b,
    input logic [23:0] packed_seq,
    input logic [2:0]  ptr_value,
    input string       test_name
  );
    integer      row_idx;
    integer      slot_idx;
    integer      src_word_idx;
    logic  [2:0] seq_value;
    logic [31:0] expected_word;
    logic  [8:0] expected_addr;
    begin
      print_info($sformatf("Run test: %s", test_name));

      seq_value     = packed_seq_elem(packed_seq, ptr_value);
      expected_addr = base_addr + {{4{1'b0}}, seq_value, 3'b000};

      @(negedge clk_i);
      weight_load_base_addr_i = base_addr;
      use_reorder_seq_i       = 1'b1;
      weight_load_start_i     = 1'b1;
      @(posedge clk_i);

      if (dut.weight_data_addr !== expected_addr) begin
        print_fail($sformatf(
          "%s first weight addr mismatch: expected %0d got %0d",
          test_name,
          expected_addr,
          dut.weight_data_addr
        ));
        $fatal(1, "%s first weight addr mismatch: expected %0d got %0d", test_name, expected_addr, dut.weight_data_addr);
      end

      @(negedge clk_i);
      weight_load_start_i     = 1'b0;
      weight_load_base_addr_i = '0;
      use_reorder_seq_i       = 1'b0;

      for (idx = 0; idx < 32; idx++) begin
        @(posedge clk_i);

        if (busy_o !== 1'b1) begin
          print_fail($sformatf("%s busy_o should stay high during load cycle %0d", test_name, idx));
          $fatal(1, "%s busy_o should stay high during load cycle %0d", test_name, idx);
        end

        if (weight_load_done_o !== 1'b0) begin
          print_fail($sformatf("%s weight_load_done_o should stay low during load cycle %0d", test_name, idx));
          $fatal(1, "%s weight_load_done_o should stay low during load cycle %0d", test_name, idx);
        end

        if (idx != 31) begin
          row_idx      = (idx + 1) % 8;
          slot_idx     = (idx + 1) / 8;
          seq_value    = packed_seq_elem(packed_seq, (ptr_value + slot_idx) & 3'h7);
          expected_addr = base_addr + {{4{1'b0}}, seq_value, 3'b000} + row_idx;
          if (dut.weight_data_addr !== expected_addr) begin
            print_fail($sformatf(
              "%s weight addr mismatch at load entry %0d: expected %0d got %0d",
              test_name,
              idx + 1,
              expected_addr,
              dut.weight_data_addr
            ));
            $fatal(1, "%s weight addr mismatch at load entry %0d: expected %0d got %0d", test_name, idx + 1, expected_addr, dut.weight_data_addr);
          end
        end

        if (idx != 0) begin
          row_idx      = (idx - 1) % 8;
          slot_idx     = (idx - 1) / 8;
          seq_value    = packed_seq_elem(packed_seq, (ptr_value + slot_idx) & 3'h7);
          src_word_idx = (seq_value * 8) + row_idx;
          expected_word = selected_weight_word(use_block_b, src_word_idx);
          check_loaded_weight_entry(idx - 1, expected_word, test_name);
        end
      end

      @(posedge clk_i);
      row_idx      = 31 % 8;
      slot_idx     = 31 / 8;
      seq_value    = packed_seq_elem(packed_seq, (ptr_value + slot_idx) & 3'h7);
      src_word_idx = (seq_value * 8) + row_idx;
      expected_word = selected_weight_word(use_block_b, src_word_idx);
      check_loaded_weight_entry(31, expected_word, test_name);

      if (busy_o !== 1'b0) begin
        print_fail($sformatf("%s busy_o should be low after final write", test_name));
        $fatal(1, "%s busy_o should be low after final write", test_name);
      end

      if (weight_load_done_o !== 1'b1) begin
        print_fail($sformatf("%s weight_load_done_o should pulse after final write", test_name));
        $fatal(1, "%s weight_load_done_o should pulse after final write", test_name);
      end

      check_reorder_ptr(ptr_value, $sformatf("%s_ptr_hold", test_name));

      @(posedge clk_i);

      if (weight_load_done_o !== 1'b0) begin
        print_fail($sformatf("%s weight_load_done_o should deassert after one cycle", test_name));
        $fatal(1, "%s weight_load_done_o should deassert after one cycle", test_name);
      end

      print_pass($sformatf("Test passed: %s", test_name));
    end
  endtask

  task automatic run_reordered_compute_no_writeback_and_check(
    input logic [8:0]  act_addr,
    input logic [5:0]  wsf_addr,
    input logic [23:0] packed_seq,
    input logic [2:0]  ptr_value,
    input logic [15:0] packed_rsel,
    input logic [7:0]  packed_flush,
    input logic [8:0]  quiet_addr,
    input string       test_name
  );
    logic [31:0] expected_activation;
    logic [39:0] expected_sf;
    logic [31:0] initial_data_word;
    logic  [4:0] initial_sf_word;
    logic  [7:0] initial_mask_word;
    logic  [2:0] seq_value;
    logic  [8:0] expected_act_addr;
    logic  [5:0] expected_wsf_addr;
    begin
      print_info($sformatf("Run test: %s", test_name));

      seq_value          = packed_seq_elem(packed_seq, ptr_value);
      expected_activation = reorder_activation_word[seq_value];
      expected_sf         = build_sf_vec(reorder_weight_sf_word[seq_value], reorder_activation_sf_word[seq_value]);
      expected_act_addr   = act_addr + {{7{1'b0}}, seq_value};
      expected_wsf_addr   = wsf_addr + {{4{1'b0}}, seq_value};

      initial_data_word = activation_mem_word(quiet_addr);
      initial_sf_word   = activation_sf_mem_word(quiet_addr);
      initial_mask_word = mask_mem_word(quiet_addr);

      @(negedge clk_i);
      activation_addr_i   = act_addr;
      weight_sf_addr_i    = wsf_addr;
      row_weight_rsel_i   = packed_rsel;
      fp_flush_flag_i     = packed_flush;
      writeback_enable_i  = 1'b0;
      writeback_addr_i    = quiet_addr;
      threshold_i         = THRESHOLD_NONE;
      use_reorder_seq_i   = 1'b1;
      compute_start_i     = 1'b1;
      @(posedge clk_i);

      if (dut.activation_data_addr !== expected_act_addr) begin
        print_fail($sformatf(
          "%s activation addr mismatch: expected %0d got %0d",
          test_name,
          expected_act_addr,
          dut.activation_data_addr
        ));
        $fatal(1, "%s activation addr mismatch: expected %0d got %0d", test_name, expected_act_addr, dut.activation_data_addr);
      end

      if (dut.activation_sf_mem_addr !== expected_act_addr) begin
        print_fail($sformatf(
          "%s activation_sf addr mismatch: expected %0d got %0d",
          test_name,
          expected_act_addr,
          dut.activation_sf_mem_addr
        ));
        $fatal(1, "%s activation_sf addr mismatch: expected %0d got %0d", test_name, expected_act_addr, dut.activation_sf_mem_addr);
      end

      if (dut.weight_sf_mem_addr !== expected_wsf_addr) begin
        print_fail($sformatf(
          "%s weight_sf addr mismatch: expected %0d got %0d",
          test_name,
          expected_wsf_addr,
          dut.weight_sf_mem_addr
        ));
        $fatal(1, "%s weight_sf addr mismatch: expected %0d got %0d", test_name, expected_wsf_addr, dut.weight_sf_mem_addr);
      end

      @(negedge clk_i);
      compute_start_i     = 1'b0;
      activation_addr_i   = '0;
      weight_sf_addr_i    = '0;
      row_weight_rsel_i   = '0;
      fp_flush_flag_i     = '0;
      writeback_enable_i  = 1'b0;
      writeback_addr_i    = '0;
      threshold_i         = '0;
      use_reorder_seq_i   = 1'b0;

      @(posedge clk_i);
      check_reorder_ptr((ptr_value + 3'd1) & 3'h7, $sformatf("%s_ptr_inc", test_name));

      if (busy_o !== 1'b1) begin
        print_fail($sformatf("%s busy_o should be high during execute cycle", test_name));
        $fatal(1, "%s busy_o should be high during execute cycle", test_name);
      end

      if (compute_done_o !== 1'b0) begin
        print_fail($sformatf("%s compute_done_o should stay low during execute cycle", test_name));
        $fatal(1, "%s compute_done_o should stay low during execute cycle", test_name);
      end

      if (dut.hybrid_compute_accumulate !== 1'b1) begin
        print_fail($sformatf("%s compute_accumulate should pulse in execute cycle", test_name));
        $fatal(1, "%s compute_accumulate should pulse in execute cycle", test_name);
      end

      if (dut.hybrid_activation_vec !== expected_activation) begin
        print_fail($sformatf(
          "%s activation mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_activation,
          dut.hybrid_activation_vec
        ));
        $fatal(1, "%s activation mismatch: expected 0x%08h got 0x%08h", test_name, expected_activation, dut.hybrid_activation_vec);
      end

      if (dut.hybrid_sf_vec !== expected_sf) begin
        print_fail($sformatf(
          "%s sf_vec mismatch: expected 0x%010h got 0x%010h",
          test_name,
          expected_sf,
          dut.hybrid_sf_vec
        ));
        $fatal(1, "%s sf_vec mismatch: expected 0x%010h got 0x%010h", test_name, expected_sf, dut.hybrid_sf_vec);
      end

      @(posedge clk_i);

      if (busy_o !== 1'b0) begin
        print_fail($sformatf("%s busy_o should drop after no-writeback compute", test_name));
        $fatal(1, "%s busy_o should drop after no-writeback compute", test_name);
      end

      if (compute_done_o !== 1'b1) begin
        print_fail($sformatf("%s compute_done_o should pulse immediately for no-writeback compute", test_name));
        $fatal(1, "%s compute_done_o should pulse immediately for no-writeback compute", test_name);
      end

      if (activation_mem_word(quiet_addr) !== initial_data_word) begin
        print_fail($sformatf("%s should not write activation SRAM", test_name));
        $fatal(1, "%s should not write activation SRAM", test_name);
      end

      if (activation_sf_mem_word(quiet_addr) !== initial_sf_word) begin
        print_fail($sformatf("%s should not write activation-sf SRAM", test_name));
        $fatal(1, "%s should not write activation-sf SRAM", test_name);
      end

      if (mask_mem_word(quiet_addr) !== initial_mask_word) begin
        print_fail($sformatf("%s should not write mask SRAM", test_name));
        $fatal(1, "%s should not write mask SRAM", test_name);
      end

      @(posedge clk_i);

      if (compute_done_o !== 1'b0) begin
        print_fail($sformatf("%s compute_done_o should deassert after one cycle", test_name));
        $fatal(1, "%s compute_done_o should deassert after one cycle", test_name);
      end

      print_pass($sformatf("Test passed: %s", test_name));
    end
  endtask

  task automatic build_expected_fp(
    input logic [31:0] activation_word,
    input logic [15:0] packed_rsel,
    input logic [39:0] packed_sf_vec
  );
    logic signed [10:0] lane_int11;
    integer             lane_idx;
    begin
      ref_activation_vec  = activation_word;
      ref_row_weight_rsel = packed_rsel;
      #1;

      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        lane_int11  = result_lane(ref_result_vec, lane_idx);
        ref_int12_i = $signed({lane_int11[10], lane_int11});
        ref_sf_i    = $signed(packed_sf_vec[lane_idx * 5 +: 5]);
        #1;
        expected_fp_lane[lane_idx] = ref_bf16_o;
      end
    end
  endtask

  task automatic build_pairwise_bf16_sum(
    input  logic [15:0] left_lanes [0:7],
    input  logic [15:0] right_lanes[0:7],
    output logic [15:0] sum_lanes  [0:7]
  );
    integer lane_idx;
    begin
      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        ref_bf16_add_a = left_lanes[lane_idx];
        ref_bf16_add_b = right_lanes[lane_idx];
        #1;
        sum_lanes[lane_idx] = ref_bf16_add_sum;
      end
    end
  endtask

  task automatic build_expected_writeback(
    input  logic [15:0] bf16_lanes [0:7],
    input  logic  [7:0] threshold_value,
    output logic [31:0] expected_mxint4,
    output logic signed [4:0] expected_scale,
    output logic  [7:0] expected_mask
  );
    integer lane_idx;
    begin
      expected_mask   = 8'h00;
      ref_quant_valid = 1'b1;
      ref_quant_bf16_vec = '0;

      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        if (bf16_lanes[lane_idx][14:7] < threshold_value) begin
          expected_mask[lane_idx] = 1'b1;
          ref_quant_bf16_vec[lane_idx * 16 +: 16] = 16'h0000;
        end else begin
          ref_quant_bf16_vec[lane_idx * 16 +: 16] = bf16_lanes[lane_idx];
        end
      end

      #1;
      expected_mxint4 = ref_quant_mxint4_vec;
      expected_scale  = ref_quant_scaling_factor;
      ref_quant_valid = 1'b0;
    end
  endtask

  task automatic run_compute_no_writeback_and_check(
    input logic [8:0]  act_addr,
    input logic [5:0]  wsf_addr,
    input logic [15:0] packed_rsel,
    input logic [7:0]  packed_flush,
    input logic [31:0] expected_activation,
    input logic [39:0] expected_sf,
    input logic [8:0]  quiet_addr,
    input string       test_name
  );
    logic [31:0] initial_data_word;
    logic  [4:0] initial_sf_word;
    logic  [7:0] initial_mask_word;
    begin
      print_info($sformatf("Run test: %s", test_name));
      initial_data_word = activation_mem_word(quiet_addr);
      initial_sf_word   = activation_sf_mem_word(quiet_addr);
      initial_mask_word = mask_mem_word(quiet_addr);

      @(negedge clk_i);
      activation_addr_i   = act_addr;
      weight_sf_addr_i    = wsf_addr;
      row_weight_rsel_i   = packed_rsel;
      fp_flush_flag_i     = packed_flush;
      writeback_enable_i  = 1'b0;
      writeback_addr_i    = quiet_addr;
      threshold_i         = THRESHOLD_NONE;
      compute_start_i     = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      compute_start_i     = 1'b0;
      activation_addr_i   = '0;
      weight_sf_addr_i    = '0;
      row_weight_rsel_i   = '0;
      fp_flush_flag_i     = '0;
      writeback_enable_i  = 1'b0;
      writeback_addr_i    = '0;
      threshold_i         = '0;

      @(posedge clk_i);

      if (busy_o !== 1'b1) begin
        print_fail($sformatf("%s busy_o should be high during execute cycle", test_name));
        $fatal(1, "%s busy_o should be high during execute cycle", test_name);
      end

      if (compute_done_o !== 1'b0) begin
        print_fail($sformatf("%s compute_done_o should stay low during execute cycle", test_name));
        $fatal(1, "%s compute_done_o should stay low during execute cycle", test_name);
      end

      if (dut.hybrid_compute_accumulate !== 1'b1) begin
        print_fail($sformatf("%s compute_accumulate should pulse in execute cycle", test_name));
        $fatal(1, "%s compute_accumulate should pulse in execute cycle", test_name);
      end

      if (dut.hybrid_activation_vec !== expected_activation) begin
        print_fail($sformatf(
          "%s activation mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_activation,
          dut.hybrid_activation_vec
        ));
        $fatal(
          1,
          "%s activation mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_activation,
          dut.hybrid_activation_vec
        );
      end

      if (dut.hybrid_sf_vec !== expected_sf) begin
        print_fail($sformatf(
          "%s sf_vec mismatch: expected 0x%010h got 0x%010h",
          test_name,
          expected_sf,
          dut.hybrid_sf_vec
        ));
        $fatal(
          1,
          "%s sf_vec mismatch: expected 0x%010h got 0x%010h",
          test_name,
          expected_sf,
          dut.hybrid_sf_vec
        );
      end

      @(posedge clk_i);

      if (busy_o !== 1'b0) begin
        print_fail($sformatf("%s busy_o should drop after no-writeback compute", test_name));
        $fatal(1, "%s busy_o should drop after no-writeback compute", test_name);
      end

      if (compute_done_o !== 1'b1) begin
        print_fail($sformatf("%s compute_done_o should pulse immediately for no-writeback compute", test_name));
        $fatal(1, "%s compute_done_o should pulse immediately for no-writeback compute", test_name);
      end

      if (activation_mem_word(quiet_addr) !== initial_data_word) begin
        print_fail($sformatf("%s should not write activation SRAM", test_name));
        $fatal(1, "%s should not write activation SRAM", test_name);
      end

      if (activation_sf_mem_word(quiet_addr) !== initial_sf_word) begin
        print_fail($sformatf("%s should not write activation-sf SRAM", test_name));
        $fatal(1, "%s should not write activation-sf SRAM", test_name);
      end

      if (mask_mem_word(quiet_addr) !== initial_mask_word) begin
        print_fail($sformatf("%s should not write mask SRAM", test_name));
        $fatal(1, "%s should not write mask SRAM", test_name);
      end

      @(posedge clk_i);

      if (compute_done_o !== 1'b0) begin
        print_fail($sformatf("%s compute_done_o should deassert after one cycle", test_name));
        $fatal(1, "%s compute_done_o should deassert after one cycle", test_name);
      end

      print_pass($sformatf("Test passed: %s", test_name));
    end
  endtask

  task automatic run_compute_writeback_and_check(
    input logic [8:0]  act_addr,
    input logic [5:0]  wsf_addr,
    input logic [15:0] packed_rsel,
    input logic [7:0]  packed_flush,
    input logic [7:0]  threshold_value,
    input logic [8:0]  wb_addr,
    input logic [31:0] expected_activation,
    input logic [39:0] expected_sf,
    input logic [31:0] expected_data_word,
    input logic  [4:0] expected_sf_word,
    input logic  [7:0] expected_mask_word,
    input int          min_wait_cycles,
    input string       test_name
  );
    logic [31:0] initial_data_word;
    logic  [4:0] initial_sf_word;
    logic  [7:0] initial_mask_word;
    integer      wait_cycles;
    begin
      print_info($sformatf("Run test: %s", test_name));
      initial_data_word = activation_mem_word(wb_addr);
      initial_sf_word   = activation_sf_mem_word(wb_addr);
      initial_mask_word = mask_mem_word(wb_addr);

      @(negedge clk_i);
      activation_addr_i   = act_addr;
      weight_sf_addr_i    = wsf_addr;
      row_weight_rsel_i   = packed_rsel;
      fp_flush_flag_i     = packed_flush;
      writeback_enable_i  = 1'b1;
      writeback_addr_i    = wb_addr;
      threshold_i         = threshold_value;
      compute_start_i     = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      compute_start_i     = 1'b0;
      activation_addr_i   = '0;
      weight_sf_addr_i    = '0;
      row_weight_rsel_i   = '0;
      fp_flush_flag_i     = '0;
      writeback_enable_i  = 1'b0;
      writeback_addr_i    = '0;
      threshold_i         = '0;

      @(posedge clk_i);

      if (busy_o !== 1'b1) begin
        print_fail($sformatf("%s busy_o should be high during execute cycle", test_name));
        $fatal(1, "%s busy_o should be high during execute cycle", test_name);
      end

      if (dut.hybrid_compute_accumulate !== 1'b1) begin
        print_fail($sformatf("%s compute_accumulate should pulse in execute cycle", test_name));
        $fatal(1, "%s compute_accumulate should pulse in execute cycle", test_name);
      end

      if (dut.hybrid_activation_vec !== expected_activation) begin
        print_fail($sformatf(
          "%s activation mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_activation,
          dut.hybrid_activation_vec
        ));
        $fatal(
          1,
          "%s activation mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_activation,
          dut.hybrid_activation_vec
        );
      end

      if (dut.hybrid_sf_vec !== expected_sf) begin
        print_fail($sformatf(
          "%s sf_vec mismatch: expected 0x%010h got 0x%010h",
          test_name,
          expected_sf,
          dut.hybrid_sf_vec
        ));
        $fatal(
          1,
          "%s sf_vec mismatch: expected 0x%010h got 0x%010h",
          test_name,
          expected_sf,
          dut.hybrid_sf_vec
        );
      end

      repeat (min_wait_cycles) begin
        @(posedge clk_i);

        if (compute_done_o !== 1'b0) begin
          print_fail($sformatf("%s compute_done_o should stay low before writeback completes", test_name));
          $fatal(1, "%s compute_done_o should stay low before writeback completes", test_name);
        end

        if (busy_o !== 1'b1) begin
          print_fail($sformatf("%s busy_o should stay high before writeback completes", test_name));
          $fatal(1, "%s busy_o should stay high before writeback completes", test_name);
        end

        if (activation_mem_word(wb_addr) !== initial_data_word) begin
          print_fail($sformatf("%s activation SRAM changed too early", test_name));
          $fatal(1, "%s activation SRAM changed too early", test_name);
        end

        if (activation_sf_mem_word(wb_addr) !== initial_sf_word) begin
          print_fail($sformatf("%s activation-sf SRAM changed too early", test_name));
          $fatal(1, "%s activation-sf SRAM changed too early", test_name);
        end

        if (mask_mem_word(wb_addr) !== initial_mask_word) begin
          print_fail($sformatf("%s mask SRAM changed too early", test_name));
          $fatal(1, "%s mask SRAM changed too early", test_name);
        end
      end

      wait_cycles = 0;
      while ((compute_done_o !== 1'b1) && (wait_cycles < 40)) begin
        @(posedge clk_i);
        wait_cycles = wait_cycles + 1;

        if (compute_done_o !== 1'b1) begin
          if (busy_o !== 1'b1) begin
            print_fail($sformatf("%s busy_o should stay high until writeback is done", test_name));
            $fatal(1, "%s busy_o should stay high until writeback is done", test_name);
          end
        end
      end

      if (wait_cycles == 40) begin
        print_fail($sformatf("%s timed out waiting for compute_done_o", test_name));
        $fatal(1, "%s timed out waiting for compute_done_o", test_name);
      end

      if (busy_o !== 1'b0) begin
        print_fail($sformatf("%s busy_o should drop when compute_done_o pulses", test_name));
        $fatal(1, "%s busy_o should drop when compute_done_o pulses", test_name);
      end

      if (activation_mem_word(wb_addr) !== expected_data_word) begin
        print_fail($sformatf(
          "%s activation writeback mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_data_word,
          activation_mem_word(wb_addr)
        ));
        $fatal(
          1,
          "%s activation writeback mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_data_word,
          activation_mem_word(wb_addr)
        );
      end

      if (activation_sf_mem_word(wb_addr) !== expected_sf_word) begin
        print_fail($sformatf(
          "%s activation-sf writeback mismatch: expected 0x%02h got 0x%02h",
          test_name,
          expected_sf_word,
          activation_sf_mem_word(wb_addr)
        ));
        $fatal(
          1,
          "%s activation-sf writeback mismatch: expected 0x%02h got 0x%02h",
          test_name,
          expected_sf_word,
          activation_sf_mem_word(wb_addr)
        );
      end

      if (mask_mem_word(wb_addr) !== expected_mask_word) begin
        print_fail($sformatf(
          "%s mask writeback mismatch: expected 0x%02h got 0x%02h",
          test_name,
          expected_mask_word,
          mask_mem_word(wb_addr)
        ));
        $fatal(
          1,
          "%s mask writeback mismatch: expected 0x%02h got 0x%02h",
          test_name,
          expected_mask_word,
          mask_mem_word(wb_addr)
        );
      end

      @(posedge clk_i);

      if (compute_done_o !== 1'b0) begin
        print_fail($sformatf("%s compute_done_o should deassert after one cycle", test_name));
        $fatal(1, "%s compute_done_o should deassert after one cycle", test_name);
      end

      print_pass($sformatf("Test passed: %s", test_name));
    end
  endtask

  task automatic run_fusion_preload(
    input logic [8:0] base_addr,
    input string      test_name
  );
    integer wait_cycles;
    integer slot_idx;
    begin
      print_info($sformatf("Run test: %s", test_name));

      @(negedge clk_i);
      fusion_mode_en_i       = 1'b1;
      activation_addr_i      = base_addr;
      writeback_enable_i     = 1'b0;
      writeback_addr_i       = '0;
      threshold_i            = '0;
      fusion_preload_start_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      fusion_preload_start_i = 1'b0;
      activation_addr_i      = '0;

      wait_cycles = 0;
      while ((busy_o !== 1'b0) && (wait_cycles < 20)) begin
        @(posedge clk_i);
        wait_cycles = wait_cycles + 1;
      end

      if (wait_cycles == 20) begin
        print_fail($sformatf("%s timed out waiting for preload completion", test_name));
        $fatal(1, "%s timed out waiting for preload completion", test_name);
      end

      for (slot_idx = 0; slot_idx < 4; slot_idx++) begin
        if (dut.fusion_buffer_u.slot_activation_q[slot_idx] !== activation_mem_word(base_addr + slot_idx[8:0])) begin
          print_fail($sformatf("%s slot %0d activation mismatch", test_name, slot_idx));
          $fatal(1, "%s slot %0d activation mismatch", test_name, slot_idx);
        end
        if (dut.fusion_buffer_u.slot_activation_sf_q[slot_idx] !== activation_sf_mem_word(base_addr + slot_idx[8:0])) begin
          print_fail($sformatf("%s slot %0d sf mismatch", test_name, slot_idx));
          $fatal(1, "%s slot %0d sf mismatch", test_name, slot_idx);
        end
        if (dut.fusion_buffer_u.slot_mask_q[slot_idx] !== mask_mem_word(base_addr + slot_idx[8:0])) begin
          print_fail($sformatf("%s slot %0d mask mismatch", test_name, slot_idx));
          $fatal(1, "%s slot %0d mask mismatch", test_name, slot_idx);
        end
      end

      fusion_mode_en_i = 1'b0;
      print_pass($sformatf("Test passed: %s", test_name));
    end
  endtask

  task automatic run_fusion_compute_and_check(
    input logic [8:0]  refill_addr,
    input logic [5:0]  wsf_addr,
    input logic [7:0]  packed_flush,
    input logic [31:0] expected_fused_activation,
    input logic [39:0] expected_fused_sf_vec,
    input logic [15:0] expected_fused_rsel,
    input string       test_name
  );
    integer wait_cycles;
    integer slot_idx;
    begin
      print_info($sformatf("Run test: %s", test_name));

      @(negedge clk_i);
      fusion_mode_en_i   = 1'b1;
      activation_addr_i  = refill_addr;
      weight_sf_addr_i   = wsf_addr;
      row_weight_rsel_i  = 16'hA55A;
      fp_flush_flag_i    = packed_flush;
      writeback_enable_i = 1'b0;
      writeback_addr_i   = '0;
      threshold_i        = '0;
      compute_start_i    = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      compute_start_i    = 1'b0;
      activation_addr_i  = '0;
      weight_sf_addr_i   = '0;
      row_weight_rsel_i  = '0;
      fp_flush_flag_i    = '0;

      @(posedge clk_i);

      if (busy_o !== 1'b1) begin
        print_fail($sformatf("%s busy_o should be high during execute cycle", test_name));
        $fatal(1, "%s busy_o should be high during execute cycle", test_name);
      end

      if (dut.hybrid_compute_accumulate !== 1'b1) begin
        print_fail($sformatf("%s compute_accumulate should pulse in execute cycle", test_name));
        $fatal(1, "%s compute_accumulate should pulse in execute cycle", test_name);
      end

      if (dut.hybrid_activation_vec !== expected_fused_activation) begin
        print_fail($sformatf(
          "%s fused activation mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_fused_activation,
          dut.hybrid_activation_vec
        ));
        $fatal(
          1,
          "%s fused activation mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_fused_activation,
          dut.hybrid_activation_vec
        );
      end

      if (dut.hybrid_sf_vec !== expected_fused_sf_vec) begin
        print_fail($sformatf(
          "%s fused sf_vec mismatch: expected 0x%010h got 0x%010h",
          test_name,
          expected_fused_sf_vec,
          dut.hybrid_sf_vec
        ));
        $fatal(
          1,
          "%s fused sf_vec mismatch: expected 0x%010h got 0x%010h",
          test_name,
          expected_fused_sf_vec,
          dut.hybrid_sf_vec
        );
      end

      if (dut.hybrid_row_weight_rsel !== expected_fused_rsel) begin
        print_fail($sformatf(
          "%s fused row-rsel mismatch: expected 0x%04h got 0x%04h",
          test_name,
          expected_fused_rsel,
          dut.hybrid_row_weight_rsel
        ));
        $fatal(
          1,
          "%s fused row-rsel mismatch: expected 0x%04h got 0x%04h",
          test_name,
          expected_fused_rsel,
          dut.hybrid_row_weight_rsel
        );
      end

      wait_cycles = 0;
      while ((compute_done_o !== 1'b1) && (wait_cycles < 40)) begin
        @(posedge clk_i);
        wait_cycles = wait_cycles + 1;
      end

      if (wait_cycles == 40) begin
        print_fail($sformatf("%s timed out waiting for compute_done_o", test_name));
        $fatal(1, "%s timed out waiting for compute_done_o", test_name);
      end

      if (busy_o !== 1'b0) begin
        print_fail($sformatf("%s busy_o should drop when compute_done_o pulses", test_name));
        $fatal(1, "%s busy_o should drop when compute_done_o pulses", test_name);
      end

      for (slot_idx = 0; slot_idx < 4; slot_idx++) begin
        if (dut.fusion_buffer_u.slot_activation_q[slot_idx] !== activation_mem_word((refill_addr - 9'd3) + slot_idx[8:0])) begin
          print_fail($sformatf("%s slot %0d refill activation mismatch", test_name, slot_idx));
          $fatal(1, "%s slot %0d refill activation mismatch", test_name, slot_idx);
        end
        if (dut.fusion_buffer_u.slot_activation_sf_q[slot_idx] !== activation_sf_mem_word((refill_addr - 9'd3) + slot_idx[8:0])) begin
          print_fail($sformatf("%s slot %0d refill sf mismatch", test_name, slot_idx));
          $fatal(1, "%s slot %0d refill sf mismatch", test_name, slot_idx);
        end
        if (dut.fusion_buffer_u.slot_mask_q[slot_idx] !== mask_mem_word((refill_addr - 9'd3) + slot_idx[8:0])) begin
          print_fail($sformatf("%s slot %0d refill mask mismatch", test_name, slot_idx));
          $fatal(1, "%s slot %0d refill mask mismatch", test_name, slot_idx);
        end
      end

      @(posedge clk_i);
      if (compute_done_o !== 1'b0) begin
        print_fail($sformatf("%s compute_done_o should deassert after one cycle", test_name));
        $fatal(1, "%s compute_done_o should deassert after one cycle", test_name);
      end

      fusion_mode_en_i = 1'b0;
      print_pass($sformatf("Test passed: %s", test_name));
    end
  endtask

  task automatic run_fusion_compute_without_preload_and_check(
    input logic [8:0] refill_addr,
    input logic [5:0] wsf_addr,
    input string      test_name
  );
    integer slot_idx;
    begin
      print_info($sformatf("Run test: %s", test_name));

      @(negedge clk_i);
      fusion_mode_en_i   = 1'b1;
      activation_addr_i  = refill_addr;
      weight_sf_addr_i   = wsf_addr;
      row_weight_rsel_i  = 16'hFFFF;
      fp_flush_flag_i    = 8'hFF;
      writeback_enable_i = 1'b0;
      writeback_addr_i   = '0;
      threshold_i        = '0;
      compute_start_i    = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      compute_start_i    = 1'b0;
      fusion_mode_en_i   = 1'b0;
      activation_addr_i  = '0;
      weight_sf_addr_i   = '0;
      row_weight_rsel_i  = '0;
      fp_flush_flag_i    = '0;

      @(posedge clk_i);

      if (busy_o !== 1'b0) begin
        print_fail($sformatf("%s busy_o should remain low when fusion buffer is not ready", test_name));
        $fatal(1, "%s busy_o should remain low when fusion buffer is not ready", test_name);
      end

      if (compute_done_o !== 1'b0) begin
        print_fail($sformatf("%s compute_done_o should remain low when fusion compute is rejected", test_name));
        $fatal(1, "%s compute_done_o should remain low when fusion compute is rejected", test_name);
      end

      if (dut.hybrid_compute_accumulate !== 1'b0) begin
        print_fail($sformatf("%s compute_accumulate should not pulse before preload", test_name));
        $fatal(1, "%s compute_accumulate should not pulse before preload", test_name);
      end

      for (slot_idx = 0; slot_idx < 4; slot_idx++) begin
        if (dut.fusion_buffer_u.slot_activation_q[slot_idx] !== 32'd0) begin
          print_fail($sformatf("%s slot %0d activation should remain empty", test_name, slot_idx));
          $fatal(1, "%s slot %0d activation should remain empty", test_name, slot_idx);
        end
        if (dut.fusion_buffer_u.slot_activation_sf_q[slot_idx] !== 5'd0) begin
          print_fail($sformatf("%s slot %0d sf should remain empty", test_name, slot_idx));
          $fatal(1, "%s slot %0d sf should remain empty", test_name, slot_idx);
        end
        if (dut.fusion_buffer_u.slot_mask_q[slot_idx] !== 8'hFF) begin
          print_fail($sformatf("%s slot %0d mask should remain empty", test_name, slot_idx));
          $fatal(1, "%s slot %0d mask should remain empty", test_name, slot_idx);
        end
      end

      print_pass($sformatf("Test passed: %s", test_name));
    end
  endtask

  initial begin
    reset_inputs();
    init_test_vectors();
    reorder_seq_pattern = '0;
    apply_async_reset();
    check_child_address_contracts();
    print_pass("child address width contract");

    write_reorder_seq(24'h56_34_12);

    @(negedge clk_i);
    weight_load_base_addr_i = WEIGHT_BASE_A;
    weight_load_start_i     = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    weight_load_start_i     = 1'b0;
    weight_load_base_addr_i = '0;

    apply_async_reset();

    if (dut.state_q !== 4'd0) begin
      print_fail($sformatf("async reset should return state_q to idle but got %0d", dut.state_q));
      $fatal(1, "async reset should return state_q to idle but got %0d", dut.state_q);
    end

    check_reorder_seq_retention(24'h56_34_12, "async_reset_keeps_reorder_seq");

    clear_wrapper();

    program_weight_block(WEIGHT_BASE_A, 1'b0);
    program_weight_block(WEIGHT_BASE_B, 1'b1);
    write_weight_sf_buf(WEIGHT_SF_ADDR_A, weight_sf_word_a);
    write_weight_sf_buf(WEIGHT_SF_ADDR_B, weight_sf_word_b);
    write_activation_buf(ACT_ADDR_A, activation_word_a);
    write_activation_buf(ACT_ADDR_B, activation_word_b);
    write_activation_sf_buf(ACT_ADDR_A, activation_sf_a);
    write_activation_sf_buf(ACT_ADDR_B, activation_sf_b);
    for (idx = 0; idx < 8; idx++) begin
      write_activation_buf(FUSION_PRELOAD_BASE + idx[8:0], fusion_activation_word[idx]);
      write_activation_sf_buf(FUSION_PRELOAD_BASE + idx[8:0], fusion_activation_sf_word[idx]);
      write_mask_buf(FUSION_PRELOAD_BASE + idx[8:0], fusion_mask_word[idx]);
    end

    load_ref_block(1'b0);
    check_weight_load(WEIGHT_BASE_A, 1'b0, "weight_load_block_a");

    run_fusion_compute_without_preload_and_check(
      FUSION_PRELOAD_BASE + 9'd4,
      WEIGHT_SF_ADDR_A,
      "fusion_compute_requires_preload"
    );

    expected_sf_vec = build_sf_vec(weight_sf_word_a, activation_sf_a);
    build_expected_fp(activation_word_a, row_rsel_zero, expected_sf_vec);
    for (idx = 0; idx < 8; idx++) begin
      expected_compute_a[idx] = expected_fp_lane[idx];
    end

    run_compute_no_writeback_and_check(
      ACT_ADDR_A,
      WEIGHT_SF_ADDR_A,
      row_rsel_zero,
      8'h00,
      activation_word_a,
      expected_sf_vec,
      WRITEBACK_ADDR_A,
      "compute_without_writeback"
    );

    build_pairwise_bf16_sum(expected_compute_a, expected_compute_a, expected_final_ab);
    build_expected_writeback(
      expected_final_ab,
      THRESHOLD_PARTIAL,
      expected_writeback_word,
      expected_writeback_sf,
      expected_writeback_mask
    );

    run_compute_writeback_and_check(
      ACT_ADDR_A,
      WEIGHT_SF_ADDR_A,
      row_rsel_zero,
      8'hFF,
      THRESHOLD_PARTIAL,
      WRITEBACK_ADDR_A,
      activation_word_a,
      expected_sf_vec,
      expected_writeback_word,
      expected_writeback_sf[4:0],
      expected_writeback_mask,
      4,
      "finalize_and_writeback_block_a"
    );
    check_accumulator_cleared("writeback_clears_state_block_a");

    load_ref_block(1'b1);
    check_weight_load(WEIGHT_BASE_B, 1'b1, "weight_load_block_b");

    expected_sf_vec = build_sf_vec(weight_sf_word_b, activation_sf_b);
    build_expected_fp(activation_word_b, row_rsel_zero, expected_sf_vec);
    for (idx = 0; idx < 8; idx++) begin
      expected_compute_b[idx] = expected_fp_lane[idx];
    end

    build_expected_writeback(
      expected_compute_b,
      THRESHOLD_NONE,
      expected_writeback_word,
      expected_writeback_sf,
      expected_writeback_mask
    );

    run_compute_writeback_and_check(
      ACT_ADDR_B,
      WEIGHT_SF_ADDR_B,
      row_rsel_zero,
      8'h00,
      THRESHOLD_NONE,
      WRITEBACK_ADDR_B,
      activation_word_b,
      expected_sf_vec,
      expected_writeback_word,
      expected_writeback_sf[4:0],
      expected_writeback_mask,
      4,
      "post_clear_writeback_block_b"
    );
    check_accumulator_cleared("writeback_clears_state_block_b");

    run_fusion_preload(FUSION_PRELOAD_BASE, "fusion_preload_fills_slots");
    run_fusion_compute_and_check(
      FUSION_PRELOAD_BASE + 9'd4,
      WEIGHT_SF_ADDR_A,
      8'h00,
      expected_fused_activation_a,
      expected_fused_sf_vec_a,
      expected_fused_rsel_a,
      "fusion_compute_uses_fused_inputs"
    );
    run_fusion_compute_and_check(
      FUSION_PRELOAD_BASE + 9'd5,
      WEIGHT_SF_ADDR_B,
      8'h00,
      expected_fused_activation_b,
      expected_fused_sf_vec_b,
      expected_fused_rsel_b,
      "fusion_compute_shift_refill_progression"
    );

    reorder_seq_pattern = pack_seq8(3'd3, 3'd5, 3'd1, 3'd7, 3'd0, 3'd6, 3'd2, 3'd4);
    program_reorder_compute_window(REORDER_ACT_BASE, REORDER_WSF_BASE);
    write_reorder_seq(reorder_seq_pattern);
    pulse_reorder_ptr_reset("reorder_ptr_reset_before_reordered_flow");

    load_ref_block(1'b0);
    check_weight_load_reordered(
      WEIGHT_BASE_A,
      1'b0,
      reorder_seq_pattern,
      3'd0,
      "reordered_weight_load_batch0"
    );

    run_reordered_compute_no_writeback_and_check(
      REORDER_ACT_BASE,
      REORDER_WSF_BASE,
      reorder_seq_pattern,
      3'd0,
      row_rsel_zero,
      8'h00,
      WRITEBACK_ADDR_A,
      "reordered_compute_0"
    );
    run_reordered_compute_no_writeback_and_check(
      REORDER_ACT_BASE,
      REORDER_WSF_BASE,
      reorder_seq_pattern,
      3'd1,
      row_rsel_zero,
      8'h00,
      WRITEBACK_ADDR_A,
      "reordered_compute_1"
    );
    run_reordered_compute_no_writeback_and_check(
      REORDER_ACT_BASE,
      REORDER_WSF_BASE,
      reorder_seq_pattern,
      3'd2,
      row_rsel_zero,
      8'h00,
      WRITEBACK_ADDR_A,
      "reordered_compute_2"
    );
    run_reordered_compute_no_writeback_and_check(
      REORDER_ACT_BASE,
      REORDER_WSF_BASE,
      reorder_seq_pattern,
      3'd3,
      row_rsel_zero,
      8'h00,
      WRITEBACK_ADDR_A,
      "reordered_compute_3"
    );

    check_weight_load_reordered(
      WEIGHT_BASE_A,
      1'b0,
      reorder_seq_pattern,
      3'd4,
      "reordered_weight_load_batch1"
    );

    run_reordered_compute_no_writeback_and_check(
      REORDER_ACT_BASE,
      REORDER_WSF_BASE,
      reorder_seq_pattern,
      3'd4,
      row_rsel_zero,
      8'h00,
      WRITEBACK_ADDR_A,
      "reordered_compute_4"
    );
    run_reordered_compute_no_writeback_and_check(
      REORDER_ACT_BASE,
      REORDER_WSF_BASE,
      reorder_seq_pattern,
      3'd5,
      row_rsel_zero,
      8'h00,
      WRITEBACK_ADDR_A,
      "reordered_compute_5"
    );
    run_reordered_compute_no_writeback_and_check(
      REORDER_ACT_BASE,
      REORDER_WSF_BASE,
      reorder_seq_pattern,
      3'd6,
      row_rsel_zero,
      8'h00,
      WRITEBACK_ADDR_A,
      "reordered_compute_6"
    );
    run_reordered_compute_no_writeback_and_check(
      REORDER_ACT_BASE,
      REORDER_WSF_BASE,
      reorder_seq_pattern,
      3'd7,
      row_rsel_zero,
      8'h00,
      WRITEBACK_ADDR_A,
      "reordered_compute_7"
    );

    pulse_reorder_ptr_reset("reorder_ptr_reset_after_reordered_flow");

    print_pass("pe_array_buffered_hybrid_top_tb PASS");
    $finish;
  end

endmodule
