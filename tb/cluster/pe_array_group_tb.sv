// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module pe_array_group_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  localparam logic [8:0] WEIGHT_BASE    = 9'd16;
  localparam logic [8:0] ACT_ADDR       = 9'd20;
  localparam logic [5:0] WEIGHT_SF_ADDR = 6'd3;
  localparam logic [8:0] WRITEBACK_ADDR = 9'd200;

  logic        clk_i;
  logic        rst_ni;
  logic        clear_i;

  logic        weight_buf_en_i;
  logic        weight_buf_wen_i;
  logic [10:0] weight_buf_addr_i;
  logic [31:0] weight_buf_wdata_i;
  logic [31:0] weight_buf_rdata_o;

  logic        weight_sf_buf_en_i;
  logic        weight_sf_buf_wen_i;
  logic  [7:0] weight_sf_buf_addr_i;
  logic [39:0] weight_sf_buf_wdata_i;
  logic [39:0] weight_sf_buf_rdata_o;

  logic        activation_buf_en_i;
  logic        activation_buf_wen_i;
  logic [10:0] activation_buf_addr_i;
  logic [31:0] activation_buf_wdata_i;
  logic [31:0] activation_buf_rdata_o;

  logic        activation_sf_buf_en_i;
  logic        activation_sf_buf_wen_i;
  logic [10:0] activation_sf_buf_addr_i;
  logic  [4:0] activation_sf_buf_wdata_i;
  logic  [4:0] activation_sf_buf_rdata_o;

  logic        mask_buf_en_i;
  logic        mask_buf_wen_i;
  logic [10:0] mask_buf_addr_i;
  logic  [7:0] mask_buf_wdata_i;
  logic  [7:0] mask_buf_rdata_o;

  logic        weight_load_start_i;
  logic  [8:0] weight_load_base_addr_i;
  logic        compute_start_i;
  logic        fusion_preload_start_i;
  logic        fusion_mode_en_i;
  logic  [8:0] activation_addr_i;
  logic  [5:0] weight_sf_addr_i;
  logic [15:0] row_weight_rsel_i;
  logic  [7:0] fp_flush_flag_i;
  logic        writeback_enable_i;
  logic  [8:0] writeback_addr_i;
  logic  [7:0] threshold_i;
  logic        use_reorder_seq_i;
  logic        sf_reorder_start_i;
  logic  [2:0] sf_reorder_threshold_i;
  logic  [2:0] sf_reorder_max_iter_i;

  logic        busy_o;
  logic        weight_load_done_o;
  logic        compute_done_o;

  logic [31:0] bank_activation_word [0:3];
  logic [39:0] bank_weight_sf_word  [0:3];
  logic  [4:0] bank_activation_sf   [0:3];
  logic [31:0] bank_writeback_word  [0:3];
  logic [31:0] expected_writeback_word [0:3];
  logic  [4:0] expected_writeback_sf   [0:3];
  logic  [7:0] expected_writeback_mask [0:3];
  logic        expected_writeback_word_valid [0:3];
  logic        expected_writeback_sf_valid   [0:3];
  logic        expected_writeback_mask_valid [0:3];
  logic [39:0] bank_reorder_weight_sf_word [0:3][0:7];
  logic signed [4:0] bank_reorder_activation_sf [0:3][0:7];
  logic [23:0] expected_reorder_seq [0:3];
  integer      idx;
  logic        ref_reorder_start_i;
  logic [39:0] ref_reorder_weight_sf_vec_i;
  logic signed [4:0] ref_reorder_activation_sf_i;
  logic        ref_reorder_done_o;
  logic [23:0] ref_reorder_seq_o;

  pe_array_group dut (
    .clk_i                    (clk_i),
    .rst_ni                   (rst_ni),
    .clear_i                  (clear_i),
    .weight_buf_en_i          (weight_buf_en_i),
    .weight_buf_wen_i         (weight_buf_wen_i),
    .weight_buf_addr_i        (weight_buf_addr_i),
    .weight_buf_wdata_i       (weight_buf_wdata_i),
    .weight_buf_rdata_o       (weight_buf_rdata_o),
    .weight_sf_buf_en_i       (weight_sf_buf_en_i),
    .weight_sf_buf_wen_i      (weight_sf_buf_wen_i),
    .weight_sf_buf_addr_i     (weight_sf_buf_addr_i),
    .weight_sf_buf_wdata_i    (weight_sf_buf_wdata_i),
    .weight_sf_buf_rdata_o    (weight_sf_buf_rdata_o),
    .activation_buf_en_i      (activation_buf_en_i),
    .activation_buf_wen_i     (activation_buf_wen_i),
    .activation_buf_addr_i    (activation_buf_addr_i),
    .activation_buf_wdata_i   (activation_buf_wdata_i),
    .activation_buf_rdata_o   (activation_buf_rdata_o),
    .activation_sf_buf_en_i   (activation_sf_buf_en_i),
    .activation_sf_buf_wen_i  (activation_sf_buf_wen_i),
    .activation_sf_buf_addr_i (activation_sf_buf_addr_i),
    .activation_sf_buf_wdata_i(activation_sf_buf_wdata_i),
    .activation_sf_buf_rdata_o(activation_sf_buf_rdata_o),
    .mask_buf_en_i            (mask_buf_en_i),
    .mask_buf_wen_i           (mask_buf_wen_i),
    .mask_buf_addr_i          (mask_buf_addr_i),
    .mask_buf_wdata_i         (mask_buf_wdata_i),
    .mask_buf_rdata_o         (mask_buf_rdata_o),
    .weight_load_start_i      (weight_load_start_i),
    .weight_load_base_addr_i  (weight_load_base_addr_i),
    .compute_start_i          (compute_start_i),
    .fusion_preload_start_i   (fusion_preload_start_i),
    .fusion_mode_en_i         (fusion_mode_en_i),
    .activation_addr_i        (activation_addr_i),
    .weight_sf_addr_i         (weight_sf_addr_i),
    .row_weight_rsel_i        (row_weight_rsel_i),
    .fp_flush_flag_i          (fp_flush_flag_i),
    .writeback_enable_i       (writeback_enable_i),
    .writeback_addr_i         (writeback_addr_i),
    .threshold_i              (threshold_i),
    .use_reorder_seq_i        (use_reorder_seq_i),
    .sf_reorder_start_i       (sf_reorder_start_i),
    .sf_reorder_threshold_i   (sf_reorder_threshold_i),
    .sf_reorder_max_iter_i    (sf_reorder_max_iter_i),
    .busy_o                   (busy_o),
    .weight_load_done_o       (weight_load_done_o),
    .compute_done_o           (compute_done_o)
  );

  sf_reorder_top ref_reorder_u (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .start_i        (ref_reorder_start_i),
    .weight_sf_vec_i(ref_reorder_weight_sf_vec_i),
    .activation_sf_i(ref_reorder_activation_sf_i),
    .threshold_i    (sf_reorder_threshold_i),
    .max_iter_i     (sf_reorder_max_iter_i),
    .done_o         (ref_reorder_done_o),
    .seq_o          (ref_reorder_seq_o)
  );

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

  task automatic check_group_address_contracts;
    begin
      if ($bits(dut.weight_buf_addr_i) != 11) begin
        $fatal(1, "weight_buf_addr_i width mismatch: expected 11, got %0d", $bits(dut.weight_buf_addr_i));
      end
      if ($bits(dut.weight_sf_buf_addr_i) != 8) begin
        $fatal(1, "weight_sf_buf_addr_i width mismatch: expected 8, got %0d", $bits(dut.weight_sf_buf_addr_i));
      end
      if ($bits(dut.activation_addr_i) != 9) begin
        $fatal(1, "activation_addr_i width mismatch: expected 9, got %0d", $bits(dut.activation_addr_i));
      end
      if ($bits(dut.weight_sf_addr_i) != 6) begin
        $fatal(1, "weight_sf_addr_i width mismatch: expected 6, got %0d", $bits(dut.weight_sf_addr_i));
      end
      if ($bits(dut.gen_banks[0].array_u.weight_buf_addr_i) != 9) begin
        $fatal(1, "child weight_buf_addr_i width mismatch: expected 9, got %0d", $bits(dut.gen_banks[0].array_u.weight_buf_addr_i));
      end
      if ($bits(dut.gen_banks[0].array_u.weight_sf_buf_addr_i) != 6) begin
        $fatal(1, "child weight_sf_buf_addr_i width mismatch: expected 6, got %0d", $bits(dut.gen_banks[0].array_u.weight_sf_buf_addr_i));
      end
    end
  endtask

  always @(posedge clk_i) begin
    if (clear_i) begin
      expected_writeback_word[0]       <= '0;
      expected_writeback_word[1]       <= '0;
      expected_writeback_word[2]       <= '0;
      expected_writeback_word[3]       <= '0;
      expected_writeback_sf[0]         <= '0;
      expected_writeback_sf[1]         <= '0;
      expected_writeback_sf[2]         <= '0;
      expected_writeback_sf[3]         <= '0;
      expected_writeback_mask[0]       <= '0;
      expected_writeback_mask[1]       <= '0;
      expected_writeback_mask[2]       <= '0;
      expected_writeback_mask[3]       <= '0;
      expected_writeback_word_valid[0] <= 1'b0;
      expected_writeback_word_valid[1] <= 1'b0;
      expected_writeback_word_valid[2] <= 1'b0;
      expected_writeback_word_valid[3] <= 1'b0;
      expected_writeback_sf_valid[0]   <= 1'b0;
      expected_writeback_sf_valid[1]   <= 1'b0;
      expected_writeback_sf_valid[2]   <= 1'b0;
      expected_writeback_sf_valid[3]   <= 1'b0;
      expected_writeback_mask_valid[0] <= 1'b0;
      expected_writeback_mask_valid[1] <= 1'b0;
      expected_writeback_mask_valid[2] <= 1'b0;
      expected_writeback_mask_valid[3] <= 1'b0;
    end else begin
      if (
        dut.gen_banks[0].array_u.activation_data_en &&
        dut.gen_banks[0].array_u.activation_data_write &&
        (dut.gen_banks[0].array_u.activation_data_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_word[0]       <= dut.gen_banks[0].array_u.activation_data_wdata;
        expected_writeback_word_valid[0] <= 1'b1;
      end
      if (
        dut.gen_banks[1].array_u.activation_data_en &&
        dut.gen_banks[1].array_u.activation_data_write &&
        (dut.gen_banks[1].array_u.activation_data_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_word[1]       <= dut.gen_banks[1].array_u.activation_data_wdata;
        expected_writeback_word_valid[1] <= 1'b1;
      end
      if (
        dut.gen_banks[2].array_u.activation_data_en &&
        dut.gen_banks[2].array_u.activation_data_write &&
        (dut.gen_banks[2].array_u.activation_data_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_word[2]       <= dut.gen_banks[2].array_u.activation_data_wdata;
        expected_writeback_word_valid[2] <= 1'b1;
      end
      if (
        dut.gen_banks[3].array_u.activation_data_en &&
        dut.gen_banks[3].array_u.activation_data_write &&
        (dut.gen_banks[3].array_u.activation_data_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_word[3]       <= dut.gen_banks[3].array_u.activation_data_wdata;
        expected_writeback_word_valid[3] <= 1'b1;
      end

      if (
        dut.gen_banks[0].array_u.activation_sf_mem_en &&
        dut.gen_banks[0].array_u.activation_sf_mem_write &&
        (dut.gen_banks[0].array_u.activation_sf_mem_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_sf[0]       <= dut.gen_banks[0].array_u.activation_sf_mem_wdata;
        expected_writeback_sf_valid[0] <= 1'b1;
      end
      if (
        dut.gen_banks[1].array_u.activation_sf_mem_en &&
        dut.gen_banks[1].array_u.activation_sf_mem_write &&
        (dut.gen_banks[1].array_u.activation_sf_mem_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_sf[1]       <= dut.gen_banks[1].array_u.activation_sf_mem_wdata;
        expected_writeback_sf_valid[1] <= 1'b1;
      end
      if (
        dut.gen_banks[2].array_u.activation_sf_mem_en &&
        dut.gen_banks[2].array_u.activation_sf_mem_write &&
        (dut.gen_banks[2].array_u.activation_sf_mem_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_sf[2]       <= dut.gen_banks[2].array_u.activation_sf_mem_wdata;
        expected_writeback_sf_valid[2] <= 1'b1;
      end
      if (
        dut.gen_banks[3].array_u.activation_sf_mem_en &&
        dut.gen_banks[3].array_u.activation_sf_mem_write &&
        (dut.gen_banks[3].array_u.activation_sf_mem_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_sf[3]       <= dut.gen_banks[3].array_u.activation_sf_mem_wdata;
        expected_writeback_sf_valid[3] <= 1'b1;
      end

      if (
        dut.gen_banks[0].array_u.mask_mem_en &&
        dut.gen_banks[0].array_u.mask_mem_write &&
        (dut.gen_banks[0].array_u.mask_mem_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_mask[0]       <= dut.gen_banks[0].array_u.mask_mem_wdata;
        expected_writeback_mask_valid[0] <= 1'b1;
      end
      if (
        dut.gen_banks[1].array_u.mask_mem_en &&
        dut.gen_banks[1].array_u.mask_mem_write &&
        (dut.gen_banks[1].array_u.mask_mem_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_mask[1]       <= dut.gen_banks[1].array_u.mask_mem_wdata;
        expected_writeback_mask_valid[1] <= 1'b1;
      end
      if (
        dut.gen_banks[2].array_u.mask_mem_en &&
        dut.gen_banks[2].array_u.mask_mem_write &&
        (dut.gen_banks[2].array_u.mask_mem_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_mask[2]       <= dut.gen_banks[2].array_u.mask_mem_wdata;
        expected_writeback_mask_valid[2] <= 1'b1;
      end
      if (
        dut.gen_banks[3].array_u.mask_mem_en &&
        dut.gen_banks[3].array_u.mask_mem_write &&
        (dut.gen_banks[3].array_u.mask_mem_addr == WRITEBACK_ADDR)
      ) begin
        expected_writeback_mask[3]       <= dut.gen_banks[3].array_u.mask_mem_wdata;
        expected_writeback_mask_valid[3] <= 1'b1;
      end
    end
  end

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

  task automatic check_bit(
    input string test_name,
    input logic  expected_value,
    input logic  actual_value
  );
    begin
      if (actual_value !== expected_value) begin
        print_fail($sformatf(
          "%s mismatch: expected %0b got %0b",
          test_name,
          expected_value,
          actual_value
        ));
        $fatal(1, "%s mismatch: expected %0b got %0b", test_name, expected_value, actual_value);
      end
    end
  endtask

  task automatic check_nibble(
    input string      test_name,
    input logic [3:0] expected_value,
    input logic [3:0] actual_value
  );
    begin
      if (actual_value !== expected_value) begin
        print_fail($sformatf(
          "%s mismatch: expected 0x%01h got 0x%01h",
          test_name,
          expected_value,
          actual_value
        ));
        $fatal(1, "%s mismatch: expected 0x%01h got 0x%01h", test_name, expected_value, actual_value);
      end
    end
  endtask

  task automatic check_5bit(
    input string      test_name,
    input logic [4:0] expected_value,
    input logic [4:0] actual_value
  );
    begin
      if (actual_value !== expected_value) begin
        print_fail($sformatf(
          "%s mismatch: expected 0x%02h got 0x%02h",
          test_name,
          expected_value,
          actual_value
        ));
        $fatal(1, "%s mismatch: expected 0x%02h got 0x%02h", test_name, expected_value, actual_value);
      end
    end
  endtask

  task automatic check_8bit(
    input string      test_name,
    input logic [7:0] expected_value,
    input logic [7:0] actual_value
  );
    begin
      if (actual_value !== expected_value) begin
        print_fail($sformatf(
          "%s mismatch: expected 0x%02h got 0x%02h",
          test_name,
          expected_value,
          actual_value
        ));
        $fatal(1, "%s mismatch: expected 0x%02h got 0x%02h", test_name, expected_value, actual_value);
      end
    end
  endtask

  task automatic check_32bit(
    input string       test_name,
    input logic [31:0] expected_value,
    input logic [31:0] actual_value
  );
    begin
      if (actual_value !== expected_value) begin
        print_fail($sformatf(
          "%s mismatch: expected 0x%08h got 0x%08h",
          test_name,
          expected_value,
          actual_value
        ));
        $fatal(1, "%s mismatch: expected 0x%08h got 0x%08h", test_name, expected_value, actual_value);
      end
    end
  endtask

  task automatic check_40bit(
    input string       test_name,
    input logic [39:0] expected_value,
    input logic [39:0] actual_value
  );
    begin
      if (actual_value !== expected_value) begin
        print_fail($sformatf(
          "%s mismatch: expected 0x%010h got 0x%010h",
          test_name,
          expected_value,
          actual_value
        ));
        $fatal(1, "%s mismatch: expected 0x%010h got 0x%010h", test_name, expected_value, actual_value);
      end
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

  function automatic logic [15:0] pack_codes8(
    input logic [1:0] lane0,
    input logic [1:0] lane1,
    input logic [1:0] lane2,
    input logic [1:0] lane3,
    input logic [1:0] lane4,
    input logic [1:0] lane5,
    input logic [1:0] lane6,
    input logic [1:0] lane7
  );
    pack_codes8 = {
      lane7, lane6, lane5, lane4,
      lane3, lane2, lane1, lane0
    };
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

  function automatic logic [39:0] pack_codes_to_weights(
    input logic [15:0]       packed_codes,
    input logic signed [4:0] bias_sf,
    input logic signed [4:0] activation_sf
  );
    logic signed [5:0] norm_sf_ext;
    logic signed [5:0] weight_sf_ext;
    logic        [39:0] packed_sf;
    integer             lane_idx;
    begin
      packed_sf = '0;
      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        norm_sf_ext   = $signed({1'b0, 2'b00, packed_codes[lane_idx * 2 +: 2], 1'b0});
        weight_sf_ext = norm_sf_ext
                      - $signed({activation_sf[4], activation_sf})
                      + $signed({bias_sf[4], bias_sf});
        packed_sf[lane_idx * 5 +: 5] = weight_sf_ext[4:0];
      end
      pack_codes_to_weights = packed_sf;
    end
  endfunction

  function automatic logic [31:0] diagonal_weight_word(
    input logic signed [3:0] diag_value,
    input int                lane_idx
  );
    begin
      diagonal_weight_word = '0;
      diagonal_weight_word[lane_idx * 4 +: 4] = diag_value[3:0];
    end
  endfunction

  task automatic reset_inputs;
    begin
      rst_ni                    = 1'b1;
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
      compute_start_i           = 1'b0;
      fusion_preload_start_i    = 1'b0;
      fusion_mode_en_i          = 1'b0;
      activation_addr_i         = '0;
      weight_sf_addr_i          = '0;
      row_weight_rsel_i         = '0;
      fp_flush_flag_i           = '0;
      writeback_enable_i        = 1'b0;
      writeback_addr_i          = '0;
      threshold_i               = '0;
      use_reorder_seq_i         = 1'b0;
      sf_reorder_start_i        = 1'b0;
      sf_reorder_threshold_i    = '0;
      sf_reorder_max_iter_i     = '0;

      ref_reorder_start_i       = 1'b0;
      ref_reorder_weight_sf_vec_i = '0;
      ref_reorder_activation_sf_i = '0;
    end
  endtask

  task automatic init_test_vectors;
    begin
      bank_activation_word[0] = pack_int4x8(4'sd1, 4'sd2, 4'sd3, 4'sd4, 4'sd1, 4'sd2, 4'sd3, 4'sd4);
      bank_activation_word[1] = pack_int4x8(-4'sd1, 4'sd1, -4'sd2, 4'sd2, -4'sd3, 4'sd3, -4'sd4, 4'sd4);
      bank_activation_word[2] = pack_int4x8(4'sd4, 4'sd0, -4'sd4, 4'sd0, 4'sd3, 4'sd0, -4'sd3, 4'sd0);
      bank_activation_word[3] = pack_int4x8(4'sd7, -4'sd7, 4'sd6, -4'sd6, 4'sd5, -4'sd5, 4'sd4, -4'sd4);

      bank_weight_sf_word[0]  = pack_sf5x8(5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0);
      bank_weight_sf_word[1]  = pack_sf5x8(5'sd1, 5'sd1, 5'sd1, 5'sd1, 5'sd1, 5'sd1, 5'sd1, 5'sd1);
      bank_weight_sf_word[2]  = pack_sf5x8(-5'sd1, -5'sd1, -5'sd1, -5'sd1, -5'sd1, -5'sd1, -5'sd1, -5'sd1);
      bank_weight_sf_word[3]  = pack_sf5x8(5'sd2, 5'sd1, 5'sd0, -5'sd1, 5'sd2, 5'sd1, 5'sd0, -5'sd1);

      bank_activation_sf[0]   = 5'sd0;
      bank_activation_sf[1]   = 5'sd1;
      bank_activation_sf[2]   = -5'sd1;
      bank_activation_sf[3]   = 5'sd2;

      for (idx = 0; idx < 8; idx++) begin
        bank_reorder_activation_sf[0][idx] = 5'sd0;
        bank_reorder_activation_sf[1][idx] = 5'sd0;
        bank_reorder_activation_sf[2][idx] = 5'sd1;
        bank_reorder_activation_sf[3][idx] = -5'sd1;
      end

      bank_reorder_weight_sf_word[0][0] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[0][0]);
      bank_reorder_weight_sf_word[0][1] = pack_codes_to_weights(pack_codes8(2'd1, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[0][1]);
      bank_reorder_weight_sf_word[0][2] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[0][2]);
      bank_reorder_weight_sf_word[0][3] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[0][3]);
      bank_reorder_weight_sf_word[0][4] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[0][4]);
      bank_reorder_weight_sf_word[0][5] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[0][5]);
      bank_reorder_weight_sf_word[0][6] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd0), 5'sd1, bank_reorder_activation_sf[0][6]);
      bank_reorder_weight_sf_word[0][7] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1), 5'sd1, bank_reorder_activation_sf[0][7]);

      bank_reorder_weight_sf_word[1][0] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[1][0]);
      bank_reorder_weight_sf_word[1][1] = pack_codes_to_weights(pack_codes8(2'd1, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[1][1]);
      bank_reorder_weight_sf_word[1][2] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[1][2]);
      bank_reorder_weight_sf_word[1][3] = pack_codes_to_weights(pack_codes8(2'd2, 2'd2, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[1][3]);
      bank_reorder_weight_sf_word[1][4] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[1][4]);
      bank_reorder_weight_sf_word[1][5] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[1][5]);
      bank_reorder_weight_sf_word[1][6] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd0), 5'sd1, bank_reorder_activation_sf[1][6]);
      bank_reorder_weight_sf_word[1][7] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1), 5'sd1, bank_reorder_activation_sf[1][7]);

      bank_reorder_weight_sf_word[2][0] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[2][0]);
      bank_reorder_weight_sf_word[2][1] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd1), 5'sd1, bank_reorder_activation_sf[2][1]);
      bank_reorder_weight_sf_word[2][2] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd1, 2'd1), 5'sd1, bank_reorder_activation_sf[2][2]);
      bank_reorder_weight_sf_word[2][3] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[2][3]);
      bank_reorder_weight_sf_word[2][4] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd1, 2'd1, 2'd1), 5'sd1, bank_reorder_activation_sf[2][4]);
      bank_reorder_weight_sf_word[2][5] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd1, 2'd1, 2'd1, 2'd1), 5'sd1, bank_reorder_activation_sf[2][5]);
      bank_reorder_weight_sf_word[2][6] = pack_codes_to_weights(pack_codes8(2'd0, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1), 5'sd1, bank_reorder_activation_sf[2][6]);
      bank_reorder_weight_sf_word[2][7] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1), 5'sd1, bank_reorder_activation_sf[2][7]);

      bank_reorder_weight_sf_word[3][0] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[3][0]);
      bank_reorder_weight_sf_word[3][1] = pack_codes_to_weights(pack_codes8(2'd1, 2'd0, 2'd1, 2'd0, 2'd1, 2'd0, 2'd1, 2'd0), 5'sd1, bank_reorder_activation_sf[3][1]);
      bank_reorder_weight_sf_word[3][2] = pack_codes_to_weights(pack_codes8(2'd2, 2'd0, 2'd2, 2'd0, 2'd2, 2'd0, 2'd2, 2'd0), 5'sd1, bank_reorder_activation_sf[3][2]);
      bank_reorder_weight_sf_word[3][3] = pack_codes_to_weights(pack_codes8(2'd0, 2'd1, 2'd0, 2'd1, 2'd0, 2'd1, 2'd0, 2'd1), 5'sd1, bank_reorder_activation_sf[3][3]);
      bank_reorder_weight_sf_word[3][4] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[3][4]);
      bank_reorder_weight_sf_word[3][5] = pack_codes_to_weights(pack_codes8(2'd0, 2'd0, 2'd0, 2'd0, 2'd1, 2'd1, 2'd1, 2'd1), 5'sd1, bank_reorder_activation_sf[3][5]);
      bank_reorder_weight_sf_word[3][6] = pack_codes_to_weights(pack_codes8(2'd2, 2'd2, 2'd2, 2'd2, 2'd0, 2'd0, 2'd0, 2'd0), 5'sd1, bank_reorder_activation_sf[3][6]);
      bank_reorder_weight_sf_word[3][7] = pack_codes_to_weights(pack_codes8(2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1, 2'd1), 5'sd1, bank_reorder_activation_sf[3][7]);
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

  task automatic clear_group;
    begin
      print_info("apply synchronous clear");
      @(negedge clk_i);
      clear_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      clear_i = 1'b0;
    end
  endtask

  task automatic reset_writeback_capture;
    integer bank_idx;
    begin
      for (bank_idx = 0; bank_idx < 4; bank_idx++) begin
        expected_writeback_word[bank_idx]       = '0;
        expected_writeback_sf[bank_idx]         = '0;
        expected_writeback_mask[bank_idx]       = '0;
        expected_writeback_word_valid[bank_idx] = 1'b0;
        expected_writeback_sf_valid[bank_idx]   = 1'b0;
        expected_writeback_mask_valid[bank_idx] = 1'b0;
      end
    end
  endtask

  task automatic weight_buf_write(
    input logic [1:0] bank_sel,
    input logic [8:0] local_addr,
    input logic [31:0] data_word
  );
    begin
      @(negedge clk_i);
      weight_buf_en_i    = 1'b1;
      weight_buf_wen_i   = 1'b1;
      weight_buf_addr_i  = {bank_sel, local_addr};
      weight_buf_wdata_i = data_word;
      @(posedge clk_i);
      @(negedge clk_i);
      weight_buf_en_i    = 1'b0;
      weight_buf_wen_i   = 1'b0;
      weight_buf_addr_i  = '0;
      weight_buf_wdata_i = '0;
    end
  endtask

  task automatic weight_buf_read_expect(
    input logic [1:0] bank_sel,
    input logic [8:0] local_addr,
    input logic [31:0] expected_word
  );
    begin
      @(negedge clk_i);
      weight_buf_en_i   = 1'b1;
      weight_buf_wen_i  = 1'b0;
      weight_buf_addr_i = {bank_sel, local_addr};
      @(posedge clk_i);
      @(negedge clk_i);
      weight_buf_en_i   = 1'b0;
      weight_buf_addr_i = '0;
      @(posedge clk_i);
      if (weight_buf_rdata_o !== expected_word) begin
        print_fail($sformatf(
          "weight bank %0d addr %0d expected 0x%08h got 0x%08h",
          bank_sel,
          local_addr,
          expected_word,
          weight_buf_rdata_o
        ));
        $fatal(
          1,
          "weight bank %0d addr %0d expected 0x%08h got 0x%08h",
          bank_sel,
          local_addr,
          expected_word,
          weight_buf_rdata_o
        );
      end
    end
  endtask

  task automatic weight_sf_buf_write(
    input logic [1:0] bank_sel,
    input logic [5:0] local_addr,
    input logic [39:0] data_word
  );
    begin
      @(negedge clk_i);
      weight_sf_buf_en_i    = 1'b1;
      weight_sf_buf_wen_i   = 1'b1;
      weight_sf_buf_addr_i  = {bank_sel, local_addr};
      weight_sf_buf_wdata_i = data_word;
      @(posedge clk_i);
      @(negedge clk_i);
      weight_sf_buf_en_i    = 1'b0;
      weight_sf_buf_wen_i   = 1'b0;
      weight_sf_buf_addr_i  = '0;
      weight_sf_buf_wdata_i = '0;
    end
  endtask

  task automatic weight_sf_buf_read_expect(
    input logic [1:0] bank_sel,
    input logic [5:0] local_addr,
    input logic [39:0] expected_word
  );
    begin
      @(negedge clk_i);
      weight_sf_buf_en_i   = 1'b1;
      weight_sf_buf_wen_i  = 1'b0;
      weight_sf_buf_addr_i = {bank_sel, local_addr};
      @(posedge clk_i);
      @(negedge clk_i);
      weight_sf_buf_en_i   = 1'b0;
      weight_sf_buf_addr_i = '0;
      @(posedge clk_i);
      if (weight_sf_buf_rdata_o !== expected_word) begin
        print_fail($sformatf(
          "weight_sf bank %0d addr %0d mismatch: expected 0x%010h got 0x%010h",
          bank_sel,
          local_addr,
          expected_word,
          weight_sf_buf_rdata_o
        ));
        $fatal(
          1,
          "weight_sf bank %0d addr %0d mismatch: expected 0x%010h got 0x%010h",
          bank_sel,
          local_addr,
          expected_word,
          weight_sf_buf_rdata_o
        );
      end
    end
  endtask

  task automatic activation_buf_write(
    input logic [1:0] bank_sel,
    input logic [8:0] local_addr,
    input logic [31:0] data_word
  );
    begin
      @(negedge clk_i);
      activation_buf_en_i    = 1'b1;
      activation_buf_wen_i   = 1'b1;
      activation_buf_addr_i  = {bank_sel, local_addr};
      activation_buf_wdata_i = data_word;
      @(posedge clk_i);
      @(negedge clk_i);
      activation_buf_en_i    = 1'b0;
      activation_buf_wen_i   = 1'b0;
      activation_buf_addr_i  = '0;
      activation_buf_wdata_i = '0;
    end
  endtask

  task automatic activation_buf_read_expect(
    input logic [1:0] bank_sel,
    input logic [8:0] local_addr,
    input logic [31:0] expected_word
  );
    begin
      @(negedge clk_i);
      activation_buf_en_i   = 1'b1;
      activation_buf_wen_i  = 1'b0;
      activation_buf_addr_i = {bank_sel, local_addr};
      @(posedge clk_i);
      @(negedge clk_i);
      activation_buf_en_i   = 1'b0;
      activation_buf_addr_i = '0;
      @(posedge clk_i);
      if (activation_buf_rdata_o !== expected_word) begin
        print_fail($sformatf(
          "activation bank %0d addr %0d expected 0x%08h got 0x%08h",
          bank_sel,
          local_addr,
          expected_word,
          activation_buf_rdata_o
        ));
        $fatal(
          1,
          "activation bank %0d addr %0d expected 0x%08h got 0x%08h",
          bank_sel,
          local_addr,
          expected_word,
          activation_buf_rdata_o
        );
      end
    end
  endtask

  task automatic activation_buf_read_capture(
    input  logic [1:0] bank_sel,
    input  logic [8:0] local_addr,
    output logic [31:0] data_word
  );
    begin
      @(negedge clk_i);
      activation_buf_en_i   = 1'b1;
      activation_buf_wen_i  = 1'b0;
      activation_buf_addr_i = {bank_sel, local_addr};
      @(posedge clk_i);
      @(negedge clk_i);
      activation_buf_en_i   = 1'b0;
      activation_buf_addr_i = '0;
      @(posedge clk_i);
      data_word = activation_buf_rdata_o;
    end
  endtask

  task automatic activation_sf_buf_write(
    input logic [1:0] bank_sel,
    input logic [8:0] local_addr,
    input logic  [4:0] data_word
  );
    begin
      @(negedge clk_i);
      activation_sf_buf_en_i    = 1'b1;
      activation_sf_buf_wen_i   = 1'b1;
      activation_sf_buf_addr_i  = {bank_sel, local_addr};
      activation_sf_buf_wdata_i = data_word;
      @(posedge clk_i);
      @(negedge clk_i);
      activation_sf_buf_en_i    = 1'b0;
      activation_sf_buf_wen_i   = 1'b0;
      activation_sf_buf_addr_i  = '0;
      activation_sf_buf_wdata_i = '0;
    end
  endtask

  task automatic activation_sf_buf_read_expect(
    input logic [1:0] bank_sel,
    input logic [8:0] local_addr,
    input logic  [4:0] expected_word
  );
    begin
      @(negedge clk_i);
      activation_sf_buf_en_i   = 1'b1;
      activation_sf_buf_wen_i  = 1'b0;
      activation_sf_buf_addr_i = {bank_sel, local_addr};
      @(posedge clk_i);
      @(negedge clk_i);
      activation_sf_buf_en_i   = 1'b0;
      activation_sf_buf_addr_i = '0;
      @(posedge clk_i);
      if (activation_sf_buf_rdata_o !== expected_word) begin
        print_fail($sformatf(
          "activation_sf bank %0d addr %0d mismatch: expected 0x%02h got 0x%02h",
          bank_sel,
          local_addr,
          expected_word,
          activation_sf_buf_rdata_o
        ));
        $fatal(
          1,
          "activation_sf bank %0d addr %0d mismatch: expected 0x%02h got 0x%02h",
          bank_sel,
          local_addr,
          expected_word,
          activation_sf_buf_rdata_o
        );
      end
    end
  endtask

  task automatic mask_buf_write(
    input logic [1:0] bank_sel,
    input logic [8:0] local_addr,
    input logic  [7:0] data_word
  );
    begin
      @(negedge clk_i);
      mask_buf_en_i    = 1'b1;
      mask_buf_wen_i   = 1'b1;
      mask_buf_addr_i  = {bank_sel, local_addr};
      mask_buf_wdata_i = data_word;
      @(posedge clk_i);
      @(negedge clk_i);
      mask_buf_en_i    = 1'b0;
      mask_buf_wen_i   = 1'b0;
      mask_buf_addr_i  = '0;
      mask_buf_wdata_i = '0;
    end
  endtask

  task automatic mask_buf_read_expect(
    input logic [1:0] bank_sel,
    input logic [8:0] local_addr,
    input logic  [7:0] expected_word
  );
    begin
      @(negedge clk_i);
      mask_buf_en_i   = 1'b1;
      mask_buf_wen_i  = 1'b0;
      mask_buf_addr_i = {bank_sel, local_addr};
      @(posedge clk_i);
      @(negedge clk_i);
      mask_buf_en_i   = 1'b0;
      mask_buf_addr_i = '0;
      @(posedge clk_i);
      if (mask_buf_rdata_o !== expected_word) begin
        print_fail($sformatf(
          "mask bank %0d addr %0d mismatch: expected 0x%02h got 0x%02h",
          bank_sel,
          local_addr,
          expected_word,
          mask_buf_rdata_o
        ));
        $fatal(
          1,
          "mask bank %0d addr %0d mismatch: expected 0x%02h got 0x%02h",
          bank_sel,
          local_addr,
          expected_word,
          mask_buf_rdata_o
        );
      end
    end
  endtask

  task automatic wait_for_weight_load_done;
    integer cycle_count;
    begin
      cycle_count = 0;
      while (weight_load_done_o !== 1'b1) begin
        @(posedge clk_i);
        cycle_count++;
        if (cycle_count > 80) begin
          print_fail($sformatf(
            "weight_load_done_o timeout: expected %0b within 80 cycles got %0b after %0d cycles",
            1'b1,
            weight_load_done_o,
            cycle_count
          ));
          $fatal(
            1,
            "weight_load_done_o timeout: expected %0b within 80 cycles got %0b after %0d cycles",
            1'b1,
            weight_load_done_o,
            cycle_count
          );
        end
      end
    end
  endtask

  task automatic wait_for_compute_done;
    integer cycle_count;
    begin
      cycle_count = 0;
      while (compute_done_o !== 1'b1) begin
        @(posedge clk_i);
        cycle_count++;
        if (cycle_count > 200) begin
          print_fail($sformatf(
            "compute_done_o timeout: expected %0b within 200 cycles got %0b after %0d cycles",
            1'b1,
            compute_done_o,
            cycle_count
          ));
          $fatal(
            1,
            "compute_done_o timeout: expected %0b within 200 cycles got %0b after %0d cycles",
            1'b1,
            compute_done_o,
            cycle_count
          );
        end
      end
    end
  endtask

  task automatic wait_for_not_busy(
    input integer max_cycles,
    input string  test_name
  );
    integer cycle_count;
    begin
      cycle_count = 0;
      while (busy_o !== 1'b0) begin
        @(posedge clk_i);
        cycle_count++;
        if (cycle_count > max_cycles) begin
          print_fail($sformatf(
            "%s timeout: busy_o stayed high for more than %0d cycles",
            test_name,
            max_cycles
          ));
          $fatal(1, "%s timeout: busy_o stayed high for more than %0d cycles", test_name, max_cycles);
        end
      end
    end
  endtask

  task automatic program_bank_reorder_window(input logic [1:0] bank_sel);
    integer sample_idx;
    begin
      for (sample_idx = 0; sample_idx < 8; sample_idx++) begin
        weight_sf_buf_write(
          bank_sel,
          WEIGHT_SF_ADDR + sample_idx[5:0],
          bank_reorder_weight_sf_word[bank_sel][sample_idx]
        );
        activation_sf_buf_write(
          bank_sel,
          ACT_ADDR + sample_idx[8:0],
          bank_reorder_activation_sf[bank_sel][sample_idx]
        );
      end
    end
  endtask

  task automatic run_reference_reorder_for_bank(input integer bank_sel);
    integer sample_idx;
    begin
      ref_reorder_start_i = 1'b0;
      for (sample_idx = 0; sample_idx < 8; sample_idx++) begin
        ref_reorder_weight_sf_vec_i = bank_reorder_weight_sf_word[bank_sel][sample_idx];
        ref_reorder_activation_sf_i = bank_reorder_activation_sf[bank_sel][sample_idx];
        ref_reorder_start_i         = (sample_idx == 0);
        @(posedge clk_i);
      end

      ref_reorder_start_i = 1'b0;
      for (sample_idx = 0; sample_idx < 8; sample_idx++) begin
        ref_reorder_weight_sf_vec_i = bank_reorder_weight_sf_word[bank_sel][sample_idx];
        ref_reorder_activation_sf_i = bank_reorder_activation_sf[bank_sel][sample_idx];
        @(posedge clk_i);
      end

      while (ref_reorder_done_o !== 1'b1) begin
        @(posedge clk_i);
      end

      expected_reorder_seq[bank_sel] = ref_reorder_seq_o;
      @(posedge clk_i);
    end
  endtask

  task automatic check_bank_reorder_seq(
    input logic [1:0]  bank_sel,
    input logic [23:0] expected_seq
  );
    logic [23:0] actual_seq;
    begin
      unique case (bank_sel)
        2'd0: actual_seq = dut.gen_banks[0].array_u.reorder_seq_q;
        2'd1: actual_seq = dut.gen_banks[1].array_u.reorder_seq_q;
        2'd2: actual_seq = dut.gen_banks[2].array_u.reorder_seq_q;
        default: actual_seq = dut.gen_banks[3].array_u.reorder_seq_q;
      endcase

      if (actual_seq !== expected_seq) begin
        print_fail($sformatf(
          "bank%0d reorder_seq mismatch: expected 0x%06h got 0x%06h",
          bank_sel,
          expected_seq,
          actual_seq
        ));
        $fatal(1, "bank%0d reorder_seq mismatch: expected 0x%06h got 0x%06h", bank_sel, expected_seq, actual_seq);
      end
    end
  endtask

  task automatic check_bank_reorder_ptr(
    input logic [1:0] bank_sel,
    input logic [2:0] expected_ptr
  );
    logic [2:0] actual_ptr;
    begin
      unique case (bank_sel)
        2'd0: actual_ptr = dut.gen_banks[0].array_u.reorder_ptr_q;
        2'd1: actual_ptr = dut.gen_banks[1].array_u.reorder_ptr_q;
        2'd2: actual_ptr = dut.gen_banks[2].array_u.reorder_ptr_q;
        default: actual_ptr = dut.gen_banks[3].array_u.reorder_ptr_q;
      endcase

      if (actual_ptr !== expected_ptr) begin
        print_fail($sformatf(
          "bank%0d reorder_ptr mismatch: expected %0d got %0d",
          bank_sel,
          expected_ptr,
          actual_ptr
        ));
        $fatal(1, "bank%0d reorder_ptr mismatch: expected %0d got %0d", bank_sel, expected_ptr, actual_ptr);
      end
    end
  endtask

  task automatic check_bank_weight_issue_addr(
    input logic [1:0] bank_sel,
    input logic [8:0] expected_addr,
    input string      test_name
  );
    logic [8:0] actual_addr;
    begin
      unique case (bank_sel)
        2'd0: actual_addr = dut.gen_banks[0].array_u.weight_data_addr;
        2'd1: actual_addr = dut.gen_banks[1].array_u.weight_data_addr;
        2'd2: actual_addr = dut.gen_banks[2].array_u.weight_data_addr;
        default: actual_addr = dut.gen_banks[3].array_u.weight_data_addr;
      endcase

      if (actual_addr !== expected_addr) begin
        print_fail($sformatf(
          "%s bank%0d weight addr mismatch: expected %0d got %0d",
          test_name,
          bank_sel,
          expected_addr,
          actual_addr
        ));
        $fatal(1, "%s bank%0d weight addr mismatch: expected %0d got %0d", test_name, bank_sel, expected_addr, actual_addr);
      end
    end
  endtask

  task automatic check_bank_compute_issue_addrs(
    input logic [1:0] bank_sel,
    input logic [8:0] expected_act_addr,
    input logic [5:0] expected_wsf_addr,
    input string      test_name
  );
    logic [8:0] actual_act_addr;
    logic [8:0] actual_act_sf_addr;
    logic [5:0] actual_wsf_addr;
    begin
      unique case (bank_sel)
        2'd0: begin
          actual_act_addr    = dut.gen_banks[0].array_u.activation_data_addr;
          actual_act_sf_addr = dut.gen_banks[0].array_u.activation_sf_mem_addr;
          actual_wsf_addr    = dut.gen_banks[0].array_u.weight_sf_mem_addr;
        end
        2'd1: begin
          actual_act_addr    = dut.gen_banks[1].array_u.activation_data_addr;
          actual_act_sf_addr = dut.gen_banks[1].array_u.activation_sf_mem_addr;
          actual_wsf_addr    = dut.gen_banks[1].array_u.weight_sf_mem_addr;
        end
        2'd2: begin
          actual_act_addr    = dut.gen_banks[2].array_u.activation_data_addr;
          actual_act_sf_addr = dut.gen_banks[2].array_u.activation_sf_mem_addr;
          actual_wsf_addr    = dut.gen_banks[2].array_u.weight_sf_mem_addr;
        end
        default: begin
          actual_act_addr    = dut.gen_banks[3].array_u.activation_data_addr;
          actual_act_sf_addr = dut.gen_banks[3].array_u.activation_sf_mem_addr;
          actual_wsf_addr    = dut.gen_banks[3].array_u.weight_sf_mem_addr;
        end
      endcase

      if (actual_act_addr !== expected_act_addr) begin
        print_fail($sformatf(
          "%s bank%0d activation addr mismatch: expected %0d got %0d",
          test_name,
          bank_sel,
          expected_act_addr,
          actual_act_addr
        ));
        $fatal(1, "%s bank%0d activation addr mismatch: expected %0d got %0d", test_name, bank_sel, expected_act_addr, actual_act_addr);
      end

      if (actual_act_sf_addr !== expected_act_addr) begin
        print_fail($sformatf(
          "%s bank%0d activation_sf addr mismatch: expected %0d got %0d",
          test_name,
          bank_sel,
          expected_act_addr,
          actual_act_sf_addr
        ));
        $fatal(1, "%s bank%0d activation_sf addr mismatch: expected %0d got %0d", test_name, bank_sel, expected_act_addr, actual_act_sf_addr);
      end

      if (actual_wsf_addr !== expected_wsf_addr) begin
        print_fail($sformatf(
          "%s bank%0d weight_sf addr mismatch: expected %0d got %0d",
          test_name,
          bank_sel,
          expected_wsf_addr,
          actual_wsf_addr
        ));
        $fatal(1, "%s bank%0d weight_sf addr mismatch: expected %0d got %0d", test_name, bank_sel, expected_wsf_addr, actual_wsf_addr);
      end
    end
  endtask

  task automatic program_weight_block(
    input logic [1:0]        bank_sel,
    input logic signed [3:0] diag_value
  );
    integer row_idx;
    begin
      for (row_idx = 0; row_idx < 8; row_idx++) begin
        weight_buf_write(
          bank_sel,
          WEIGHT_BASE + row_idx[8:0],
          diagonal_weight_word(diag_value, row_idx)
        );
      end
    end
  endtask

  task automatic pulse_weight_load(input logic use_reorder);
    begin
      @(negedge clk_i);
      weight_load_base_addr_i = WEIGHT_BASE;
      use_reorder_seq_i       = use_reorder;
      weight_load_start_i     = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      weight_load_start_i     = 1'b0;
      use_reorder_seq_i       = 1'b0;
    end
  endtask

  task automatic pulse_compute(
    input logic use_writeback,
    input logic use_reorder
  );
    begin
      @(negedge clk_i);
      activation_addr_i   = ACT_ADDR;
      weight_sf_addr_i    = WEIGHT_SF_ADDR;
      row_weight_rsel_i   = '0;
      fp_flush_flag_i     = '0;
      writeback_enable_i  = use_writeback;
      writeback_addr_i    = WRITEBACK_ADDR;
      threshold_i         = 8'd0;
      use_reorder_seq_i   = use_reorder;
      compute_start_i     = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      compute_start_i     = 1'b0;
      use_reorder_seq_i   = 1'b0;
    end
  endtask

  task automatic check_reordered_weight_load_broadcast(
    input logic [2:0] ptr_value,
    input string      test_name
  );
    integer bank_idx;
    integer load_entry_idx;
    integer slot_idx;
    integer row_idx;
    logic [2:0] seq_value;
    logic [8:0] expected_addr;
    begin
      print_info(test_name);

      @(negedge clk_i);
      weight_load_base_addr_i = WEIGHT_BASE;
      use_reorder_seq_i       = 1'b1;
      weight_load_start_i     = 1'b1;
      @(posedge clk_i);

      for (bank_idx = 0; bank_idx < 4; bank_idx++) begin
        seq_value     = packed_seq_elem(expected_reorder_seq[bank_idx], ptr_value);
        expected_addr = WEIGHT_BASE + {{4{1'b0}}, seq_value, 3'b000};
        check_bank_weight_issue_addr(bank_idx[1:0], expected_addr, test_name);
      end

      @(negedge clk_i);
      weight_load_start_i     = 1'b0;
      weight_load_base_addr_i = '0;
      use_reorder_seq_i       = 1'b0;

      for (load_entry_idx = 0; load_entry_idx < 32; load_entry_idx++) begin
        @(posedge clk_i);
        if (busy_o !== 1'b1) begin
          print_fail($sformatf("%s busy_o should stay high during load entry %0d", test_name, load_entry_idx));
          $fatal(1, "%s busy_o should stay high during load entry %0d", test_name, load_entry_idx);
        end

        if (weight_load_done_o !== 1'b0) begin
          print_fail($sformatf("%s weight_load_done_o should stay low during load entry %0d", test_name, load_entry_idx));
          $fatal(1, "%s weight_load_done_o should stay low during load entry %0d", test_name, load_entry_idx);
        end

        if (load_entry_idx != 31) begin
          row_idx  = (load_entry_idx + 1) % 8;
          slot_idx = (load_entry_idx + 1) / 8;
          for (bank_idx = 0; bank_idx < 4; bank_idx++) begin
            seq_value     = packed_seq_elem(expected_reorder_seq[bank_idx], (ptr_value + slot_idx) & 3'h7);
            expected_addr = WEIGHT_BASE + {{4{1'b0}}, seq_value, 3'b000} + row_idx;
            check_bank_weight_issue_addr(bank_idx[1:0], expected_addr, test_name);
          end
        end
      end

      @(posedge clk_i);
      check_bit($sformatf("%s done", test_name), 1'b1, weight_load_done_o);
      check_bit($sformatf("%s idle", test_name), 1'b0, busy_o);
      check_bank_reorder_ptr(2'd0, ptr_value);
      check_bank_reorder_ptr(2'd1, ptr_value);
      check_bank_reorder_ptr(2'd2, ptr_value);
      check_bank_reorder_ptr(2'd3, ptr_value);

      @(posedge clk_i);
      check_bit($sformatf("%s done_clear", test_name), 1'b0, weight_load_done_o);
      print_pass(test_name);
    end
  endtask

  task automatic run_reordered_compute_broadcast_step(
    input logic [2:0] ptr_value,
    input string      test_name
  );
    integer bank_idx;
    logic [2:0] seq_value;
    logic [8:0] expected_act_addr;
    logic [5:0] expected_wsf_addr;
    logic [2:0] next_ptr;
    begin
      print_info(test_name);
      next_ptr = (ptr_value + 3'd1) & 3'h7;

      @(negedge clk_i);
      activation_addr_i   = ACT_ADDR;
      weight_sf_addr_i    = WEIGHT_SF_ADDR;
      row_weight_rsel_i   = '0;
      fp_flush_flag_i     = '0;
      writeback_enable_i  = 1'b0;
      writeback_addr_i    = WRITEBACK_ADDR;
      threshold_i         = 8'd0;
      use_reorder_seq_i   = 1'b1;
      compute_start_i     = 1'b1;
      @(posedge clk_i);

      for (bank_idx = 0; bank_idx < 4; bank_idx++) begin
        seq_value         = packed_seq_elem(expected_reorder_seq[bank_idx], ptr_value);
        expected_act_addr = ACT_ADDR + {{7{1'b0}}, seq_value};
        expected_wsf_addr = WEIGHT_SF_ADDR + {{4{1'b0}}, seq_value};
        check_bank_compute_issue_addrs(bank_idx[1:0], expected_act_addr, expected_wsf_addr, test_name);
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
      check_bank_reorder_ptr(2'd0, next_ptr);
      check_bank_reorder_ptr(2'd1, next_ptr);
      check_bank_reorder_ptr(2'd2, next_ptr);
      check_bank_reorder_ptr(2'd3, next_ptr);
      check_bit($sformatf("%s busy", test_name), 1'b1, busy_o);
      check_bit($sformatf("%s done_low", test_name), 1'b0, compute_done_o);

      wait_for_compute_done();
      check_bit($sformatf("%s idle", test_name), 1'b0, busy_o);

      @(posedge clk_i);
      check_bit($sformatf("%s done_clear", test_name), 1'b0, compute_done_o);
      print_pass(test_name);
    end
  endtask

  task automatic test_shared_buffer_decode;
    begin
      print_info("test shared buffer decode and readback");

      weight_buf_write(2'd0, 9'd7, 32'h0123_4567);
      weight_buf_write(2'd1, 9'd7, 32'h89AB_CDEF);
      weight_buf_write(2'd2, 9'd7, 32'h1357_9BDF);
      weight_buf_write(2'd3, 9'd7, 32'h2468_ACE0);

      weight_buf_read_expect(2'd0, 9'd7, 32'h0123_4567);
      weight_buf_read_expect(2'd1, 9'd7, 32'h89AB_CDEF);
      weight_buf_read_expect(2'd2, 9'd7, 32'h1357_9BDF);
      weight_buf_read_expect(2'd3, 9'd7, 32'h2468_ACE0);

      weight_sf_buf_write(2'd0, 6'd9, 40'h01_02_03_04_05);
      weight_sf_buf_write(2'd1, 6'd9, 40'h06_07_08_09_0A);
      weight_sf_buf_write(2'd2, 6'd9, 40'h0B_0C_0D_0E_0F);
      weight_sf_buf_write(2'd3, 6'd9, 40'h10_11_12_13_14);

      weight_sf_buf_read_expect(2'd0, 6'd9, 40'h01_02_03_04_05);
      weight_sf_buf_read_expect(2'd1, 6'd9, 40'h06_07_08_09_0A);
      weight_sf_buf_read_expect(2'd2, 6'd9, 40'h0B_0C_0D_0E_0F);
      weight_sf_buf_read_expect(2'd3, 6'd9, 40'h10_11_12_13_14);

      activation_buf_write(2'd0, 9'd21, 32'h1111_2222);
      activation_buf_write(2'd1, 9'd21, 32'h3333_4444);
      activation_buf_write(2'd2, 9'd21, 32'h5555_6666);
      activation_buf_write(2'd3, 9'd21, 32'h7777_8888);

      activation_buf_read_expect(2'd0, 9'd21, 32'h1111_2222);
      activation_buf_read_expect(2'd1, 9'd21, 32'h3333_4444);
      activation_buf_read_expect(2'd2, 9'd21, 32'h5555_6666);
      activation_buf_read_expect(2'd3, 9'd21, 32'h7777_8888);

      activation_sf_buf_write(2'd0, 9'd21, 5'sd1);
      activation_sf_buf_write(2'd1, 9'd21, 5'sd2);
      activation_sf_buf_write(2'd2, 9'd21, 5'sd3);
      activation_sf_buf_write(2'd3, 9'd21, 5'sd4);

      activation_sf_buf_read_expect(2'd0, 9'd21, 5'sd1);
      activation_sf_buf_read_expect(2'd1, 9'd21, 5'sd2);
      activation_sf_buf_read_expect(2'd2, 9'd21, 5'sd3);
      activation_sf_buf_read_expect(2'd3, 9'd21, 5'sd4);

      mask_buf_write(2'd0, 9'd21, 8'h11);
      mask_buf_write(2'd1, 9'd21, 8'h22);
      mask_buf_write(2'd2, 9'd21, 8'h44);
      mask_buf_write(2'd3, 9'd21, 8'h88);

      mask_buf_read_expect(2'd0, 9'd21, 8'h11);
      mask_buf_read_expect(2'd1, 9'd21, 8'h22);
      mask_buf_read_expect(2'd2, 9'd21, 8'h44);
      mask_buf_read_expect(2'd3, 9'd21, 8'h88);

      print_pass("shared buffer decode and readback");
    end
  endtask

  task automatic test_weight_load_broadcast;
    begin
      print_info("test weight load broadcast");

      program_weight_block(2'd0, 4'sd1);
      program_weight_block(2'd1, 4'sd2);
      program_weight_block(2'd2, 4'sd3);
      program_weight_block(2'd3, 4'sd4);

      pulse_weight_load(1'b0);
      @(posedge clk_i);
      check_bit("weight_load busy_o", 1'b1, busy_o);

      wait_for_weight_load_done();
      @(posedge clk_i);

      check_nibble("bank0 row0 col0 slot0", 4'h1, dut.gen_banks[0].array_u.hybrid_acc_u.pe_array_u.gen_rows[0].gen_cols[0].pe_u.weight_slot_q[0]);
      check_nibble("bank1 row0 col0 slot0", 4'h2, dut.gen_banks[1].array_u.hybrid_acc_u.pe_array_u.gen_rows[0].gen_cols[0].pe_u.weight_slot_q[0]);
      check_nibble("bank2 row0 col0 slot0", 4'h3, dut.gen_banks[2].array_u.hybrid_acc_u.pe_array_u.gen_rows[0].gen_cols[0].pe_u.weight_slot_q[0]);
      check_nibble("bank3 row0 col0 slot0", 4'h4, dut.gen_banks[3].array_u.hybrid_acc_u.pe_array_u.gen_rows[0].gen_cols[0].pe_u.weight_slot_q[0]);

      check_nibble("bank0 row7 col7 slot0", 4'h1, dut.gen_banks[0].array_u.hybrid_acc_u.pe_array_u.gen_rows[7].gen_cols[7].pe_u.weight_slot_q[0]);
      check_nibble("bank1 row7 col7 slot0", 4'h2, dut.gen_banks[1].array_u.hybrid_acc_u.pe_array_u.gen_rows[7].gen_cols[7].pe_u.weight_slot_q[0]);
      check_nibble("bank2 row7 col7 slot0", 4'h3, dut.gen_banks[2].array_u.hybrid_acc_u.pe_array_u.gen_rows[7].gen_cols[7].pe_u.weight_slot_q[0]);
      check_nibble("bank3 row7 col7 slot0", 4'h4, dut.gen_banks[3].array_u.hybrid_acc_u.pe_array_u.gen_rows[7].gen_cols[7].pe_u.weight_slot_q[0]);

      print_pass("weight load broadcast");
    end
  endtask

  task automatic test_compute_broadcast_without_writeback;
    begin
      print_info("test compute broadcast without writeback");

      activation_buf_write(2'd0, ACT_ADDR, bank_activation_word[0]);
      activation_buf_write(2'd1, ACT_ADDR, bank_activation_word[1]);
      activation_buf_write(2'd2, ACT_ADDR, bank_activation_word[2]);
      activation_buf_write(2'd3, ACT_ADDR, bank_activation_word[3]);

      activation_sf_buf_write(2'd0, ACT_ADDR, bank_activation_sf[0]);
      activation_sf_buf_write(2'd1, ACT_ADDR, bank_activation_sf[1]);
      activation_sf_buf_write(2'd2, ACT_ADDR, bank_activation_sf[2]);
      activation_sf_buf_write(2'd3, ACT_ADDR, bank_activation_sf[3]);

      weight_sf_buf_write(2'd0, WEIGHT_SF_ADDR, bank_weight_sf_word[0]);
      weight_sf_buf_write(2'd1, WEIGHT_SF_ADDR, bank_weight_sf_word[1]);
      weight_sf_buf_write(2'd2, WEIGHT_SF_ADDR, bank_weight_sf_word[2]);
      weight_sf_buf_write(2'd3, WEIGHT_SF_ADDR, bank_weight_sf_word[3]);

      pulse_compute(1'b0, 1'b0);
      @(posedge clk_i);
      check_bit("compute busy_o", 1'b1, busy_o);

      wait_for_compute_done();
      @(posedge clk_i);

      check_32bit("bank0 activation read data", bank_activation_word[0], dut.gen_banks[0].array_u.activation_data_rdata);
      check_32bit("bank1 activation read data", bank_activation_word[1], dut.gen_banks[1].array_u.activation_data_rdata);
      check_32bit("bank2 activation read data", bank_activation_word[2], dut.gen_banks[2].array_u.activation_data_rdata);
      check_32bit("bank3 activation read data", bank_activation_word[3], dut.gen_banks[3].array_u.activation_data_rdata);

      check_40bit("bank0 weight_sf read data", bank_weight_sf_word[0], dut.gen_banks[0].array_u.weight_sf_mem_rdata);
      check_40bit("bank1 weight_sf read data", bank_weight_sf_word[1], dut.gen_banks[1].array_u.weight_sf_mem_rdata);
      check_40bit("bank2 weight_sf read data", bank_weight_sf_word[2], dut.gen_banks[2].array_u.weight_sf_mem_rdata);
      check_40bit("bank3 weight_sf read data", bank_weight_sf_word[3], dut.gen_banks[3].array_u.weight_sf_mem_rdata);

      check_5bit("bank0 activation_sf read data", bank_activation_sf[0], dut.gen_banks[0].array_u.activation_sf_mem_rdata);
      check_5bit("bank1 activation_sf read data", bank_activation_sf[1], dut.gen_banks[1].array_u.activation_sf_mem_rdata);
      check_5bit("bank2 activation_sf read data", bank_activation_sf[2], dut.gen_banks[2].array_u.activation_sf_mem_rdata);
      check_5bit("bank3 activation_sf read data", bank_activation_sf[3], dut.gen_banks[3].array_u.activation_sf_mem_rdata);

      print_pass("compute broadcast without writeback");
    end
  endtask

  task automatic test_compute_broadcast_with_writeback;
    begin
      print_info("test compute broadcast with writeback");
      reset_writeback_capture();

      pulse_compute(1'b1, 1'b0);
      @(posedge clk_i);
      check_bit("writeback compute busy_o", 1'b1, busy_o);

      wait_for_compute_done();
      @(posedge clk_i);

      check_bit("bank0 writeback data valid", 1'b1, expected_writeback_word_valid[0]);
      check_bit("bank1 writeback data valid", 1'b1, expected_writeback_word_valid[1]);
      check_bit("bank2 writeback data valid", 1'b1, expected_writeback_word_valid[2]);
      check_bit("bank3 writeback data valid", 1'b1, expected_writeback_word_valid[3]);
      check_bit("bank0 writeback sf valid", 1'b1, expected_writeback_sf_valid[0]);
      check_bit("bank1 writeback sf valid", 1'b1, expected_writeback_sf_valid[1]);
      check_bit("bank2 writeback sf valid", 1'b1, expected_writeback_sf_valid[2]);
      check_bit("bank3 writeback sf valid", 1'b1, expected_writeback_sf_valid[3]);
      check_bit("bank0 writeback mask valid", 1'b1, expected_writeback_mask_valid[0]);
      check_bit("bank1 writeback mask valid", 1'b1, expected_writeback_mask_valid[1]);
      check_bit("bank2 writeback mask valid", 1'b1, expected_writeback_mask_valid[2]);
      check_bit("bank3 writeback mask valid", 1'b1, expected_writeback_mask_valid[3]);

      activation_buf_read_expect(2'd0, WRITEBACK_ADDR, expected_writeback_word[0]);
      activation_buf_read_expect(2'd1, WRITEBACK_ADDR, expected_writeback_word[1]);
      activation_buf_read_expect(2'd2, WRITEBACK_ADDR, expected_writeback_word[2]);
      activation_buf_read_expect(2'd3, WRITEBACK_ADDR, expected_writeback_word[3]);

      activation_buf_read_capture(2'd0, WRITEBACK_ADDR, bank_writeback_word[0]);
      activation_buf_read_capture(2'd1, WRITEBACK_ADDR, bank_writeback_word[1]);
      activation_buf_read_capture(2'd2, WRITEBACK_ADDR, bank_writeback_word[2]);
      activation_buf_read_capture(2'd3, WRITEBACK_ADDR, bank_writeback_word[3]);

      check_32bit("bank0 writeback data", expected_writeback_word[0], bank_writeback_word[0]);
      check_32bit("bank1 writeback data", expected_writeback_word[1], bank_writeback_word[1]);
      check_32bit("bank2 writeback data", expected_writeback_word[2], bank_writeback_word[2]);
      check_32bit("bank3 writeback data", expected_writeback_word[3], bank_writeback_word[3]);

      activation_sf_buf_read_expect(2'd0, WRITEBACK_ADDR, expected_writeback_sf[0]);
      activation_sf_buf_read_expect(2'd1, WRITEBACK_ADDR, expected_writeback_sf[1]);
      activation_sf_buf_read_expect(2'd2, WRITEBACK_ADDR, expected_writeback_sf[2]);
      activation_sf_buf_read_expect(2'd3, WRITEBACK_ADDR, expected_writeback_sf[3]);

      mask_buf_read_expect(2'd0, WRITEBACK_ADDR, expected_writeback_mask[0]);
      mask_buf_read_expect(2'd1, WRITEBACK_ADDR, expected_writeback_mask[1]);
      mask_buf_read_expect(2'd2, WRITEBACK_ADDR, expected_writeback_mask[2]);
      mask_buf_read_expect(2'd3, WRITEBACK_ADDR, expected_writeback_mask[3]);

      if (bank_writeback_word[0] === bank_writeback_word[1]) begin
        print_fail($sformatf(
          "bank0/bank1 writeback words should differ: expected different values got 0x%08h and 0x%08h",
          bank_writeback_word[0],
          bank_writeback_word[1]
        ));
        $fatal(
          1,
          "bank0/bank1 writeback words should differ: expected different values got 0x%08h and 0x%08h",
          bank_writeback_word[0],
          bank_writeback_word[1]
        );
      end
      if (bank_writeback_word[1] === bank_writeback_word[2]) begin
        print_fail($sformatf(
          "bank1/bank2 writeback words should differ: expected different values got 0x%08h and 0x%08h",
          bank_writeback_word[1],
          bank_writeback_word[2]
        ));
        $fatal(
          1,
          "bank1/bank2 writeback words should differ: expected different values got 0x%08h and 0x%08h",
          bank_writeback_word[1],
          bank_writeback_word[2]
        );
      end
      if (bank_writeback_word[2] === bank_writeback_word[3]) begin
        print_fail($sformatf(
          "bank2/bank3 writeback words should differ: expected different values got 0x%08h and 0x%08h",
          bank_writeback_word[2],
          bank_writeback_word[3]
        ));
        $fatal(
          1,
          "bank2/bank3 writeback words should differ: expected different values got 0x%08h and 0x%08h",
          bank_writeback_word[2],
          bank_writeback_word[3]
        );
      end

      print_pass("compute broadcast with writeback");
    end
  endtask

  task automatic test_reordered_address_flow;
    begin
      print_info("test reordered address flow");

      check_bank_reorder_ptr(2'd0, 3'd0);
      check_bank_reorder_ptr(2'd1, 3'd0);
      check_bank_reorder_ptr(2'd2, 3'd0);
      check_bank_reorder_ptr(2'd3, 3'd0);

      check_reordered_weight_load_broadcast(3'd0, "reordered_weight_load_batch0");
      run_reordered_compute_broadcast_step(3'd0, "reordered_compute_0");
      run_reordered_compute_broadcast_step(3'd1, "reordered_compute_1");
      run_reordered_compute_broadcast_step(3'd2, "reordered_compute_2");
      run_reordered_compute_broadcast_step(3'd3, "reordered_compute_3");

      check_reordered_weight_load_broadcast(3'd4, "reordered_weight_load_batch1");
      run_reordered_compute_broadcast_step(3'd4, "reordered_compute_4");
      run_reordered_compute_broadcast_step(3'd5, "reordered_compute_5");
      run_reordered_compute_broadcast_step(3'd6, "reordered_compute_6");
      run_reordered_compute_broadcast_step(3'd7, "reordered_compute_7");

      check_bank_reorder_ptr(2'd0, 3'd0);
      check_bank_reorder_ptr(2'd1, 3'd0);
      check_bank_reorder_ptr(2'd2, 3'd0);
      check_bank_reorder_ptr(2'd3, 3'd0);

      print_pass("reordered address flow");
    end
  endtask

  task automatic test_reorder_start_ptr_reset_behavior;
    begin
      print_info("test reorder start ptr reset behavior");

      sf_reorder_threshold_i = 3'd3;
      sf_reorder_max_iter_i  = 3'd2;

      run_reordered_compute_broadcast_step(3'd0, "reorder_ptr_reset_setup_compute");

      @(negedge clk_i);
      weight_load_base_addr_i = WEIGHT_BASE;
      use_reorder_seq_i       = 1'b1;
      weight_load_start_i     = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      weight_load_start_i     = 1'b0;
      weight_load_base_addr_i = '0;
      use_reorder_seq_i       = 1'b0;
      activation_addr_i       = ACT_ADDR;
      weight_sf_addr_i        = WEIGHT_SF_ADDR;
      sf_reorder_start_i      = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      sf_reorder_start_i      = 1'b0;
      activation_addr_i       = '0;
      weight_sf_addr_i        = '0;

      wait_for_weight_load_done();
      @(posedge clk_i);
      check_bank_reorder_ptr(2'd0, 3'd1);
      check_bank_reorder_ptr(2'd1, 3'd1);
      check_bank_reorder_ptr(2'd2, 3'd1);
      check_bank_reorder_ptr(2'd3, 3'd1);

      program_bank_reorder_window(2'd0);
      program_bank_reorder_window(2'd1);
      program_bank_reorder_window(2'd2);
      program_bank_reorder_window(2'd3);

      run_reference_reorder_for_bank(0);
      run_reference_reorder_for_bank(1);
      run_reference_reorder_for_bank(2);
      run_reference_reorder_for_bank(3);

      @(negedge clk_i);
      activation_addr_i       = ACT_ADDR;
      weight_sf_addr_i        = WEIGHT_SF_ADDR;
      sf_reorder_start_i      = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      sf_reorder_start_i      = 1'b0;
      activation_addr_i       = '0;
      weight_sf_addr_i        = '0;
      @(posedge clk_i);
      check_bank_reorder_ptr(2'd0, 3'd0);
      check_bank_reorder_ptr(2'd1, 3'd0);
      check_bank_reorder_ptr(2'd2, 3'd0);
      check_bank_reorder_ptr(2'd3, 3'd0);

      wait_for_not_busy(160, "reorder_start_ptr_reset_accept");
      check_bank_reorder_seq(2'd0, expected_reorder_seq[0]);
      check_bank_reorder_seq(2'd1, expected_reorder_seq[1]);
      check_bank_reorder_seq(2'd2, expected_reorder_seq[2]);
      check_bank_reorder_seq(2'd3, expected_reorder_seq[3]);

      sf_reorder_threshold_i = '0;
      sf_reorder_max_iter_i  = '0;

      print_pass("reorder start ptr reset behavior");
    end
  endtask

  task automatic test_serial_sf_reorder;
    begin
      print_info("test shared serial sf reorder");

      apply_async_reset();

      sf_reorder_threshold_i = 3'd3;
      sf_reorder_max_iter_i  = 3'd2;

      program_bank_reorder_window(2'd0);
      program_bank_reorder_window(2'd1);
      program_bank_reorder_window(2'd2);
      program_bank_reorder_window(2'd3);

      run_reference_reorder_for_bank(0);
      run_reference_reorder_for_bank(1);
      run_reference_reorder_for_bank(2);
      run_reference_reorder_for_bank(3);

      @(negedge clk_i);
      activation_addr_i   = ACT_ADDR;
      weight_sf_addr_i    = WEIGHT_SF_ADDR;
      sf_reorder_start_i  = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      sf_reorder_start_i  = 1'b0;

      @(posedge clk_i);
      check_bit("sf_reorder busy_o", 1'b1, busy_o);

      wait_for_not_busy(160, "serial_sf_reorder_complete");

      check_bank_reorder_seq(2'd0, expected_reorder_seq[0]);
      check_bank_reorder_seq(2'd1, expected_reorder_seq[1]);
      check_bank_reorder_seq(2'd2, expected_reorder_seq[2]);
      check_bank_reorder_seq(2'd3, expected_reorder_seq[3]);

      @(negedge clk_i);
      activation_addr_i   = ACT_ADDR;
      weight_sf_addr_i    = WEIGHT_SF_ADDR;
      sf_reorder_start_i  = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      sf_reorder_start_i  = 1'b0;

      @(posedge clk_i);
      check_bit("sf_reorder restart busy_o", 1'b1, busy_o);

      apply_async_reset();
      check_bit("sf_reorder async reset busy_o", 1'b0, busy_o);

      check_bank_reorder_seq(2'd0, expected_reorder_seq[0]);
      check_bank_reorder_seq(2'd1, expected_reorder_seq[1]);
      check_bank_reorder_seq(2'd2, expected_reorder_seq[2]);
      check_bank_reorder_seq(2'd3, expected_reorder_seq[3]);

      sf_reorder_threshold_i = '0;
      sf_reorder_max_iter_i  = '0;
      activation_addr_i      = '0;
      weight_sf_addr_i       = '0;

      print_pass("shared serial sf reorder");
    end
  endtask

  initial begin
    reset_inputs();
    init_test_vectors();
    reset_writeback_capture();
    apply_async_reset();
    check_group_address_contracts();
    print_pass("group/local address width contract");
    clear_group();

    test_shared_buffer_decode();
    test_serial_sf_reorder();
    clear_group();
    test_weight_load_broadcast();
    test_compute_broadcast_without_writeback();
    test_compute_broadcast_with_writeback();
    test_reordered_address_flow();
    test_reorder_start_ptr_reset_behavior();

    print_pass("pe_array_group_tb PASS");
    #1000;
    $finish;
  end

endmodule
