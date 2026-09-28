// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module prob_cache_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic        clk_i;
  logic        rst_ni;
  logic        clear_i;
  logic        update_valid_i;
  logic [12:0] index_i;
  logic  [7:0] probability_i;
  logic  [3:0] timestamp_i;
  logic  [7:0] threshold_i;
  logic  [4:0] seq_read_addr_i;

  logic        busy_o;
  logic        seq_full_o;
  logic  [5:0] seq_count_o;
  logic [16:0] seq_read_data_o;

  prob_cache dut (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .clear_i        (clear_i),
    .update_valid_i (update_valid_i),
    .index_i        (index_i),
    .probability_i  (probability_i),
    .timestamp_i    (timestamp_i),
    .threshold_i    (threshold_i),
    .seq_read_addr_i(seq_read_addr_i),
    .busy_o         (busy_o),
    .seq_full_o     (seq_full_o),
    .seq_count_o    (seq_count_o),
    .seq_read_data_o(seq_read_data_o)
  );

  initial begin
    clk_i = 1'b0;
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

  task automatic check_bit(
    input string name,
    input logic  expected_value,
    input logic  actual_value
  );
    begin
      if (actual_value !== expected_value) begin
        print_fail($sformatf("%s mismatch: expected %0b got %0b", name, expected_value, actual_value));
        $fatal(1, "%s mismatch: expected %0b got %0b", name, expected_value, actual_value);
      end
      print_pass($sformatf("%s matched expected %0b", name, expected_value));
    end
  endtask

  task automatic check_u6(
    input string      name,
    input logic [5:0] expected_value,
    input logic [5:0] actual_value
  );
    begin
      if (actual_value !== expected_value) begin
        print_fail($sformatf("%s mismatch: expected %0d got %0d", name, expected_value, actual_value));
        $fatal(1, "%s mismatch: expected %0d got %0d", name, expected_value, actual_value);
      end
      print_pass($sformatf("%s matched expected %0d", name, expected_value));
    end
  endtask

  task automatic apply_reset;
    begin
      clear_i         = 1'b0;
      update_valid_i  = 1'b0;
      index_i         = '0;
      probability_i   = '0;
      timestamp_i     = '0;
      threshold_i     = '0;
      seq_read_addr_i = '0;

      rst_ni = 1'b0;
      repeat (2) @(posedge clk_i);
      rst_ni = 1'b1;
      @(posedge clk_i);
    end
  endtask

  task automatic send_update(
    input logic [12:0] index_value,
    input logic  [7:0] probability_value,
    input logic  [3:0] timestamp_value,
    input logic  [7:0] threshold_value
  );
    begin
      if (busy_o !== 1'b0) begin
        print_fail("send_update called while busy_o is high");
        $fatal(1, "send_update called while busy_o is high");
      end

      @(negedge clk_i);
      update_valid_i = 1'b1;
      index_i        = index_value;
      probability_i  = probability_value;
      timestamp_i    = timestamp_value;
      threshold_i    = threshold_value;

      @(posedge clk_i);
      @(negedge clk_i);
      update_valid_i = 1'b0;
      index_i        = '0;
      probability_i  = '0;
      timestamp_i    = '0;
      threshold_i    = '0;

      wait (busy_o == 1'b0);
      @(posedge clk_i);
    end
  endtask

  task automatic seq_read_expect(
    input logic [4:0]  read_addr,
    input logic [16:0] expected_data,
    input string       test_name
  );
    begin
      @(negedge clk_i);
      seq_read_addr_i = read_addr;
      @(posedge clk_i);
      if (seq_read_data_o !== expected_data) begin
        print_fail($sformatf("%s seq read mismatch at addr %0d: expected 0x%05h got 0x%05h",
                             test_name, read_addr, expected_data, seq_read_data_o));
        $fatal(1, "%s seq read mismatch at addr %0d: expected 0x%05h got 0x%05h",
               test_name, read_addr, expected_data, seq_read_data_o);
      end
      print_pass($sformatf("%s seq read addr %0d matched 0x%05h", test_name, read_addr, expected_data));
    end
  endtask

  function automatic logic [16:0] pack_seq_word(
    input logic [12:0] index_value,
    input logic  [3:0] timestamp_value
  );
    pack_seq_word = {index_value, timestamp_value};
  endfunction

  task automatic check_line_flags(
    input logic [8:0] line_addr,
    input logic       expected_valid,
    input logic       expected_queued,
    input string      test_name
  );
    begin
      check_bit($sformatf("%s valid_mem[%0d]", test_name, line_addr), expected_valid, dut.valid_mem[line_addr]);
      check_bit($sformatf("%s queued_mem[%0d]", test_name, line_addr), expected_queued, dut.queued_mem[line_addr]);
    end
  endtask

  task automatic check_base_reset_state;
    integer seq_idx;
    begin
      check_bit("reset busy_o", 1'b0, busy_o);
      check_bit("reset seq_full_o", 1'b0, seq_full_o);
      check_u6("reset seq_count_o", 6'd0, seq_count_o);

      for (seq_idx = 0; seq_idx < 4; seq_idx++) begin
        seq_read_expect(seq_idx[4:0], 17'd0, "reset empty sequence");
      end

      check_line_flags(9'd0, 1'b0, 1'b0, "reset line0");
      print_pass("reset state is empty");
    end
  endtask

  task automatic check_miss_below_threshold;
    logic [8:0] line_addr;
    begin
      line_addr = 9'h155;
      send_update(13'h1555, 8'd9, 4'hA, 8'd20);

      check_line_flags(line_addr, 1'b1, 1'b0, "miss below threshold");
      check_u6("miss below threshold seq_count_o", 6'd0, seq_count_o);
      seq_read_expect(5'd0, 17'd0, "miss below threshold sequence stays empty");
      print_pass("miss below threshold updated cache only");
    end
  endtask

  task automatic check_hit_crosses_threshold_once;
    logic [8:0] line_addr;
    begin
      line_addr = 9'h123;

      send_update(13'h123, 8'd7, 4'h4, 8'd20);
      check_line_flags(line_addr, 1'b1, 1'b0, "initial miss below threshold");

      send_update(13'h123, 8'd15, 4'hF, 8'd20);
      check_line_flags(line_addr, 1'b1, 1'b1, "hit crossing threshold");
      check_u6("first threshold crossing seq_count_o", 6'd1, seq_count_o);
      seq_read_expect(5'd0, pack_seq_word(13'h123, 4'h4), "first threshold insert");

      send_update(13'h123, 8'd3, 4'h1, 8'd20);
      check_line_flags(line_addr, 1'b1, 1'b1, "hit above threshold without duplicate insert");
      check_u6("duplicate threshold hit seq_count_o", 6'd1, seq_count_o);
      seq_read_expect(5'd0, pack_seq_word(13'h123, 4'h4), "duplicate suppression holds first entry");
    end
  endtask

  task automatic check_miss_insert_and_tag_replacement;
    logic [8:0] line_addr;
    begin
      line_addr = 9'h077;

      send_update(13'h277, 8'd30, 4'h9, 8'd20);
      check_line_flags(line_addr, 1'b1, 1'b1, "miss immediate threshold insert");
      check_u6("miss immediate threshold insert seq_count_o", 6'd2, seq_count_o);
      seq_read_expect(5'd1, pack_seq_word(13'h277, 4'h9), "miss insert uses input timestamp");

      send_update(13'h677, 8'd5, 4'h2, 8'd20);
      check_line_flags(line_addr, 1'b1, 1'b0, "tag mismatch replacement clears queued state");
      check_u6("replacement below threshold seq_count_o", 6'd2, seq_count_o);
      send_update(13'h677, 8'd20, 4'h0, 8'd20);
      check_line_flags(line_addr, 1'b1, 1'b1, "replacement entry can insert after replacement");
      check_u6("replacement entry post-threshold seq_count_o", 6'd3, seq_count_o);
      seq_read_expect(5'd2, pack_seq_word(13'h677, 4'h2), "replacement insert uses cached timestamp");
    end
  endtask

  task automatic check_sequence_full_behavior;
    integer      entry_idx;
    logic [12:0] fill_index;
    logic [16:0] last_expected_word;
    begin
      for (entry_idx = 0; entry_idx < 29; entry_idx++) begin
        fill_index = {entry_idx[3:0], entry_idx[8:0]};
        send_update(fill_index, 8'd40, entry_idx[3:0], 8'd20);
      end

      check_u6("sequence full seq_count_o", 6'd32, seq_count_o);
      check_bit("sequence full seq_full_o", 1'b1, seq_full_o);
      last_expected_word = pack_seq_word({4'hC, 9'h01C}, 4'hC);
      seq_read_expect(5'd31, last_expected_word, "last sequence slot before full");

      send_update(13'h1FF, 8'd50, 4'h6, 8'd20);
      check_u6("blocked insert when full seq_count_o", 6'd32, seq_count_o);
      check_bit("blocked insert when full seq_full_o", 1'b1, seq_full_o);
      check_line_flags(9'h1FF, 1'b1, 1'b0, "blocked insert while full");
    end
  endtask

  task automatic check_clear_behavior;
    integer line_idx;
    integer cycle_count;
    begin
      check_bit("clear start busy_o", 1'b0, busy_o);

      @(negedge clk_i);
      clear_i = 1'b1;
      seq_read_addr_i = 5'd0;
      @(posedge clk_i);
      @(negedge clk_i);
      clear_i = 1'b0;

      cycle_count = 0;
      while (1) begin
        @(posedge clk_i);

        if (dut.state_q == dut.CLEAR) begin
          cycle_count++;

          if (cycle_count == 1) begin
            check_bit("clear immediate busy_o", 1'b1, busy_o);
            check_u6("clear immediate seq_count_o", 6'd0, seq_count_o);
            check_bit("clear immediate seq_full_o", 1'b0, seq_full_o);

            if (seq_read_data_o !== 17'd0) begin
              print_fail($sformatf("clear immediate sequence reset mismatch at addr 0: expected 0x%05h got 0x%05h",
                                   17'd0, seq_read_data_o));
              $fatal(1, "clear immediate sequence reset mismatch at addr 0: expected 0x%05h got 0x%05h",
                     17'd0, seq_read_data_o);
            end
            print_pass("clear immediate sequence reset addr 0 matched 0x00000");
          end

          if (cycle_count > 512) begin
            print_fail($sformatf("clear should take exactly 512 cycles, got more than 512 (currently %0d)",
                                 cycle_count));
            $fatal(1, "clear should take exactly 512 cycles, got more than 512 (currently %0d)",
                   cycle_count);
          end
        end else begin
          break;
        end
      end

      if (cycle_count != 512) begin
        print_fail($sformatf("clear should take exactly 512 cycles, got %0d", cycle_count));
        $fatal(1, "clear should take exactly 512 cycles, got %0d", cycle_count);
      end

      for (line_idx = 0; line_idx < 512; line_idx++) begin
        if (dut.valid_mem[line_idx] !== 1'b0) begin
          print_fail($sformatf("valid_mem[%0d] not cleared", line_idx));
          $fatal(1, "valid_mem[%0d] not cleared", line_idx);
        end

        if (dut.queued_mem[line_idx] !== 1'b0) begin
          print_fail($sformatf("queued_mem[%0d] not cleared", line_idx));
          $fatal(1, "queued_mem[%0d] not cleared", line_idx);
        end
      end

      seq_read_expect(5'd0, 17'd0, "sequence remains empty after clear");
      check_bit("clear final busy_o", 1'b0, busy_o);
      check_u6("clear final seq_count_o", 6'd0, seq_count_o);
      check_bit("clear final seq_full_o", 1'b0, seq_full_o);
      print_pass("clear swept valid_mem and queued_mem");
    end
  endtask

  initial begin
    apply_reset();

    print_info("Check reset state");
    check_base_reset_state();

    print_info("Check one miss below threshold");
    check_miss_below_threshold();

    print_info("Check hit accumulation and first threshold insert");
    check_hit_crosses_threshold_once();

    print_info("Check miss insert and tag replacement reuse");
    check_miss_insert_and_tag_replacement();

    print_info("Check sequence full behavior");
    check_sequence_full_behavior();

    print_info("Check clear behavior");
    check_clear_behavior();

    print_pass("prob_cache directed testbench completed");
    $finish;
  end

endmodule
