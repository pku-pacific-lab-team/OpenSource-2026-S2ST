// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module flush_serializer_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";
  localparam int    REQUEST_QUEUE_DEPTH       = 8;
  localparam int    SPARSE_PAIR_BATCH_COUNT  = REQUEST_QUEUE_DEPTH - 1;
  localparam int    SPARSE_PAIR_REQUEST_COUNT = SPARSE_PAIR_BATCH_COUNT * 2;

  logic        clk_i;
  logic        rst_ni;
  logic        clear_i;
  logic        enqueue_i;
  logic  [7:0] flush_mask_i;
  logic [95:0] flush_result_vec_i;
  logic [39:0] flush_sf_vec_i;

  logic               flush_valid_o;
  logic         [2:0] flush_lane_idx_o;
  logic signed [11:0] flush_result_o;
  logic         [4:0] flush_sf_o;

  flush_serializer #(
    .REQUEST_QUEUE_DEPTH(REQUEST_QUEUE_DEPTH)
  ) dut (
    .clk_i           (clk_i),
    .rst_ni          (rst_ni),
    .clear_i         (clear_i),
    .enqueue_i       (enqueue_i),
    .flush_mask_i    (flush_mask_i),
    .flush_result_vec_i(flush_result_vec_i),
    .flush_sf_vec_i  (flush_sf_vec_i),
    .flush_valid_o   (flush_valid_o),
    .flush_lane_idx_o(flush_lane_idx_o),
    .flush_result_o  (flush_result_o),
    .flush_sf_o      (flush_sf_o)
  );

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b1;
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

  function automatic logic [95:0] pack_int12x8(
    input logic signed [11:0] lane0,
    input logic signed [11:0] lane1,
    input logic signed [11:0] lane2,
    input logic signed [11:0] lane3,
    input logic signed [11:0] lane4,
    input logic signed [11:0] lane5,
    input logic signed [11:0] lane6,
    input logic signed [11:0] lane7
  );
    pack_int12x8 = {
      lane7[11:0], lane6[11:0], lane5[11:0], lane4[11:0],
      lane3[11:0], lane2[11:0], lane1[11:0], lane0[11:0]
    };
  endfunction

  function automatic logic [39:0] pack_sf5x8(
    input logic [4:0] lane0,
    input logic [4:0] lane1,
    input logic [4:0] lane2,
    input logic [4:0] lane3,
    input logic [4:0] lane4,
    input logic [4:0] lane5,
    input logic [4:0] lane6,
    input logic [4:0] lane7
  );
    pack_sf5x8 = {
      lane7, lane6, lane5, lane4,
      lane3, lane2, lane1, lane0
    };
  endfunction

  task automatic check_request(
    input logic               expected_valid,
    input logic         [2:0] expected_lane_idx,
    input logic signed [11:0] expected_result,
    input logic         [4:0] expected_sf,
    input string              test_name
  );
    begin
      print_info($sformatf("Run request check: %s", test_name));

      if (flush_valid_o !== expected_valid) begin
        print_fail($sformatf(
          "%s valid mismatch: expected %0b got %0b",
          test_name,
          expected_valid,
          flush_valid_o
        ));
        $fatal(1, "%s valid mismatch: expected %0b got %0b", test_name, expected_valid, flush_valid_o);
      end

      if (flush_lane_idx_o !== expected_lane_idx) begin
        print_fail($sformatf(
          "%s lane mismatch: expected %0d got %0d",
          test_name,
          expected_lane_idx,
          flush_lane_idx_o
        ));
        $fatal(1, "%s lane mismatch: expected %0d got %0d", test_name, expected_lane_idx, flush_lane_idx_o);
      end

      if (flush_result_o !== expected_result) begin
        print_fail($sformatf(
          "%s result mismatch: expected %0d got %0d",
          test_name,
          expected_result,
          flush_result_o
        ));
        $fatal(1, "%s result mismatch: expected %0d got %0d", test_name, expected_result, flush_result_o);
      end

      if (flush_sf_o !== expected_sf) begin
        print_fail($sformatf(
          "%s sf mismatch: expected %0d got %0d",
          test_name,
          expected_sf,
          flush_sf_o
        ));
        $fatal(1, "%s sf mismatch: expected %0d got %0d", test_name, expected_sf, flush_sf_o);
      end

      print_pass($sformatf("Request check passed: %s", test_name));
    end
  endtask

  task automatic check_no_request(input string test_name);
    begin
      check_request(1'b0, 3'd0, 12'sd0, 5'd0, test_name);
    end
  endtask

  task automatic drive_inputs(
    input logic        next_clear,
    input logic        next_enqueue,
    input logic  [7:0] next_flush_mask,
    input logic [95:0] next_flush_result_vec,
    input logic [39:0] next_flush_sf_vec
  );
    begin
      clear_i            = next_clear;
      enqueue_i          = next_enqueue;
      flush_mask_i       = next_flush_mask;
      flush_result_vec_i = next_flush_result_vec;
      flush_sf_vec_i     = next_flush_sf_vec;
    end
  endtask

  task automatic drive_sparse_pair_batch(input int batch_idx);
    begin
      drive_inputs(
        1'b0,
        1'b1,
        8'b0100_0010,
        pack_int12x8(
          -12'sd500 - batch_idx,
           12'sd100 + batch_idx,
           12'sd300 + batch_idx,
          -12'sd400 - batch_idx,
           12'sd500 + batch_idx,
          -12'sd600 - batch_idx,
          -12'sd200 - batch_idx,
           12'sd700 + batch_idx
        ),
        pack_sf5x8(
          batch_idx + 5'd20,
          batch_idx + 5'd1,
          batch_idx + 5'd21,
          batch_idx + 5'd22,
          batch_idx + 5'd23,
          batch_idx + 5'd24,
          batch_idx + 5'd9,
          batch_idx + 5'd25
        )
      );
    end
  endtask

  task automatic check_sparse_pair_request(
    input int    request_idx,
    input string test_name
  );
    int                  batch_idx;
    logic          [2:0] expected_lane_idx;
    logic signed  [11:0] expected_result;
    logic          [4:0] expected_sf;
    begin
      batch_idx = request_idx / 2;

      if ((request_idx % 2) == 0) begin
        expected_lane_idx = 3'd1;
        expected_result   = 12'sd100 + batch_idx;
        expected_sf       = batch_idx + 5'd1;
      end else begin
        expected_lane_idx = 3'd6;
        expected_result   = -12'sd200 - batch_idx;
        expected_sf       = batch_idx + 5'd9;
      end

      check_request(1'b1, expected_lane_idx, expected_result, expected_sf, test_name);
    end
  endtask

  initial begin
    drive_inputs(1'b1, 1'b0, 8'h00, '0, '0);

    print_info("Check synchronous clear initializes serializer state");
    @(posedge clk_i);
    #1;
    check_no_request("clear_initializes_state");

    drive_inputs(
      1'b0,
      1'b1,
      8'h00,
      pack_int12x8(
        12'sd1, 12'sd2, 12'sd3, 12'sd4,
        12'sd5, 12'sd6, 12'sd7, 12'sd8
      ),
      pack_sf5x8(
        5'd1, 5'd2, 5'd3, 5'd4,
        5'd5, 5'd6, 5'd7, 5'd8
      )
    );
    @(posedge clk_i);
    #1;
    check_no_request("empty_batch_has_no_output");
    drive_inputs(1'b0, 1'b0, 8'h00, '0, '0);
    @(posedge clk_i);
    #1;
    check_no_request("empty_batch_is_not_queued");

    drive_inputs(
      1'b0,
      1'b1,
      8'b0010_0000,
      pack_int12x8(
        12'sd11,  12'sd12,  12'sd13,  12'sd14,
        12'sd15, -12'sd205, 12'sd17,  12'sd18
      ),
      pack_sf5x8(
        5'd1, 5'd2, 5'd3, 5'd4,
        5'd5, 5'd21, 5'd7, 5'd8
      )
    );
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd5, -12'sd205, 5'd21, "single_lane_batch_emits_correct_payload");
    drive_inputs(1'b0, 1'b0, 8'h00, '0, '0);
    @(posedge clk_i);
    #1;
    check_no_request("single_lane_batch_drains_in_one_cycle");

    drive_inputs(
      1'b0,
      1'b1,
      8'b1011_0101,
      pack_int12x8(
        12'sd10,  12'sd111, -12'sd20, 12'sd222,
        12'sd30, -12'sd40,  12'sd333, 12'sd50
      ),
      pack_sf5x8(
        5'd2, 5'd12, 5'd4, 5'd14,
        5'd6, 5'd8, 5'd16, 5'd10
      )
    );
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd0, 12'sd10, 5'd2, "multi_lane_batch_lane0");
    drive_inputs(1'b0, 1'b0, 8'h00, '0, '0);
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd2, -12'sd20, 5'd4, "multi_lane_batch_lane2");
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd4, 12'sd30, 5'd6, "multi_lane_batch_lane4");
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd5, -12'sd40, 5'd8, "multi_lane_batch_lane5");
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd7, 12'sd50, 5'd10, "multi_lane_batch_lane7");
    @(posedge clk_i);
    #1;
    check_no_request("multi_lane_batch_drains_without_bubbles");

    drive_inputs(
      1'b0,
      1'b1,
      8'b0000_1010,
      pack_int12x8(
        12'sd700, 12'sd101, 12'sd702, -12'sd303,
        12'sd704, 12'sd705, 12'sd706, 12'sd707
      ),
      pack_sf5x8(
        5'd7, 5'd11, 5'd9, 5'd13,
        5'd5, 5'd4,  5'd3, 5'd2
      )
    );
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd1, 12'sd101, 5'd11, "consecutive_batches_batch_a_lane1");

    drive_inputs(
      1'b0,
      1'b1,
      8'b0100_0001,
      pack_int12x8(
        -12'sd400, 12'sd801, 12'sd802, 12'sd803,
        12'sd804,  12'sd805, 12'sd906, 12'sd807
      ),
      pack_sf5x8(
        5'd19, 5'd18, 5'd17, 5'd16,
        5'd15, 5'd14, 5'd23, 5'd12
      )
    );
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd3, -12'sd303, 5'd13, "consecutive_batches_batch_a_lane3");
    drive_inputs(1'b0, 1'b0, 8'h00, '0, '0);
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd0, -12'sd400, 5'd19, "consecutive_batches_batch_b_lane0");
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd6, 12'sd906, 5'd23, "consecutive_batches_batch_b_lane6");
    @(posedge clk_i);
    #1;
    check_no_request("consecutive_batches_drain_cleanly");

    print_info("Check sparse batches stay within the configured per-lane request FIFO depth");
    for (int batch_idx = 0; batch_idx < SPARSE_PAIR_BATCH_COUNT; batch_idx++) begin
      drive_sparse_pair_batch(batch_idx);
      @(posedge clk_i);
      #1;
      check_sparse_pair_request(
        batch_idx,
        $sformatf("sparse_pair_enqueue_phase_req_%0d", batch_idx)
      );
    end

    drive_inputs(1'b0, 1'b0, 8'h00, '0, '0);
    for (int req_idx = SPARSE_PAIR_BATCH_COUNT; req_idx < SPARSE_PAIR_REQUEST_COUNT; req_idx++) begin
      @(posedge clk_i);
      #1;
      check_sparse_pair_request(
        req_idx,
        $sformatf("sparse_pair_drain_phase_req_%0d", req_idx)
      );
    end
    @(posedge clk_i);
    #1;
    check_no_request("sparse_pair_queue_drains_cleanly");

    drive_inputs(
      1'b0,
      1'b1,
      8'b0001_0101,
      pack_int12x8(
        12'sd31, 12'sd32, 12'sd33, 12'sd34,
        12'sd35, 12'sd36, 12'sd37, 12'sd38
      ),
      pack_sf5x8(
        5'd1, 5'd2, 5'd3, 5'd4,
        5'd5, 5'd6, 5'd7, 5'd8
      )
    );
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd0, 12'sd31, 5'd1, "clear_flushes_active_lane0_before_reset_sequence");

    drive_inputs(
      1'b0,
      1'b1,
      8'b1000_0000,
      pack_int12x8(
        12'sd41, 12'sd42, 12'sd43, 12'sd44,
        12'sd45, 12'sd46, 12'sd47, -12'sd480
      ),
      pack_sf5x8(
        5'd9,  5'd10, 5'd11, 5'd12,
        5'd13, 5'd14, 5'd15, 5'd16
      )
    );
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd2, 12'sd33, 5'd3, "clear_test_batch_keeps_running_before_reset");

    drive_inputs(
      1'b1,
      1'b1,
      8'b1111_1111,
      pack_int12x8(
        -12'sd1, -12'sd2, -12'sd3, -12'sd4,
        -12'sd5, -12'sd6, -12'sd7, -12'sd8
      ),
      pack_sf5x8(
        5'd31, 5'd30, 5'd29, 5'd28,
        5'd27, 5'd26, 5'd25, 5'd24
      )
    );
    @(posedge clk_i);
    #1;
    check_no_request("clear_empties_active_and_queued_state");

    drive_inputs(1'b0, 1'b0, 8'h00, '0, '0);
    @(posedge clk_i);
    #1;
    check_no_request("serializer_stays_empty_after_clear");

    drive_inputs(
      1'b0,
      1'b1,
      8'b1000_0000,
      pack_int12x8(
        12'sd51, 12'sd52, 12'sd53, 12'sd54,
        12'sd55, 12'sd56, 12'sd57, 12'sd888
      ),
      pack_sf5x8(
        5'd1, 5'd2, 5'd3, 5'd4,
        5'd5, 5'd6, 5'd7, 5'd18
      )
    );
    @(posedge clk_i);
    #1;
    check_request(1'b1, 3'd7, 12'sd888, 5'd18, "post_clear_new_batch_starts_cleanly");
    drive_inputs(1'b0, 1'b0, 8'h00, '0, '0);
    @(posedge clk_i);
    #1;
    check_no_request("post_clear_batch_drains_cleanly");

    print_pass("flush_serializer_tb PASS");
    $finish;
  end

endmodule
