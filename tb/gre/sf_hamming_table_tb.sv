// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module sf_hamming_table_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic               clk_i;
  logic               rst_ni;
  logic               start_i;
  logic        [39:0] weight_sf_vec_i;
  logic signed  [4:0] activation_sf_i;
  logic signed  [4:0] bias_i;
  logic               done_o;
  logic        [83:0] dist_table_o;

  logic        [39:0] sample_weight_vec [0:7];
  logic signed  [4:0] sample_activation [0:7];
  logic signed  [4:0] window_bias;
  logic signed  [4:0] second_window_bias;

  sf_hamming_table dut (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .start_i        (start_i),
    .weight_sf_vec_i(weight_sf_vec_i),
    .activation_sf_i(activation_sf_i),
    .bias_i         (bias_i),
    .done_o         (done_o),
    .dist_table_o   (dist_table_o)
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

  function automatic logic signed [6:0] lane_norm_value(
    input logic [39:0]        packed_sf,
    input integer             lane_idx,
    input logic signed [4:0]  activation_sf,
    input logic signed [4:0]  bias_sf
  );
    logic signed [4:0] weight_sf;
    logic signed [6:0] norm_sf;
    begin
      weight_sf = $signed(packed_sf[lane_idx * 5 +: 5]);
      norm_sf   = $signed({{2{weight_sf[4]}}, weight_sf})
                + $signed({{2{activation_sf[4]}}, activation_sf})
                - $signed({{2{bias_sf[4]}}, bias_sf});
      lane_norm_value = norm_sf;
    end
  endfunction

  function automatic logic [1:0] lane_code_from_inputs(
    input logic [39:0]        packed_sf,
    input integer             lane_idx,
    input logic signed [4:0]  activation_sf,
    input logic signed [4:0]  bias_sf
  );
    logic signed [6:0] norm_sf;
    begin
      norm_sf = lane_norm_value(packed_sf, lane_idx, activation_sf, bias_sf);
      lane_code_from_inputs = norm_sf[2:1];
    end
  endfunction

  function automatic logic [15:0] build_vec_from_inputs(
    input logic [39:0]        packed_sf,
    input logic signed [4:0]  activation_sf,
    input logic signed [4:0]  bias_sf
  );
    logic   [15:0] packed_vec;
    integer        lane_idx;
    begin
      packed_vec = 16'd0;
      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        packed_vec[lane_idx * 2 +: 2] = lane_code_from_inputs(
          packed_sf,
          lane_idx,
          activation_sf,
          bias_sf
        );
      end
      build_vec_from_inputs = packed_vec;
    end
  endfunction

  function automatic logic [2:0] hamming_distance_sat(
    input logic [15:0] vec_a,
    input logic [15:0] vec_b
  );
    logic   [3:0] raw_dist;
    integer       lane_idx;
    begin
      raw_dist = 4'd0;
      for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
        if (vec_a[lane_idx * 2 +: 2] != vec_b[lane_idx * 2 +: 2]) begin
          raw_dist = raw_dist + 4'd1;
        end
      end

      if (raw_dist == 4'd8) begin
        hamming_distance_sat = 3'd7;
      end else begin
        hamming_distance_sat = raw_dist[2:0];
      end
    end
  endfunction

  function automatic integer pair_index(
    input integer older_idx,
    input integer newer_idx
  );
    integer row_idx;
    integer pair_idx;
    begin
      pair_idx = 0;
      for (row_idx = 0; row_idx < older_idx; row_idx++) begin
        pair_idx = pair_idx + (7 - row_idx);
      end
      pair_idx = pair_idx + (newer_idx - older_idx - 1);
      pair_index = pair_idx;
    end
  endfunction

  function automatic logic [83:0] expected_table_after_sample(
    input integer            last_sample_idx,
    input logic signed [4:0] bias_sf
  );
    logic   [83:0] expected_table;
    logic   [15:0] older_vec;
    logic   [15:0] newer_vec;
    integer        older_idx;
    integer        newer_idx;
    integer        slot_idx;
    begin
      expected_table = 84'd0;

      for (newer_idx = 1; newer_idx <= last_sample_idx; newer_idx++) begin
        newer_vec = build_vec_from_inputs(
          sample_weight_vec[newer_idx],
          sample_activation[newer_idx],
          bias_sf
        );

        for (older_idx = 0; older_idx < newer_idx; older_idx++) begin
          older_vec = build_vec_from_inputs(
            sample_weight_vec[older_idx],
            sample_activation[older_idx],
            bias_sf
          );
          slot_idx = pair_index(older_idx, newer_idx);
          expected_table[slot_idx * 3 +: 3] = hamming_distance_sat(older_vec, newer_vec);
        end
      end

      expected_table_after_sample = expected_table;
    end
  endfunction

  task automatic clear_window;
    integer sample_idx;
    begin
      for (sample_idx = 0; sample_idx < 8; sample_idx++) begin
        sample_weight_vec[sample_idx] = pack_sf5x8(
          5'sd0, 5'sd0, 5'sd0, 5'sd0,
          5'sd0, 5'sd0, 5'sd0, 5'sd0
        );
        sample_activation[sample_idx] = 5'sd0;
      end
    end
  endtask

  task automatic fill_window(
    input logic [39:0]       packed_sf,
    input logic signed [4:0] activation_sf
  );
    integer sample_idx;
    begin
      for (sample_idx = 0; sample_idx < 8; sample_idx++) begin
        sample_weight_vec[sample_idx] = packed_sf;
        sample_activation[sample_idx] = activation_sf;
      end
    end
  endtask

  task automatic set_sample(
    input integer            sample_idx,
    input logic [39:0]       packed_sf,
    input logic signed [4:0] activation_sf
  );
    begin
      sample_weight_vec[sample_idx] = packed_sf;
      sample_activation[sample_idx] = activation_sf;
    end
  endtask

  task automatic check_outputs(
    input logic        expected_done,
    input logic [83:0] expected_table,
    input string       test_name
  );
    begin
      if (done_o !== expected_done) begin
        print_fail($sformatf(
          "%s done mismatch: expected %0b got %0b",
          test_name,
          expected_done,
          done_o
        ));
        $fatal(1, "%s done mismatch: expected %0b got %0b", test_name, expected_done, done_o);
      end

      if (dist_table_o !== expected_table) begin
        print_fail($sformatf(
          "%s table mismatch: expected 0x%021h got 0x%021h",
          test_name,
          expected_table,
          dist_table_o
        ));
        $fatal(1, "%s table mismatch: expected 0x%021h got 0x%021h", test_name, expected_table, dist_table_o);
      end

      print_pass($sformatf("Check passed: %s", test_name));
    end
  endtask

  task automatic check_norm_range_up_to_sample(
    input integer            last_sample_idx,
    input logic signed [4:0] bias_sf,
    input string             test_name
  );
    logic signed [6:0] norm_sf;
    integer            sample_idx;
    integer            lane_idx;
    begin
      for (sample_idx = 0; sample_idx <= last_sample_idx; sample_idx++) begin
        for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
          norm_sf = lane_norm_value(
            sample_weight_vec[sample_idx],
            lane_idx,
            sample_activation[sample_idx],
            bias_sf
          );

          if ((norm_sf < 7'sd0) || (norm_sf > 7'sd7)) begin
            print_fail($sformatf(
              "%s norm range violation at sample %0d lane %0d: got %0d",
              test_name,
              sample_idx,
              lane_idx,
              norm_sf
            ));
            $fatal(
              1,
              "%s norm range violation at sample %0d lane %0d: got %0d",
              test_name,
              sample_idx,
              lane_idx,
              norm_sf
            );
          end
        end
      end
    end
  endtask

  task automatic apply_reset;
    begin
      print_info("Apply reset");
      rst_ni          = 1'b0;
      start_i         = 1'b0;
      weight_sf_vec_i = pack_sf5x8(
        5'sd0, 5'sd0, 5'sd0, 5'sd0,
        5'sd0, 5'sd0, 5'sd0, 5'sd0
      );
      activation_sf_i = 5'sd0;
      bias_i          = 5'sd0;

      #1;
      check_outputs(1'b0, 84'd0, "reset outputs");

      repeat (2) @(posedge clk_i);
      #1;
      check_outputs(1'b0, 84'd0, "reset hold");

      @(negedge clk_i);
      rst_ni = 1'b1;
      print_pass("Reset released");
    end
  endtask

  task automatic run_window(
    input string             test_name,
    input logic signed [4:0] bias_sf,
    input integer            overlap_start_sample,
    input logic              check_post_hold,
    output logic [83:0]      observed_table
  );
    logic [83:0] expected_table;
    integer      sample_idx;
    begin
      print_info($sformatf("Run window: %s", test_name));

      @(negedge clk_i);
      weight_sf_vec_i = sample_weight_vec[0];
      activation_sf_i = sample_activation[0];
      bias_i          = bias_sf;
      start_i         = 1'b1;

      for (sample_idx = 0; sample_idx < 8; sample_idx++) begin
        @(posedge clk_i);
        #1;

        check_norm_range_up_to_sample(sample_idx, bias_sf, test_name);
        expected_table = expected_table_after_sample(sample_idx, bias_sf);

        if (sample_idx < 7) begin
          check_outputs(
            1'b0,
            expected_table,
            $sformatf("%s sample %0d partial table", test_name, sample_idx)
          );
        end else begin
          check_outputs(
            1'b1,
            expected_table,
            $sformatf("%s sample %0d complete table", test_name, sample_idx)
          );
        end

        start_i = 1'b0;

        if (sample_idx < 7) begin
          @(negedge clk_i);
          weight_sf_vec_i = sample_weight_vec[sample_idx + 1];
          activation_sf_i = sample_activation[sample_idx + 1];
          bias_i          = bias_sf;
          start_i         = ((sample_idx + 1) == overlap_start_sample);
        end
      end

      observed_table = dist_table_o;

      if (check_post_hold) begin
        @(posedge clk_i);
        #1;
        check_outputs(
          1'b0,
          observed_table,
          $sformatf("%s post-completion hold", test_name)
        );
      end

      print_pass($sformatf("%s finished with expected table", test_name));
    end
  endtask

  task automatic run_reset_priority_window(
    input string             test_name,
    input logic signed [4:0] bias_sf
  );
    logic [83:0] expected_table;
    begin
      print_info($sformatf("Run window: %s", test_name));

      @(negedge clk_i);
      weight_sf_vec_i = sample_weight_vec[0];
      activation_sf_i = sample_activation[0];
      bias_i          = bias_sf;
      start_i         = 1'b1;

      @(posedge clk_i);
      #1;
      check_norm_range_up_to_sample(0, bias_sf, test_name);
      check_outputs(1'b0, 84'd0, $sformatf("%s sample 0", test_name));

      start_i = 1'b0;

      @(negedge clk_i);
      weight_sf_vec_i = sample_weight_vec[1];
      activation_sf_i = sample_activation[1];
      bias_i          = bias_sf;

      @(posedge clk_i);
      #1;
      check_norm_range_up_to_sample(1, bias_sf, test_name);
      expected_table = expected_table_after_sample(1, bias_sf);
      check_outputs(1'b0, expected_table, $sformatf("%s sample 1", test_name));

      @(negedge clk_i);
      weight_sf_vec_i = sample_weight_vec[2];
      activation_sf_i = sample_activation[2];
      bias_i          = bias_sf;

      @(posedge clk_i);
      #1;
      check_norm_range_up_to_sample(2, bias_sf, test_name);
      expected_table = expected_table_after_sample(2, bias_sf);
      check_outputs(1'b0, expected_table, $sformatf("%s sample 2", test_name));

      @(negedge clk_i);
      weight_sf_vec_i = sample_weight_vec[3];
      activation_sf_i = sample_activation[3];
      bias_i          = bias_sf;
      start_i         = 1'b1;
      rst_ni          = 1'b0;

      #1;
      check_outputs(1'b0, 84'd0, $sformatf("%s async reset clear", test_name));

      @(posedge clk_i);
      #1;
      check_outputs(1'b0, 84'd0, $sformatf("%s reset wins over start", test_name));

      @(negedge clk_i);
      start_i = 1'b0;
      rst_ni  = 1'b1;
      bias_i  = 5'sd0;

      @(posedge clk_i);
      #1;
      check_outputs(1'b0, 84'd0, $sformatf("%s idle after reset release", test_name));

      print_pass($sformatf("%s reset priority checks passed", test_name));
    end
  endtask

  initial begin
    logic [83:0] observed_table;

    apply_reset();

    window_bias = 5'sd3;
    clear_window();
    set_sample(0, pack_sf5x8(-5'sd1, 5'sd0, 5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6), 5'sd4);
    set_sample(1, pack_sf5x8(5'sd0, 5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6, -5'sd1), 5'sd4);
    set_sample(2, pack_sf5x8(5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6, -5'sd1, 5'sd0), 5'sd4);
    set_sample(3, pack_sf5x8(5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6, -5'sd1, 5'sd0, 5'sd1), 5'sd4);
    set_sample(4, pack_sf5x8(5'sd3, 5'sd4, 5'sd5, 5'sd6, -5'sd1, 5'sd0, 5'sd1, 5'sd2), 5'sd4);
    set_sample(5, pack_sf5x8(5'sd4, 5'sd5, 5'sd6, -5'sd1, 5'sd0, 5'sd1, 5'sd2, 5'sd3), 5'sd4);
    set_sample(6, pack_sf5x8(5'sd5, 5'sd6, -5'sd1, 5'sd0, 5'sd1, 5'sd2, 5'sd3, 5'sd4), 5'sd4);
    set_sample(7, pack_sf5x8(5'sd6, -5'sd1, 5'sd0, 5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5), 5'sd4);
    run_window("arithmetic and packing window", window_bias, -1, 1'b1, observed_table);

    window_bias = 5'sd0;
    clear_window();
    set_sample(0, pack_sf5x8(5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0), 5'sd0);
    set_sample(1, pack_sf5x8(5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2), 5'sd0);
    set_sample(2, pack_sf5x8(5'sd4, 5'sd4, 5'sd4, 5'sd4, 5'sd4, 5'sd4, 5'sd4, 5'sd4), 5'sd0);
    set_sample(3, pack_sf5x8(5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6), 5'sd0);
    set_sample(4, pack_sf5x8(5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0), 5'sd0);
    set_sample(5, pack_sf5x8(5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2), 5'sd0);
    set_sample(6, pack_sf5x8(5'sd4, 5'sd4, 5'sd4, 5'sd4, 5'sd4, 5'sd4, 5'sd4, 5'sd4), 5'sd0);
    set_sample(7, pack_sf5x8(5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6), 5'sd0);
    run_window("distance saturation window", window_bias, -1, 1'b1, observed_table);

    window_bias = 5'sd1;
    clear_window();
    set_sample(0, pack_sf5x8(5'sd0, 5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6, 5'sd7), 5'sd1);
    set_sample(1, pack_sf5x8(5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6, 5'sd7, 5'sd0), 5'sd1);
    set_sample(2, pack_sf5x8(5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6, 5'sd7, 5'sd0, 5'sd1), 5'sd1);
    set_sample(3, pack_sf5x8(5'sd3, 5'sd4, 5'sd5, 5'sd6, 5'sd7, 5'sd0, 5'sd1, 5'sd2), 5'sd1);
    set_sample(4, pack_sf5x8(5'sd4, 5'sd5, 5'sd6, 5'sd7, 5'sd0, 5'sd1, 5'sd2, 5'sd3), 5'sd1);
    set_sample(5, pack_sf5x8(5'sd5, 5'sd6, 5'sd7, 5'sd0, 5'sd1, 5'sd2, 5'sd3, 5'sd4), 5'sd1);
    set_sample(6, pack_sf5x8(5'sd6, 5'sd7, 5'sd0, 5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5), 5'sd1);
    set_sample(7, pack_sf5x8(5'sd7, 5'sd0, 5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6), 5'sd1);
    run_window("ignore overlapping start window", window_bias, 5, 1'b1, observed_table);

    second_window_bias = 5'sd2;
    clear_window();
    set_sample(0, pack_sf5x8(5'sd0, 5'sd0, 5'sd2, 5'sd2, 5'sd4, 5'sd4, 5'sd6, 5'sd6), 5'sd2);
    set_sample(1, pack_sf5x8(5'sd1, 5'sd1, 5'sd3, 5'sd3, 5'sd5, 5'sd5, 5'sd7, 5'sd7), 5'sd2);
    set_sample(2, pack_sf5x8(5'sd2, 5'sd4, 5'sd6, 5'sd0, 5'sd2, 5'sd4, 5'sd6, 5'sd0), 5'sd2);
    set_sample(3, pack_sf5x8(5'sd7, 5'sd5, 5'sd3, 5'sd1, 5'sd7, 5'sd5, 5'sd3, 5'sd1), 5'sd2);
    set_sample(4, pack_sf5x8(5'sd0, 5'sd2, 5'sd4, 5'sd6, 5'sd1, 5'sd3, 5'sd5, 5'sd7), 5'sd2);
    set_sample(5, pack_sf5x8(5'sd7, 5'sd5, 5'sd3, 5'sd1, 5'sd6, 5'sd4, 5'sd2, 5'sd0), 5'sd2);
    set_sample(6, pack_sf5x8(5'sd3, 5'sd3, 5'sd3, 5'sd3, 5'sd5, 5'sd5, 5'sd5, 5'sd5), 5'sd2);
    set_sample(7, pack_sf5x8(5'sd4, 5'sd4, 5'sd4, 5'sd4, 5'sd6, 5'sd6, 5'sd6, 5'sd6), 5'sd2);
    run_window("second window clears previous table", second_window_bias, -1, 1'b1, observed_table);

    window_bias = 5'sd2;
    clear_window();
    fill_window(pack_sf5x8(5'sd0, 5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6, 5'sd7), 5'sd2);
    set_sample(1, pack_sf5x8(5'sd7, 5'sd6, 5'sd5, 5'sd4, 5'sd3, 5'sd2, 5'sd1, 5'sd0), 5'sd2);
    set_sample(2, pack_sf5x8(5'sd1, 5'sd3, 5'sd5, 5'sd7, 5'sd0, 5'sd2, 5'sd4, 5'sd6), 5'sd2);
    set_sample(3, pack_sf5x8(5'sd6, 5'sd4, 5'sd2, 5'sd0, 5'sd7, 5'sd5, 5'sd3, 5'sd1), 5'sd2);
    run_reset_priority_window("reset priority window", window_bias);

    print_pass("sf_hamming_table_tb PASS");
    $finish;
  end

endmodule
