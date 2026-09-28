// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module sf_min_tracker_tb;

  localparam string C_INFO  = "\033[1;34m";
  localparam string C_PASS  = "\033[1;32m";
  localparam string C_FAIL  = "\033[1;31m";
  localparam string C_RESET = "\033[0m";

  logic               clk_i;
  logic               rst_ni;
  logic               start_i;
  logic        [39:0] weight_sf_vec_i;
  logic signed  [4:0] activation_sf_i;
  logic               done_o;
  logic signed  [4:0] min_sf_o;

  logic        [39:0] sample_weight_vec [0:7];
  logic signed  [4:0] sample_activation [0:7];

  sf_min_tracker dut (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .start_i        (start_i),
    .weight_sf_vec_i(weight_sf_vec_i),
    .activation_sf_i(activation_sf_i),
    .done_o         (done_o),
    .min_sf_o       (min_sf_o)
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

  function automatic logic signed [4:0] min_weight_from_vec(input logic [39:0] packed_sf);
    logic signed [4:0] current_min;
    logic signed [4:0] lane_value;
    integer            lane_idx;
    begin
      current_min = $signed(packed_sf[4:0]);
      for (lane_idx = 1; lane_idx < 8; lane_idx++) begin
        lane_value = $signed(packed_sf[lane_idx * 5 +: 5]);
        if (lane_value < current_min) begin
          current_min = lane_value;
        end
      end
      min_weight_from_vec = current_min;
    end
  endfunction

  function automatic logic signed [4:0] candidate_from_inputs(
    input logic [39:0]       packed_sf,
    input logic signed [4:0] activation_sf
  );
    logic signed [4:0] weight_min;
    logic signed [5:0] candidate_ext;
    begin
      weight_min    = min_weight_from_vec(packed_sf);
      candidate_ext = $signed({weight_min[4], weight_min}) + $signed({activation_sf[4], activation_sf});
      candidate_from_inputs = candidate_ext[4:0];
    end
  endfunction

  function automatic logic signed [4:0] expected_window_min;
    logic signed [4:0] current_min;
    logic signed [4:0] candidate;
    integer            sample_idx;
    begin
      current_min = candidate_from_inputs(sample_weight_vec[0], sample_activation[0]);
      for (sample_idx = 1; sample_idx < 8; sample_idx++) begin
        candidate = candidate_from_inputs(sample_weight_vec[sample_idx], sample_activation[sample_idx]);
        if (candidate < current_min) begin
          current_min = candidate;
        end
      end
      expected_window_min = current_min;
    end
  endfunction

  task automatic clear_window;
    integer sample_idx;
    begin
      for (sample_idx = 0; sample_idx < 8; sample_idx++) begin
        sample_weight_vec[sample_idx] = pack_sf5x8(5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0);
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

  task automatic check_idle_outputs(
    input logic              expected_done,
    input logic signed [4:0] expected_min,
    input string             test_name
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

      if (min_sf_o !== expected_min) begin
        print_fail($sformatf(
          "%s min mismatch: expected %0d got %0d",
          test_name,
          expected_min,
          min_sf_o
        ));
        $fatal(1, "%s min mismatch: expected %0d got %0d", test_name, expected_min, min_sf_o);
      end

      print_pass($sformatf("Check passed: %s", test_name));
    end
  endtask

  task automatic run_window(
    input string  test_name,
    input integer overlap_start_sample,
    output logic signed [4:0] observed_min
  );
    logic signed [4:0] expected_min;
    integer            sample_idx;
    begin
      expected_min = expected_window_min();
      print_info($sformatf("Run window: %s", test_name));

      @(negedge clk_i);
      weight_sf_vec_i = sample_weight_vec[0];
      activation_sf_i = sample_activation[0];
      start_i         = 1'b1;

      for (sample_idx = 0; sample_idx < 8; sample_idx++) begin
        @(posedge clk_i);
        #1;

        if (sample_idx < 7) begin
          if (done_o !== 1'b0) begin
            print_fail($sformatf("%s asserted done early at sample %0d", test_name, sample_idx));
            $fatal(1, "%s asserted done early at sample %0d", test_name, sample_idx);
          end
        end else begin
          if (done_o !== 1'b1) begin
            print_fail($sformatf("%s did not assert done on sample 7", test_name));
            $fatal(1, "%s did not assert done on sample 7", test_name);
          end

          if (min_sf_o !== expected_min) begin
            print_fail($sformatf(
              "%s final min mismatch: expected %0d got %0d",
              test_name,
              expected_min,
              min_sf_o
            ));
            $fatal(1, "%s final min mismatch: expected %0d got %0d", test_name, expected_min, min_sf_o);
          end
        end

        start_i = 1'b0;

        if (sample_idx < 7) begin
          @(negedge clk_i);
          weight_sf_vec_i = sample_weight_vec[sample_idx + 1];
          activation_sf_i = sample_activation[sample_idx + 1];
          start_i         = ((sample_idx + 1) == overlap_start_sample);
        end
      end

      observed_min = min_sf_o;
      print_pass($sformatf("%s final window result matched %0d", test_name, expected_min));
    end
  endtask

  task automatic apply_reset;
    begin
      print_info("Apply reset");
      rst_ni         = 1'b0;
      start_i        = 1'b0;
      weight_sf_vec_i = pack_sf5x8(5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0, 5'sd0);
      activation_sf_i = 5'sd0;

      repeat (2) @(posedge clk_i);
      #1;
      check_idle_outputs(1'b0, 5'sd0, "reset outputs");

      @(negedge clk_i);
      rst_ni = 1'b1;
      print_pass("Reset released");
    end
  endtask

  initial begin
    logic signed [4:0] observed_min;
    logic signed [4:0] held_min;

    apply_reset();

    clear_window();
    fill_window(pack_sf5x8(5'sd4, 5'sd5, 5'sd6, 5'sd7, 5'sd8, 5'sd9, 5'sd10, 5'sd11), 5'sd1);
    set_sample(0, pack_sf5x8(-5'sd6, 5'sd3, 5'sd4, 5'sd5, 5'sd6, 5'sd7, 5'sd8, 5'sd9), 5'sd1);
    run_window("minimum on first sample", -1, observed_min);

    @(posedge clk_i);
    #1;
    held_min = observed_min;
    check_idle_outputs(1'b0, held_min, "hold result after first window");

    clear_window();
    fill_window(pack_sf5x8(5'sd3, 5'sd4, 5'sd5, 5'sd6, 5'sd7, 5'sd8, 5'sd9, 5'sd10), 5'sd2);
    set_sample(3, pack_sf5x8(5'sd5, 5'sd4, 5'sd3, 5'sd2, 5'sd1, 5'sd0, -5'sd7, 5'sd6), -5'sd2);
    run_window("minimum in middle with negative activation", -1, observed_min);

    @(posedge clk_i);
    #1;
    held_min = observed_min;
    check_idle_outputs(1'b0, held_min, "hold result after middle-min window");

    clear_window();
    fill_window(pack_sf5x8(5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6, 5'sd6), 5'sd0);
    set_sample(7, pack_sf5x8(5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2, 5'sd2), -5'sd5);
    run_window("minimum on last sample with equal weights", -1, observed_min);

    @(posedge clk_i);
    #1;
    held_min = observed_min;
    check_idle_outputs(1'b0, held_min, "hold result after last-sample window");

    clear_window();
    fill_window(pack_sf5x8(5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6, 5'sd7, 5'sd8), 5'sd1);
    set_sample(6, pack_sf5x8(-5'sd8, 5'sd0, 5'sd1, 5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6), 5'sd0);
    run_window("ignore overlapping start while running", 5, observed_min);

    @(posedge clk_i);
    #1;
    held_min = observed_min;
    check_idle_outputs(1'b0, held_min, "hold result after ignored overlap start");

    clear_window();
    fill_window(pack_sf5x8(5'sd2, 5'sd3, 5'sd4, 5'sd5, 5'sd6, 5'sd7, 5'sd8, 5'sd9), 5'sd1);
    set_sample(3, pack_sf5x8(-5'sd9, -5'sd8, -5'sd7, -5'sd6, -5'sd5, -5'sd4, -5'sd3, -5'sd2), -5'sd1);
    run_window("negative final result", -1, observed_min);

    @(posedge clk_i);
    #1;
    check_idle_outputs(1'b0, observed_min, "hold negative final result");

    print_pass("sf_min_tracker_tb PASS");
    $finish;
  end

endmodule
