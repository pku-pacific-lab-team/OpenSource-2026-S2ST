// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module pe_array_group (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        clear_i,

  input  logic        weight_buf_en_i,
  input  logic        weight_buf_wen_i,
  input  logic [10:0] weight_buf_addr_i,
  input  logic [31:0] weight_buf_wdata_i,
  output logic [31:0] weight_buf_rdata_o,

  input  logic        weight_sf_buf_en_i,
  input  logic        weight_sf_buf_wen_i,
  input  logic  [7:0] weight_sf_buf_addr_i,
  input  logic [39:0] weight_sf_buf_wdata_i,
  output logic [39:0] weight_sf_buf_rdata_o,

  input  logic        activation_buf_en_i,
  input  logic        activation_buf_wen_i,
  input  logic [10:0] activation_buf_addr_i,
  input  logic [31:0] activation_buf_wdata_i,
  output logic [31:0] activation_buf_rdata_o,

  input  logic        activation_sf_buf_en_i,
  input  logic        activation_sf_buf_wen_i,
  input  logic [10:0] activation_sf_buf_addr_i,
  input  logic  [4:0] activation_sf_buf_wdata_i,
  output logic  [4:0] activation_sf_buf_rdata_o,

  input  logic        mask_buf_en_i,
  input  logic        mask_buf_wen_i,
  input  logic [10:0] mask_buf_addr_i,
  input  logic  [7:0] mask_buf_wdata_i,
  output logic  [7:0] mask_buf_rdata_o,

  input  logic        weight_load_start_i,
  input  logic  [8:0] weight_load_base_addr_i,
  input  logic        compute_start_i,
  input  logic        fusion_preload_start_i,
  input  logic        fusion_mode_en_i,
  input  logic  [8:0] activation_addr_i,
  input  logic  [5:0] weight_sf_addr_i,
  input  logic [15:0] row_weight_rsel_i,
  input  logic  [7:0] fp_flush_flag_i,
  input  logic        writeback_enable_i,
  input  logic  [8:0] writeback_addr_i,
  input  logic  [7:0] threshold_i,
  input  logic        use_reorder_seq_i,
  input  logic        sf_reorder_start_i,
  input  logic  [2:0] sf_reorder_threshold_i,
  input  logic  [2:0] sf_reorder_max_iter_i,

  output logic        busy_o,
  output logic        weight_load_done_o,
  output logic        compute_done_o
);

  localparam int NUM_BANKS = 4;

  typedef enum logic [2:0] {
    REORDER_IDLE         = 3'd0,
    REORDER_ISSUE_FIRST  = 3'd1,
    REORDER_STREAM_FIRST = 3'd2,
    REORDER_STREAM_SECOND= 3'd3,
    REORDER_WAIT_DONE    = 3'd4,
    REORDER_WRITE_SEQ    = 3'd5
  } reorder_state_t;

  logic [1:0] weight_bank_sel;
  logic [8:0] weight_local_addr;
  logic [1:0] weight_sf_bank_sel;
  logic [5:0] weight_sf_local_addr;
  logic [1:0] activation_bank_sel;
  logic [8:0] activation_local_addr;
  logic [1:0] activation_sf_bank_sel;
  logic [8:0] activation_sf_local_addr;
  logic [1:0] mask_bank_sel;
  logic [8:0] mask_local_addr;

  logic [1:0] weight_read_bank_q;
  logic [1:0] weight_sf_read_bank_q;
  logic [1:0] activation_read_bank_q;
  logic [1:0] activation_sf_read_bank_q;
  logic [1:0] mask_read_bank_q;

  logic [NUM_BANKS-1:0] child_weight_buf_en;
  logic [NUM_BANKS-1:0] child_weight_buf_wen;
  logic [NUM_BANKS-1:0] child_weight_sf_buf_en;
  logic [NUM_BANKS-1:0] child_weight_sf_buf_wen;
  logic [NUM_BANKS-1:0] child_activation_buf_en;
  logic [NUM_BANKS-1:0] child_activation_buf_wen;
  logic [NUM_BANKS-1:0] child_activation_sf_buf_en;
  logic [NUM_BANKS-1:0] child_activation_sf_buf_wen;
  logic [NUM_BANKS-1:0] child_mask_buf_en;
  logic [NUM_BANKS-1:0] child_mask_buf_wen;
  logic [NUM_BANKS-1:0] child_reorder_seq_we;

  logic [8:0] child_weight_buf_addr [0:NUM_BANKS-1];
  logic [5:0] child_weight_sf_buf_addr [0:NUM_BANKS-1];
  logic [8:0] child_activation_buf_addr [0:NUM_BANKS-1];
  logic [8:0] child_activation_sf_buf_addr [0:NUM_BANKS-1];
  logic [8:0] child_mask_buf_addr [0:NUM_BANKS-1];

  logic [31:0] child_weight_buf_wdata [0:NUM_BANKS-1];
  logic [39:0] child_weight_sf_buf_wdata [0:NUM_BANKS-1];
  logic [31:0] child_activation_buf_wdata [0:NUM_BANKS-1];
  logic  [4:0] child_activation_sf_buf_wdata [0:NUM_BANKS-1];
  logic  [7:0] child_mask_buf_wdata [0:NUM_BANKS-1];
  logic [23:0] child_reorder_seq [0:NUM_BANKS-1];

  logic [NUM_BANKS-1:0] child_busy;
  logic [NUM_BANKS-1:0] child_weight_load_done;
  logic [NUM_BANKS-1:0] child_compute_done;

  logic [31:0] child_weight_rdata [0:NUM_BANKS-1];
  logic [39:0] child_weight_sf_rdata [0:NUM_BANKS-1];
  logic [31:0] child_activation_rdata [0:NUM_BANKS-1];
  logic  [4:0] child_activation_sf_rdata [0:NUM_BANKS-1];
  logic  [7:0] child_mask_rdata [0:NUM_BANKS-1];

  logic           child_weight_load_start;
  logic           child_compute_start;
  logic           child_fusion_preload_start;
  logic           child_reorder_ptr_reset;

  reorder_state_t reorder_state_d;
  reorder_state_t reorder_state_q;
  logic     [1:0] reorder_bank_d;
  logic     [1:0] reorder_bank_q;
  logic     [2:0] reorder_sample_d;
  logic     [2:0] reorder_sample_q;
  logic     [8:0] reorder_activation_base_addr_d;
  logic     [8:0] reorder_activation_base_addr_q;
  logic     [5:0] reorder_weight_sf_base_addr_d;
  logic     [5:0] reorder_weight_sf_base_addr_q;
  logic           reorder_accept;
  logic           reorder_busy;
  logic           reorder_start;
  logic           reorder_issue_valid;
  logic     [2:0] reorder_issue_sample;
  logic     [8:0] reorder_activation_sf_addr;
  logic     [5:0] reorder_weight_sf_addr;
  logic    [39:0] reorder_weight_sf_vec;
  logic signed [4:0] reorder_activation_sf;
  logic           reorder_done;
  logic    [23:0] reorder_seq;

  assign weight_bank_sel          = weight_buf_addr_i[10:9];
  assign weight_local_addr        = weight_buf_addr_i[8:0];
  assign weight_sf_bank_sel       = weight_sf_buf_addr_i[7:6];
  assign weight_sf_local_addr     = weight_sf_buf_addr_i[5:0];
  assign activation_bank_sel      = activation_buf_addr_i[10:9];
  assign activation_local_addr    = activation_buf_addr_i[8:0];
  assign activation_sf_bank_sel   = activation_sf_buf_addr_i[10:9];
  assign activation_sf_local_addr = activation_sf_buf_addr_i[8:0];
  assign mask_bank_sel            = mask_buf_addr_i[10:9];
  assign mask_local_addr          = mask_buf_addr_i[8:0];

  assign reorder_accept            = sf_reorder_start_i && (reorder_state_q == REORDER_IDLE) && !(|child_busy);
  assign reorder_busy              = reorder_accept || (reorder_state_q != REORDER_IDLE);
  assign child_weight_load_start   = weight_load_start_i && !reorder_busy;
  assign child_compute_start       = compute_start_i && !reorder_busy;
  assign child_fusion_preload_start= fusion_preload_start_i && !reorder_busy;
  assign child_reorder_ptr_reset   = reorder_accept;

  always_comb begin
    reorder_start       = 1'b0;
    reorder_issue_valid = 1'b0;
    reorder_issue_sample= 3'd0;
    reorder_state_d     = reorder_state_q;
    reorder_bank_d      = reorder_bank_q;
    reorder_sample_d    = reorder_sample_q;
    reorder_activation_base_addr_d = reorder_activation_base_addr_q;
    reorder_weight_sf_base_addr_d  = reorder_weight_sf_base_addr_q;
    child_reorder_seq_we = '0;

    for (int bank_idx = 0; bank_idx < NUM_BANKS; bank_idx++) begin
      child_reorder_seq[bank_idx] = reorder_seq;
    end

    unique case (reorder_state_q)
      REORDER_IDLE: begin
        reorder_bank_d   = 2'd0;
        reorder_sample_d = 3'd0;
        if (reorder_accept) begin
          reorder_activation_base_addr_d = activation_addr_i;
          reorder_weight_sf_base_addr_d  = weight_sf_addr_i;
          reorder_state_d = REORDER_ISSUE_FIRST;
        end
      end

      REORDER_ISSUE_FIRST: begin
        reorder_issue_valid = 1'b1;
        reorder_issue_sample= 3'd0;
        reorder_sample_d    = 3'd0;
        reorder_state_d     = REORDER_STREAM_FIRST;
      end

      REORDER_STREAM_FIRST: begin
        reorder_start = (reorder_sample_q == 3'd0);
        reorder_issue_valid = 1'b1;

        if (reorder_sample_q == 3'd7) begin
          reorder_issue_sample = 3'd0;
          reorder_sample_d     = 3'd0;
          reorder_state_d      = REORDER_STREAM_SECOND;
        end else begin
          reorder_issue_sample = reorder_sample_q + 3'd1;
          reorder_sample_d     = reorder_sample_q + 3'd1;
        end
      end

      REORDER_STREAM_SECOND: begin
        if (reorder_sample_q == 3'd7) begin
          reorder_sample_d = 3'd0;
          reorder_state_d  = REORDER_WAIT_DONE;
        end else begin
          reorder_issue_valid = 1'b1;
          reorder_issue_sample= reorder_sample_q + 3'd1;
          reorder_sample_d    = reorder_sample_q + 3'd1;
        end
      end

      REORDER_WAIT_DONE: begin
        if (reorder_done) begin
          reorder_state_d = REORDER_WRITE_SEQ;
        end
      end

      REORDER_WRITE_SEQ: begin
        unique case (reorder_bank_q)
          2'd0: child_reorder_seq_we[0] = 1'b1;
          2'd1: child_reorder_seq_we[1] = 1'b1;
          2'd2: child_reorder_seq_we[2] = 1'b1;
          default: child_reorder_seq_we[3] = 1'b1;
        endcase

        reorder_sample_d = 3'd0;
        if (reorder_bank_q == 2'd3) begin
          reorder_bank_d  = 2'd0;
          reorder_state_d = REORDER_IDLE;
        end else begin
          reorder_bank_d  = reorder_bank_q + 2'd1;
          reorder_state_d = REORDER_ISSUE_FIRST;
        end
      end

      default: begin
        reorder_bank_d   = 2'd0;
        reorder_sample_d = 3'd0;
        reorder_state_d  = REORDER_IDLE;
      end
    endcase
  end

  assign reorder_activation_sf_addr = reorder_activation_base_addr_q + {{6{1'b0}}, reorder_issue_sample};
  assign reorder_weight_sf_addr     = reorder_weight_sf_base_addr_q + {{3{1'b0}}, reorder_issue_sample};

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      weight_read_bank_q        <= 2'd0;
      weight_sf_read_bank_q     <= 2'd0;
      activation_read_bank_q    <= 2'd0;
      activation_sf_read_bank_q <= 2'd0;
      mask_read_bank_q          <= 2'd0;
    end else if (clear_i) begin
      weight_read_bank_q        <= 2'd0;
      weight_sf_read_bank_q     <= 2'd0;
      activation_read_bank_q    <= 2'd0;
      activation_sf_read_bank_q <= 2'd0;
      mask_read_bank_q          <= 2'd0;
    end else begin
      if (weight_buf_en_i && !weight_buf_wen_i) begin
        weight_read_bank_q <= weight_bank_sel;
      end
      if (weight_sf_buf_en_i && !weight_sf_buf_wen_i) begin
        weight_sf_read_bank_q <= weight_sf_bank_sel;
      end
      if (activation_buf_en_i && !activation_buf_wen_i) begin
        activation_read_bank_q <= activation_bank_sel;
      end
      if (activation_sf_buf_en_i && !activation_sf_buf_wen_i) begin
        activation_sf_read_bank_q <= activation_sf_bank_sel;
      end
      if (mask_buf_en_i && !mask_buf_wen_i) begin
        mask_read_bank_q <= mask_bank_sel;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      reorder_state_q  <= REORDER_IDLE;
      reorder_bank_q   <= 2'd0;
      reorder_sample_q <= 3'd0;
      reorder_activation_base_addr_q <= '0;
      reorder_weight_sf_base_addr_q  <= '0;
    end else begin
      reorder_state_q  <= reorder_state_d;
      reorder_bank_q   <= reorder_bank_d;
      reorder_sample_q <= reorder_sample_d;
      reorder_activation_base_addr_q <= reorder_activation_base_addr_d;
      reorder_weight_sf_base_addr_q  <= reorder_weight_sf_base_addr_d;
    end
  end

  always_comb begin
    unique case (reorder_bank_q)
      2'd0: begin
        reorder_weight_sf_vec = child_weight_sf_rdata[0];
        reorder_activation_sf = $signed(child_activation_sf_rdata[0]);
      end
      2'd1: begin
        reorder_weight_sf_vec = child_weight_sf_rdata[1];
        reorder_activation_sf = $signed(child_activation_sf_rdata[1]);
      end
      2'd2: begin
        reorder_weight_sf_vec = child_weight_sf_rdata[2];
        reorder_activation_sf = $signed(child_activation_sf_rdata[2]);
      end
      default: begin
        reorder_weight_sf_vec = child_weight_sf_rdata[3];
        reorder_activation_sf = $signed(child_activation_sf_rdata[3]);
      end
    endcase
  end

  genvar bank_idx;
  generate
    for (bank_idx = 0; bank_idx < NUM_BANKS; bank_idx++) begin : gen_banks
      localparam logic [1:0] BANK_SEL = bank_idx[1:0];

      assign child_weight_buf_en[bank_idx]            = weight_buf_en_i && (weight_bank_sel == BANK_SEL);
      assign child_weight_buf_wen[bank_idx]           = weight_buf_wen_i;
      assign child_weight_buf_addr[bank_idx]          = weight_local_addr;
      assign child_weight_buf_wdata[bank_idx]         = weight_buf_wdata_i;

      assign child_activation_buf_en[bank_idx]        = activation_buf_en_i && (activation_bank_sel == BANK_SEL);
      assign child_activation_buf_wen[bank_idx]       = activation_buf_wen_i;
      assign child_activation_buf_addr[bank_idx]      = activation_local_addr;
      assign child_activation_buf_wdata[bank_idx]     = activation_buf_wdata_i;

      assign child_mask_buf_en[bank_idx]              = mask_buf_en_i && (mask_bank_sel == BANK_SEL);
      assign child_mask_buf_wen[bank_idx]             = mask_buf_wen_i;
      assign child_mask_buf_addr[bank_idx]            = mask_local_addr;
      assign child_mask_buf_wdata[bank_idx]           = mask_buf_wdata_i;

      assign child_weight_sf_buf_wdata[bank_idx]      = weight_sf_buf_wdata_i;
      assign child_activation_sf_buf_wdata[bank_idx]  = activation_sf_buf_wdata_i;

      if (1) begin : gen_reorder_mux
        logic reorder_bank_active;

        assign reorder_bank_active = (reorder_state_q != REORDER_IDLE) && (reorder_bank_q == BANK_SEL);

        assign child_weight_sf_buf_en[bank_idx] = (reorder_bank_active && reorder_issue_valid)
          ? 1'b1
          : (weight_sf_buf_en_i && (weight_sf_bank_sel == BANK_SEL));
        assign child_weight_sf_buf_wen[bank_idx] = (reorder_bank_active && reorder_issue_valid)
          ? 1'b0
          : weight_sf_buf_wen_i;
        assign child_weight_sf_buf_addr[bank_idx] = (reorder_bank_active && reorder_issue_valid)
          ? reorder_weight_sf_addr
          : weight_sf_local_addr;

        assign child_activation_sf_buf_en[bank_idx] = (reorder_bank_active && reorder_issue_valid)
          ? 1'b1
          : (activation_sf_buf_en_i && (activation_sf_bank_sel == BANK_SEL));
        assign child_activation_sf_buf_wen[bank_idx] = (reorder_bank_active && reorder_issue_valid)
          ? 1'b0
          : activation_sf_buf_wen_i;
        assign child_activation_sf_buf_addr[bank_idx] = (reorder_bank_active && reorder_issue_valid)
          ? reorder_activation_sf_addr
          : activation_sf_local_addr;
      end

      pe_array_buffered_hybrid_top array_u (
        .clk_i                    (clk_i),
        .rst_ni                   (rst_ni),
        .clear_i                  (clear_i),
        .weight_buf_en_i          (child_weight_buf_en[bank_idx]),
        .weight_buf_wen_i         (child_weight_buf_wen[bank_idx]),
        .weight_buf_addr_i        (child_weight_buf_addr[bank_idx]),
        .weight_buf_wdata_i       (child_weight_buf_wdata[bank_idx]),
        .weight_buf_rdata_o       (child_weight_rdata[bank_idx]),
        .weight_sf_buf_en_i       (child_weight_sf_buf_en[bank_idx]),
        .weight_sf_buf_wen_i      (child_weight_sf_buf_wen[bank_idx]),
        .weight_sf_buf_addr_i     (child_weight_sf_buf_addr[bank_idx]),
        .weight_sf_buf_wdata_i    (child_weight_sf_buf_wdata[bank_idx]),
        .weight_sf_buf_rdata_o    (child_weight_sf_rdata[bank_idx]),
        .activation_buf_en_i      (child_activation_buf_en[bank_idx]),
        .activation_buf_wen_i     (child_activation_buf_wen[bank_idx]),
        .activation_buf_addr_i    (child_activation_buf_addr[bank_idx]),
        .activation_buf_wdata_i   (child_activation_buf_wdata[bank_idx]),
        .activation_buf_rdata_o   (child_activation_rdata[bank_idx]),
        .activation_sf_buf_en_i   (child_activation_sf_buf_en[bank_idx]),
        .activation_sf_buf_wen_i  (child_activation_sf_buf_wen[bank_idx]),
        .activation_sf_buf_addr_i (child_activation_sf_buf_addr[bank_idx]),
        .activation_sf_buf_wdata_i(child_activation_sf_buf_wdata[bank_idx]),
        .activation_sf_buf_rdata_o(child_activation_sf_rdata[bank_idx]),
        .mask_buf_en_i            (child_mask_buf_en[bank_idx]),
        .mask_buf_wen_i           (child_mask_buf_wen[bank_idx]),
        .mask_buf_addr_i          (child_mask_buf_addr[bank_idx]),
        .mask_buf_wdata_i         (child_mask_buf_wdata[bank_idx]),
        .mask_buf_rdata_o         (child_mask_rdata[bank_idx]),
        .weight_load_start_i      (child_weight_load_start),
        .weight_load_base_addr_i  (weight_load_base_addr_i),
        .compute_start_i          (child_compute_start),
        .fusion_preload_start_i   (child_fusion_preload_start),
        .fusion_mode_en_i         (fusion_mode_en_i),
        .activation_addr_i        (activation_addr_i),
        .weight_sf_addr_i         (weight_sf_addr_i),
        .row_weight_rsel_i        (row_weight_rsel_i),
        .fp_flush_flag_i          (fp_flush_flag_i),
        .writeback_enable_i       (writeback_enable_i),
        .writeback_addr_i         (writeback_addr_i),
        .threshold_i              (threshold_i),
        .use_reorder_seq_i        (use_reorder_seq_i),
        .reorder_ptr_reset_i      (child_reorder_ptr_reset),
        .reorder_seq_we_i         (child_reorder_seq_we[bank_idx]),
        .reorder_seq_i            (child_reorder_seq[bank_idx]),
        .busy_o                   (child_busy[bank_idx]),
        .weight_load_done_o       (child_weight_load_done[bank_idx]),
        .compute_done_o           (child_compute_done[bank_idx])
      );
    end
  endgenerate

  sf_reorder_top sf_reorder_u (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .start_i        (reorder_start),
    .weight_sf_vec_i(reorder_weight_sf_vec),
    .activation_sf_i(reorder_activation_sf),
    .threshold_i    (sf_reorder_threshold_i),
    .max_iter_i     (sf_reorder_max_iter_i),
    .done_o         (reorder_done),
    .seq_o          (reorder_seq)
  );

  always_comb begin
    unique case (weight_read_bank_q)
      2'd0: weight_buf_rdata_o = child_weight_rdata[0];
      2'd1: weight_buf_rdata_o = child_weight_rdata[1];
      2'd2: weight_buf_rdata_o = child_weight_rdata[2];
      default: weight_buf_rdata_o = child_weight_rdata[3];
    endcase
  end

  always_comb begin
    unique case (weight_sf_read_bank_q)
      2'd0: weight_sf_buf_rdata_o = child_weight_sf_rdata[0];
      2'd1: weight_sf_buf_rdata_o = child_weight_sf_rdata[1];
      2'd2: weight_sf_buf_rdata_o = child_weight_sf_rdata[2];
      default: weight_sf_buf_rdata_o = child_weight_sf_rdata[3];
    endcase
  end

  always_comb begin
    unique case (activation_read_bank_q)
      2'd0: activation_buf_rdata_o = child_activation_rdata[0];
      2'd1: activation_buf_rdata_o = child_activation_rdata[1];
      2'd2: activation_buf_rdata_o = child_activation_rdata[2];
      default: activation_buf_rdata_o = child_activation_rdata[3];
    endcase
  end

  always_comb begin
    unique case (activation_sf_read_bank_q)
      2'd0: activation_sf_buf_rdata_o = child_activation_sf_rdata[0];
      2'd1: activation_sf_buf_rdata_o = child_activation_sf_rdata[1];
      2'd2: activation_sf_buf_rdata_o = child_activation_sf_rdata[2];
      default: activation_sf_buf_rdata_o = child_activation_sf_rdata[3];
    endcase
  end

  always_comb begin
    unique case (mask_read_bank_q)
      2'd0: mask_buf_rdata_o = child_mask_rdata[0];
      2'd1: mask_buf_rdata_o = child_mask_rdata[1];
      2'd2: mask_buf_rdata_o = child_mask_rdata[2];
      default: mask_buf_rdata_o = child_mask_rdata[3];
    endcase
  end

  assign busy_o             = (|child_busy) || reorder_busy;
  assign weight_load_done_o = &child_weight_load_done;
  assign compute_done_o     = &child_compute_done;

endmodule
