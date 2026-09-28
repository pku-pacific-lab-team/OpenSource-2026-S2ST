// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Integrate external SRAM-backed weight / activation buffers with the existing
// `pe_array_hybrid_accumulator`, and optionally write finalized BF16 results
// back into activation SRAM space after MXINT4 quantization.
//
// Key I/O semantics:
// `weight_load_start_i` copies 32 weight-buffer words into the internal PE-array
// 8x4 weight slots in slot-major order:
// slot0 rows0..7, then slot1 rows0..7, then slot2 rows0..7, then slot3 rows0..7.
// When `use_reorder_seq_i = 1`, the source reads are regrouped into four
// sequence-selected 8-word blocks while the internal write order stays slot-major.
// `compute_start_i` reads one activation word, one activation scaling factor,
// and one packed weight-sf word, builds per-lane `sf_vec`, and issues exactly
// one `compute_accumulate` pulse into the reused hybrid accumulator.
// When `writeback_enable_i = 1`, the wrapper waits for any queued FP flushes to
// drain, issues one residual finalize pulse, waits for the FP path to empty,
// quantizes the internal BF16 results to MXINT4, writes the results to the
// activation / activation-sf / mask SRAMs at `writeback_addr_i`, then clears the
// accumulator state.
//
// Timing / latency:
// All SRAMs are treated as synchronous-read macros with active-low `cen` /
// `wen`. Weight load accepts one command, launches the first SRAM read on that
// edge, then streams 32 PE writes in `weight_addr=0..31` order. Compute accepts
// one command, launches the SRAM reads on that edge, and drives exactly one
// hybrid-accumulator execute pulse after the SRAM outputs settle. Writeback
// latency depends on the serializer drain time.

module pe_array_buffered_hybrid_top (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               clear_i,

  input  logic               weight_buf_en_i,
  input  logic               weight_buf_wen_i,
  input  logic         [8:0] weight_buf_addr_i,
  input  logic        [31:0] weight_buf_wdata_i,
  output logic        [31:0] weight_buf_rdata_o,

  input  logic               weight_sf_buf_en_i,
  input  logic               weight_sf_buf_wen_i,
  input  logic         [5:0] weight_sf_buf_addr_i,
  input  logic        [39:0] weight_sf_buf_wdata_i,
  output logic        [39:0] weight_sf_buf_rdata_o,

  input  logic               activation_buf_en_i,
  input  logic               activation_buf_wen_i,
  input  logic         [8:0] activation_buf_addr_i,
  input  logic        [31:0] activation_buf_wdata_i,
  output logic        [31:0] activation_buf_rdata_o,

  input  logic               activation_sf_buf_en_i,
  input  logic               activation_sf_buf_wen_i,
  input  logic         [8:0] activation_sf_buf_addr_i,
  input  logic         [4:0] activation_sf_buf_wdata_i,
  output logic         [4:0] activation_sf_buf_rdata_o,

  input  logic               mask_buf_en_i,
  input  logic               mask_buf_wen_i,
  input  logic         [8:0] mask_buf_addr_i,
  input  logic         [7:0] mask_buf_wdata_i,
  output logic         [7:0] mask_buf_rdata_o,

  input  logic               weight_load_start_i,
  input  logic         [8:0] weight_load_base_addr_i,
  input  logic               compute_start_i,
  input  logic               fusion_preload_start_i,
  input  logic               fusion_mode_en_i,
  input  logic         [8:0] activation_addr_i,
  input  logic         [5:0] weight_sf_addr_i,
  input  logic        [15:0] row_weight_rsel_i,
  input  logic         [7:0] fp_flush_flag_i,
  input  logic               writeback_enable_i,
  input  logic         [8:0] writeback_addr_i,
  input  logic         [7:0] threshold_i,
  input  logic               use_reorder_seq_i,
  input  logic               reorder_ptr_reset_i,
  input  logic               reorder_seq_we_i,
  input  logic        [23:0] reorder_seq_i,

  output logic               busy_o,
  output logic               weight_load_done_o,
  output logic               compute_done_o
);

  typedef enum logic [3:0] {
    STATE_IDLE                = 4'd0,
    STATE_LOAD_STREAM         = 4'd1,
    STATE_FUSION_PRELOAD      = 4'd2,
    STATE_COMP_EXEC           = 4'd3,
    STATE_WAIT_FINALIZE_EMPTY = 4'd4,
    STATE_ISSUE_FINALIZE      = 4'd5,
    STATE_WAIT_FINALIZE_DRAIN = 4'd6,
    STATE_WRITEBACK           = 4'd7,
    STATE_CLEAR_ACCUM         = 4'd8
  } state_t;

  state_t             state_d;
  state_t             state_q;
  logic         [8:0] load_base_addr_d;
  logic         [8:0] load_base_addr_q;
  logic         [4:0] load_idx_d;
  logic         [4:0] load_idx_q;
  logic        [15:0] compute_row_weight_rsel_d;
  logic        [15:0] compute_row_weight_rsel_q;
  logic         [7:0] compute_fp_flush_flag_d;
  logic         [7:0] compute_fp_flush_flag_q;
  logic               compute_writeback_enable_d;
  logic               compute_writeback_enable_q;
  logic         [8:0] compute_writeback_addr_d;
  logic         [8:0] compute_writeback_addr_q;
  logic         [7:0] compute_threshold_d;
  logic         [7:0] compute_threshold_q;
  logic               fusion_ready_d;
  logic               fusion_ready_q;
  logic               compute_use_fusion_d;
  logic               compute_use_fusion_q;
  logic         [8:0] fusion_preload_base_addr_d;
  logic         [8:0] fusion_preload_base_addr_q;
  logic         [1:0] fusion_preload_idx_d;
  logic         [1:0] fusion_preload_idx_q;
  logic               weight_load_done_d;
  logic               weight_load_done_q;
  logic               compute_done_d;
  logic               compute_done_q;
  logic               load_use_reorder_q;
  logic        [23:0] reorder_seq_q;
  logic         [2:0] reorder_ptr_q;

  logic               fusion_preload_accept;
  logic               weight_load_accept;
  logic               compute_accept;
  logic               compute_reorder_accept;

  logic               weight_data_en;
  logic               weight_data_write;
  logic         [8:0] weight_data_addr;
  logic        [31:0] weight_data_wdata;
  logic        [31:0] weight_data_rdata;

  logic               weight_sf_mem_en;
  logic               weight_sf_mem_write;
  logic         [5:0] weight_sf_mem_addr;
  logic        [39:0] weight_sf_mem_wdata;
  logic        [39:0] weight_sf_mem_rdata;

  logic               activation_data_en;
  logic               activation_data_write;
  logic         [8:0] activation_data_addr;
  logic        [31:0] activation_data_wdata;
  logic        [31:0] activation_data_rdata;

  logic               activation_sf_mem_en;
  logic               activation_sf_mem_write;
  logic         [8:0] activation_sf_mem_addr;
  logic         [4:0] activation_sf_mem_wdata;
  logic         [4:0] activation_sf_mem_rdata;

  logic               mask_mem_en;
  logic               mask_mem_write;
  logic         [8:0] mask_mem_addr;
  logic         [7:0] mask_mem_wdata;
  logic         [7:0] mask_mem_rdata;

  logic               hybrid_clear;
  logic               hybrid_weight_we;
  logic         [4:0] hybrid_weight_addr;
  logic        [31:0] hybrid_weight_row;
  logic        [15:0] hybrid_row_weight_rsel;
  logic        [31:0] hybrid_activation_vec;
  logic        [39:0] hybrid_sf_vec;
  logic         [7:0] hybrid_fp_flush_flag;
  logic               hybrid_compute_accumulate;
  logic               hybrid_finalize;
  logic               hybrid_fp_path_busy;
  logic [7:0][15:0]   hybrid_fp_result;
  logic         [2:0] compute_seq_value;

  logic signed  [4:0] weight_sf_lane;
  logic signed  [5:0] sf_sum_lane;
  logic       [127:0] quant_bf16_vec;
  logic         [7:0] sparse_mask;
  logic        [31:0] writeback_mxint4_vec;
  logic signed  [4:0] writeback_scaling_factor;
  logic               fusion_slots_clear;
  logic               fusion_shift_refill;
  logic        [31:0] fusion_refill_activation;
  logic signed  [4:0] fusion_refill_sf;
  logic         [7:0] fusion_refill_mask;
  logic        [31:0] fusion_activation_vec;
  logic signed  [4:0] fusion_activation_sf;
  logic        [15:0] fusion_row_weight_rsel;
  logic signed  [4:0] selected_activation_sf;

  function automatic logic [2:0] seq_elem(
    input logic [23:0] packed_seq,
    input logic [2:0]  seq_idx
  );
    begin
      unique case (seq_idx)
        3'd0: seq_elem = packed_seq[2:0];
        3'd1: seq_elem = packed_seq[5:3];
        3'd2: seq_elem = packed_seq[8:6];
        3'd3: seq_elem = packed_seq[11:9];
        3'd4: seq_elem = packed_seq[14:12];
        3'd5: seq_elem = packed_seq[17:15];
        3'd6: seq_elem = packed_seq[20:18];
        3'd7: seq_elem = packed_seq[23:21];
        default: seq_elem = 3'bxxx;
      endcase
    end
  endfunction

  function automatic logic [8:0] reordered_weight_src_addr(
    input logic [8:0]  base_addr,
    input logic [4:0]  load_entry_idx,
    input logic [2:0]  ptr_value,
    input logic [23:0] packed_seq
  );
    logic [1:0] block_sel;
    logic [2:0] row_in_block;
    logic [2:0] seq_value;
    begin
      block_sel    = load_entry_idx[4:3];
      row_in_block = load_entry_idx[2:0];
      seq_value    = seq_elem(packed_seq, ptr_value + {1'b0, block_sel});
      reordered_weight_src_addr = base_addr
                                + {{3{1'b0}}, seq_value, 3'b000}
                                + {{6{1'b0}}, row_in_block};
    end
  endfunction

  assign fusion_preload_accept = (state_q == STATE_IDLE)
                              && !clear_i
                              && !weight_load_start_i
                              && fusion_preload_start_i;
  assign weight_load_accept = (state_q == STATE_IDLE)
                           && !clear_i
                           && weight_load_start_i;
  assign compute_accept = (state_q == STATE_IDLE)
                       && !clear_i
                       && !weight_load_start_i
                       && !fusion_preload_accept
                       && compute_start_i
                       && (!fusion_mode_en_i || fusion_ready_q);
  assign compute_reorder_accept = compute_accept && use_reorder_seq_i && !fusion_mode_en_i;
  assign compute_seq_value      = seq_elem(reorder_seq_q, reorder_ptr_q);

  always_comb begin
    state_d                    = state_q;
    load_base_addr_d           = load_base_addr_q;
    load_idx_d                 = load_idx_q;
    compute_row_weight_rsel_d  = compute_row_weight_rsel_q;
    compute_fp_flush_flag_d    = compute_fp_flush_flag_q;
    compute_writeback_enable_d = compute_writeback_enable_q;
    compute_writeback_addr_d   = compute_writeback_addr_q;
    compute_threshold_d        = compute_threshold_q;
    fusion_ready_d             = fusion_ready_q;
    compute_use_fusion_d       = compute_use_fusion_q;
    fusion_preload_base_addr_d = fusion_preload_base_addr_q;
    fusion_preload_idx_d       = fusion_preload_idx_q;
    weight_load_done_d         = 1'b0;
    compute_done_d             = 1'b0;

    if (clear_i) begin
      state_d                    = STATE_IDLE;
      load_base_addr_d           = '0;
      load_idx_d                 = '0;
      compute_row_weight_rsel_d  = '0;
      compute_fp_flush_flag_d    = '0;
      compute_writeback_enable_d = 1'b0;
      compute_writeback_addr_d   = '0;
      compute_threshold_d        = '0;
      fusion_ready_d             = 1'b0;
      compute_use_fusion_d       = 1'b0;
      fusion_preload_base_addr_d = '0;
      fusion_preload_idx_d       = '0;
    end else begin
      unique case (state_q)
        STATE_IDLE: begin
          if (weight_load_accept) begin
            state_d          = STATE_LOAD_STREAM;
            load_base_addr_d = weight_load_base_addr_i;
            load_idx_d       = 5'd0;
          end else if (fusion_preload_accept) begin
            state_d                    = STATE_FUSION_PRELOAD;
            fusion_ready_d             = 1'b0;
            compute_use_fusion_d       = 1'b0;
            fusion_preload_base_addr_d = activation_addr_i;
            fusion_preload_idx_d       = 2'd0;
          end else if (compute_accept) begin
            state_d                    = STATE_COMP_EXEC;
            compute_row_weight_rsel_d  = fusion_mode_en_i ? '0 : row_weight_rsel_i;
            compute_fp_flush_flag_d    = fp_flush_flag_i;
            compute_writeback_enable_d = writeback_enable_i;
            compute_writeback_addr_d   = writeback_addr_i;
            compute_threshold_d        = threshold_i;
            compute_use_fusion_d       = fusion_mode_en_i;
          end
        end

        STATE_LOAD_STREAM: begin
          if (load_idx_q == 5'd31) begin
            state_d            = STATE_IDLE;
            load_base_addr_d   = '0;
            load_idx_d         = '0;
            weight_load_done_d = 1'b1;
          end else begin
            load_idx_d = load_idx_q + 5'd1;
          end
        end

        STATE_FUSION_PRELOAD: begin
          if (fusion_preload_idx_q == 2'd3) begin
            state_d                    = STATE_IDLE;
            fusion_ready_d             = 1'b1;
            fusion_preload_base_addr_d = '0;
            fusion_preload_idx_d       = '0;
          end else begin
            fusion_preload_idx_d = fusion_preload_idx_q + 2'd1;
          end
        end

        STATE_COMP_EXEC: begin
          if (compute_writeback_enable_q) begin
            state_d = STATE_WAIT_FINALIZE_EMPTY;
          end else begin
            state_d                    = STATE_IDLE;
            compute_row_weight_rsel_d  = '0;
            compute_fp_flush_flag_d    = '0;
            compute_writeback_enable_d = 1'b0;
            compute_writeback_addr_d   = '0;
            compute_threshold_d        = '0;
            compute_use_fusion_d       = 1'b0;
            compute_done_d             = 1'b1;
          end
        end

        STATE_WAIT_FINALIZE_EMPTY: begin
          if (!hybrid_fp_path_busy) begin
            state_d = STATE_ISSUE_FINALIZE;
          end
        end

        STATE_ISSUE_FINALIZE: begin
          state_d = STATE_WAIT_FINALIZE_DRAIN;
        end

        STATE_WAIT_FINALIZE_DRAIN: begin
          if (!hybrid_fp_path_busy) begin
            state_d = STATE_WRITEBACK;
          end
        end

        STATE_WRITEBACK: begin
          state_d = STATE_CLEAR_ACCUM;
        end

        STATE_CLEAR_ACCUM: begin
          state_d                    = STATE_IDLE;
          compute_row_weight_rsel_d  = '0;
          compute_fp_flush_flag_d    = '0;
          compute_writeback_enable_d = 1'b0;
          compute_writeback_addr_d   = '0;
          compute_threshold_d        = '0;
          compute_use_fusion_d       = 1'b0;
          compute_done_d             = 1'b1;
        end

        default: begin
          state_d = STATE_IDLE;
        end
      endcase
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q                    <= STATE_IDLE;
      load_base_addr_q           <= '0;
      load_idx_q                 <= '0;
      compute_row_weight_rsel_q  <= '0;
      compute_fp_flush_flag_q    <= '0;
      compute_writeback_enable_q <= 1'b0;
      compute_writeback_addr_q   <= '0;
      compute_threshold_q        <= '0;
      fusion_ready_q             <= 1'b0;
      compute_use_fusion_q       <= 1'b0;
      fusion_preload_base_addr_q <= '0;
      fusion_preload_idx_q       <= '0;
      weight_load_done_q         <= 1'b0;
      compute_done_q             <= 1'b0;
    end else begin
      state_q                    <= state_d;
      load_base_addr_q           <= load_base_addr_d;
      load_idx_q                 <= load_idx_d;
      compute_row_weight_rsel_q  <= compute_row_weight_rsel_d;
      compute_fp_flush_flag_q    <= compute_fp_flush_flag_d;
      compute_writeback_enable_q <= compute_writeback_enable_d;
      compute_writeback_addr_q   <= compute_writeback_addr_d;
      compute_threshold_q        <= compute_threshold_d;
      fusion_ready_q             <= fusion_ready_d;
      compute_use_fusion_q       <= compute_use_fusion_d;
      fusion_preload_base_addr_q <= fusion_preload_base_addr_d;
      fusion_preload_idx_q       <= fusion_preload_idx_d;
      weight_load_done_q         <= weight_load_done_d;
      compute_done_q             <= compute_done_d;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      load_use_reorder_q <= 1'b0;
    end else if (clear_i) begin
      load_use_reorder_q <= 1'b0;
    end else if (weight_load_accept) begin
      load_use_reorder_q <= use_reorder_seq_i;
    end else if ((state_q == STATE_LOAD_STREAM) && (load_idx_q == 5'd31)) begin
      load_use_reorder_q <= 1'b0;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      reorder_ptr_q <= 3'd0;
    end else if (reorder_ptr_reset_i) begin
      reorder_ptr_q <= 3'd0;
    end else if (compute_reorder_accept) begin
      reorder_ptr_q <= reorder_ptr_q + 3'd1;
    end
  end

  always_ff @(posedge clk_i) begin
    if (reorder_seq_we_i) begin
      reorder_seq_q <= reorder_seq_i;
    end
  end

  always_comb begin
    weight_data_en    = 1'b0;
    weight_data_write = 1'b0;
    weight_data_addr  = '0;
    weight_data_wdata = '0;

    if (!clear_i) begin
      if (state_q == STATE_IDLE) begin
        if (weight_load_accept) begin
          weight_data_en    = 1'b1;
          weight_data_write = 1'b0;
          weight_data_addr  = use_reorder_seq_i
            ? reordered_weight_src_addr(weight_load_base_addr_i, 5'd0, reorder_ptr_q, reorder_seq_q)
            : weight_load_base_addr_i;
        end else begin
          weight_data_en    = weight_buf_en_i;
          weight_data_write = weight_buf_wen_i;
          weight_data_addr  = weight_buf_addr_i;
          weight_data_wdata = weight_buf_wdata_i;
        end
      end else if (state_q == STATE_LOAD_STREAM) begin
        if (load_idx_q != 5'd31) begin
          weight_data_en    = 1'b1;
          weight_data_write = 1'b0;
          weight_data_addr  = load_use_reorder_q
            ? reordered_weight_src_addr(load_base_addr_q, load_idx_q + 5'd1, reorder_ptr_q, reorder_seq_q)
            : (load_base_addr_q + {4'd0, load_idx_q} + 9'd1);
        end
      end
    end
  end

  always_comb begin
    weight_sf_mem_en    = 1'b0;
    weight_sf_mem_write = 1'b0;
    weight_sf_mem_addr  = '0;
    weight_sf_mem_wdata = '0;

    if (!clear_i) begin
      if (state_q == STATE_IDLE) begin
        if (compute_accept) begin
          weight_sf_mem_en    = 1'b1;
          weight_sf_mem_write = 1'b0;
          weight_sf_mem_addr  = compute_reorder_accept
            ? (weight_sf_addr_i + {{3{1'b0}}, compute_seq_value})
            : weight_sf_addr_i;
        end else begin
          weight_sf_mem_en    = weight_sf_buf_en_i;
          weight_sf_mem_write = weight_sf_buf_wen_i;
          weight_sf_mem_addr  = weight_sf_buf_addr_i;
          weight_sf_mem_wdata = weight_sf_buf_wdata_i;
        end
      end
    end
  end

  always_comb begin
    activation_data_en    = 1'b0;
    activation_data_write = 1'b0;
    activation_data_addr  = '0;
    activation_data_wdata = '0;

    if (!clear_i) begin
      if (state_q == STATE_IDLE) begin
        if (fusion_preload_accept || compute_accept) begin
          activation_data_en    = 1'b1;
          activation_data_write = 1'b0;
          activation_data_addr  = compute_reorder_accept
            ? (activation_addr_i + {{6{1'b0}}, compute_seq_value})
            : activation_addr_i;
        end else begin
          activation_data_en    = activation_buf_en_i;
          activation_data_write = activation_buf_wen_i;
          activation_data_addr  = activation_buf_addr_i;
          activation_data_wdata = activation_buf_wdata_i;
        end
      end else if (state_q == STATE_FUSION_PRELOAD) begin
        if (fusion_preload_idx_q != 2'd3) begin
          activation_data_en    = 1'b1;
          activation_data_write = 1'b0;
          activation_data_addr  = fusion_preload_base_addr_q
                                + {{7{1'b0}}, fusion_preload_idx_q}
                                + 9'd1;
        end
      end else if (state_q == STATE_WRITEBACK) begin
        activation_data_en    = 1'b1;
        activation_data_write = 1'b1;
        activation_data_addr  = compute_writeback_addr_q;
        activation_data_wdata = writeback_mxint4_vec;
      end
    end
  end

  always_comb begin
    activation_sf_mem_en    = 1'b0;
    activation_sf_mem_write = 1'b0;
    activation_sf_mem_addr  = '0;
    activation_sf_mem_wdata = '0;

    if (!clear_i) begin
      if (state_q == STATE_IDLE) begin
        if (fusion_preload_accept || compute_accept) begin
          activation_sf_mem_en    = 1'b1;
          activation_sf_mem_write = 1'b0;
          activation_sf_mem_addr  = compute_reorder_accept
            ? (activation_addr_i + {{6{1'b0}}, compute_seq_value})
            : activation_addr_i;
        end else begin
          activation_sf_mem_en    = activation_sf_buf_en_i;
          activation_sf_mem_write = activation_sf_buf_wen_i;
          activation_sf_mem_addr  = activation_sf_buf_addr_i;
          activation_sf_mem_wdata = activation_sf_buf_wdata_i;
        end
      end else if (state_q == STATE_FUSION_PRELOAD) begin
        if (fusion_preload_idx_q != 2'd3) begin
          activation_sf_mem_en    = 1'b1;
          activation_sf_mem_write = 1'b0;
          activation_sf_mem_addr  = fusion_preload_base_addr_q
                                  + {{7{1'b0}}, fusion_preload_idx_q}
                                  + 9'd1;
        end
      end else if (state_q == STATE_WRITEBACK) begin
        activation_sf_mem_en    = 1'b1;
        activation_sf_mem_write = 1'b1;
        activation_sf_mem_addr  = compute_writeback_addr_q;
        activation_sf_mem_wdata = writeback_scaling_factor[4:0];
      end
    end
  end

  always_comb begin
    mask_mem_en    = 1'b0;
    mask_mem_write = 1'b0;
    mask_mem_addr  = '0;
    mask_mem_wdata = '0;

    if (!clear_i) begin
      if (state_q == STATE_WRITEBACK) begin
        mask_mem_en    = 1'b1;
        mask_mem_write = 1'b1;
        mask_mem_addr  = compute_writeback_addr_q;
        mask_mem_wdata = sparse_mask;
      end else if (state_q == STATE_FUSION_PRELOAD) begin
        if (fusion_preload_idx_q != 2'd3) begin
          mask_mem_en    = 1'b1;
          mask_mem_write = 1'b0;
          mask_mem_addr  = fusion_preload_base_addr_q
                         + {{7{1'b0}}, fusion_preload_idx_q}
                         + 9'd1;
        end
      end else if ((state_q == STATE_IDLE) && (fusion_preload_accept || (compute_accept && fusion_mode_en_i))) begin
        mask_mem_en    = 1'b1;
        mask_mem_write = 1'b0;
        mask_mem_addr  = activation_addr_i;
      end else begin
        mask_mem_en    = mask_buf_en_i;
        mask_mem_write = mask_buf_wen_i;
        mask_mem_addr  = mask_buf_addr_i;
        mask_mem_wdata = mask_buf_wdata_i;
      end
    end
  end

  always_comb begin
    if (compute_use_fusion_q) begin
      selected_activation_sf = fusion_activation_sf;
    end else begin
      selected_activation_sf = $signed(activation_sf_mem_rdata);
    end

    hybrid_sf_vec = '0;

    for (int lane_idx = 0; lane_idx < 8; lane_idx++) begin
      weight_sf_lane = $signed(weight_sf_mem_rdata[lane_idx * 5 +: 5]);
      sf_sum_lane = $signed({weight_sf_lane[4], weight_sf_lane})
                  + $signed({selected_activation_sf[4], selected_activation_sf});
      hybrid_sf_vec[lane_idx * 5 +: 5] = sf_sum_lane[4:0];
    end
  end

  always_comb begin
    fusion_slots_clear       = 1'b0;
    fusion_shift_refill      = 1'b0;
    fusion_refill_activation = activation_data_rdata;
    fusion_refill_sf         = $signed(activation_sf_mem_rdata);
    fusion_refill_mask       = mask_mem_rdata;

    if (!clear_i) begin
      if (fusion_preload_accept) begin
        fusion_slots_clear = 1'b1;
      end

      if (state_q == STATE_FUSION_PRELOAD) begin
        fusion_shift_refill = 1'b1;
      end else if ((state_q == STATE_COMP_EXEC) && compute_use_fusion_q) begin
        fusion_shift_refill = 1'b1;
      end
    end
  end

  always_comb begin
    sparse_mask    = 8'h00;
    quant_bf16_vec = '0;

    for (int lane_idx = 0; lane_idx < 8; lane_idx++) begin
      if (hybrid_fp_result[lane_idx][14:7] < compute_threshold_q) begin
        sparse_mask[lane_idx] = 1'b1;
        quant_bf16_vec[lane_idx * 16 +: 16] = 16'h0000;
      end else begin
        quant_bf16_vec[lane_idx * 16 +: 16] = hybrid_fp_result[lane_idx];
      end
    end
  end

  always_comb begin
    hybrid_weight_we          = 1'b0;
    hybrid_weight_addr        = '0;
    hybrid_weight_row         = '0;
    hybrid_row_weight_rsel    = '0;
    hybrid_activation_vec     = '0;
    hybrid_fp_flush_flag      = '0;
    hybrid_compute_accumulate = 1'b0;
    hybrid_finalize           = 1'b0;

    if (!clear_i) begin
      if (state_q == STATE_LOAD_STREAM) begin
        hybrid_weight_we   = 1'b1;
        hybrid_weight_addr = {load_idx_q[2:0], load_idx_q[4:3]};
        hybrid_weight_row  = weight_data_rdata;
      end else if (state_q == STATE_COMP_EXEC) begin
        hybrid_row_weight_rsel    = compute_use_fusion_q ? fusion_row_weight_rsel
                                                         : compute_row_weight_rsel_q;
        hybrid_activation_vec     = compute_use_fusion_q ? fusion_activation_vec
                                                         : activation_data_rdata;
        hybrid_fp_flush_flag      = compute_fp_flush_flag_q;
        hybrid_compute_accumulate = 1'b1;
      end else if (state_q == STATE_ISSUE_FINALIZE) begin
        hybrid_finalize = 1'b1;
      end
    end
  end

  assign hybrid_clear = clear_i || (state_q == STATE_CLEAR_ACCUM);

  bf16_to_mxint4 quantizer_u (
    .valid_i          (state_q == STATE_WRITEBACK),
    .bf16_vec_i       (quant_bf16_vec),
    .mxint4_vec_o     (writeback_mxint4_vec),
    .scaling_factor_o (writeback_scaling_factor)
  );

  activation_fusion_buffer fusion_buffer_u (
    .clk_i                   (clk_i),
    .clear_i                 (clear_i),
    .slots_clear_i           (fusion_slots_clear),
    .shift_refill_i          (fusion_shift_refill),
    .refill_activation_i     (fusion_refill_activation),
    .refill_activation_sf_i  (fusion_refill_sf),
    .refill_mask_i           (fusion_refill_mask),
    .fused_activation_vec_o  (fusion_activation_vec),
    .fused_activation_sf_o   (fusion_activation_sf),
    .fused_row_weight_rsel_o (fusion_row_weight_rsel)
  );

  sram512x32 weight_data_sram_u (
    .clk   (clk_i),
    .cen   (~weight_data_en),
    .wen   (~weight_data_write),
    .a     (weight_data_addr),
    .d     (weight_data_wdata),
    .q     (weight_data_rdata),
    .ema   (3'b100),
    .emaw  (2'b00),
    .emas  (1'b0),
    .ret1n (1'b1),
    .rawl  (1'b0),
    .rawlm (2'b00),
    .wabl  (1'b1),
    .wablm (2'b01)
  );

  sram64x40 weight_sf_sram_u (
    .clk   (clk_i),
    .cen   (~weight_sf_mem_en),
    .wen   (~weight_sf_mem_write),
    .a     (weight_sf_mem_addr),
    .d     (weight_sf_mem_wdata),
    .q     (weight_sf_mem_rdata),
    .ema   (3'b100),
    .emaw  (2'b00),
    .emas  (1'b0),
    .ret1n (1'b1),
    .rawl  (1'b0),
    .rawlm (2'b00),
    .wabl  (1'b1),
    .wablm (2'b01)
  );

  sram512x32 activation_data_sram_u (
    .clk   (clk_i),
    .cen   (~activation_data_en),
    .wen   (~activation_data_write),
    .a     (activation_data_addr),
    .d     (activation_data_wdata),
    .q     (activation_data_rdata),
    .ema   (3'b100),
    .emaw  (2'b00),
    .emas  (1'b0),
    .ret1n (1'b1),
    .rawl  (1'b0),
    .rawlm (2'b00),
    .wabl  (1'b1),
    .wablm (2'b01)
  );

  sram512x5 activation_sf_sram_u (
    .clk   (clk_i),
    .cen   (~activation_sf_mem_en),
    .wen   (~activation_sf_mem_write),
    .a     (activation_sf_mem_addr),
    .d     (activation_sf_mem_wdata),
    .q     (activation_sf_mem_rdata),
    .ema   (3'b100),
    .emaw  (2'b00),
    .emas  (1'b0),
    .ret1n (1'b1),
    .rawl  (1'b0),
    .rawlm (2'b00),
    .wabl  (1'b1),
    .wablm (2'b01)
  );

  sram512x8 mask_sram_u (
    .clk   (clk_i),
    .cen   (~mask_mem_en),
    .wen   (~mask_mem_write),
    .a     (mask_mem_addr),
    .d     (mask_mem_wdata),
    .q     (mask_mem_rdata),
    .ema   (3'b100),
    .emaw  (2'b00),
    .emas  (1'b0),
    .ret1n (1'b1),
    .rawl  (1'b0),
    .rawlm (2'b00),
    .wabl  (1'b1),
    .wablm (2'b01)
  );

  pe_array_hybrid_accumulator hybrid_acc_u (
    .clk_i               (clk_i),
    .rst_ni              (rst_ni),
    .clear_i             (hybrid_clear),
    .weight_we_i         (hybrid_weight_we),
    .weight_addr_i       (hybrid_weight_addr),
    .weight_row_i        (hybrid_weight_row),
    .row_weight_rsel_i   (hybrid_row_weight_rsel),
    .activation_vec_i    (hybrid_activation_vec),
    .sf_vec_i            (hybrid_sf_vec),
    .fp_flush_flag_i     (hybrid_fp_flush_flag),
    .compute_accumulate_i(hybrid_compute_accumulate),
    .finalize_i          (hybrid_finalize),
    .fp_path_busy_o      (hybrid_fp_path_busy),
    .fp_result_o         (hybrid_fp_result)
  );

  assign weight_buf_rdata_o        = weight_data_rdata;
  assign weight_sf_buf_rdata_o     = weight_sf_mem_rdata;
  assign activation_buf_rdata_o    = activation_data_rdata;
  assign activation_sf_buf_rdata_o = activation_sf_mem_rdata;
  assign mask_buf_rdata_o          = mask_mem_rdata;

  assign busy_o                    = (state_q != STATE_IDLE);
  assign weight_load_done_o        = weight_load_done_q;
  assign compute_done_o            = compute_done_q;

endmodule
