// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Maintain a 512-line tagged probability cache and a 32-entry processing sequence.
//
// Key I/O semantics:
// `update_valid_i` is accepted only when `busy_o` is low.
// `clear_i` clears the logical module state and has priority over update acceptance.
// `seq_read_data_o` returns one packed `{index,timestamp}` sequence word or zero when out of range.
//
// Timing / latency:
// Cache payload accesses go through a single-port SRAM with one-cycle synchronous reads.
// Each accepted update consumes one read-launch cycle and one update cycle.

module prob_cache (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        clear_i,
  input  logic        update_valid_i,
  input  logic [12:0] index_i,
  input  logic  [7:0] probability_i,
  input  logic  [3:0] timestamp_i,
  input  logic  [7:0] threshold_i,
  input  logic  [4:0] seq_read_addr_i,
  output logic        busy_o,
  output logic        seq_full_o,
  output logic  [5:0] seq_count_o,
  output logic [16:0] seq_read_data_o
);

  localparam int CACHE_DEPTH = 512;
  localparam int SEQ_DEPTH   = 32;

  typedef enum logic [1:0] {
    IDLE,
    LOOKUP_UPDATE,
    CLEAR
  } state_t;

  state_t state_q;

  logic        cache_data_en;
  logic        cache_data_wen;
  logic  [8:0] cache_data_addr;
  logic [16:0] cache_data_wdata;
  logic [16:0] cache_data_rdata;

  logic        valid_mem  [0:CACHE_DEPTH-1];
  logic        queued_mem [0:CACHE_DEPTH-1];
  logic [16:0] seq_mem    [0:SEQ_DEPTH-1];

  logic [12:0] req_index_q;
  logic  [7:0] req_probability_q;
  logic  [3:0] req_timestamp_q;
  logic  [7:0] req_threshold_q;
  logic  [8:0] read_addr_q;
  logic  [5:0] seq_count_q;
  logic  [8:0] clear_idx_q;

  logic        hit;
  logic        inserted_before;
  logic  [7:0] probability_new;
  logic  [3:0] seq_timestamp;
  logic [16:0] updated_word;

  integer line_idx;
  integer seq_idx;

  function automatic logic [3:0] cache_tag(input logic [16:0] word);
    cache_tag = word[16:13];
  endfunction

  function automatic logic [7:0] cache_probability(input logic [16:0] word);
    cache_probability = word[12:5];
  endfunction

  function automatic logic [3:0] cache_timestamp(input logic [16:0] word);
    cache_timestamp = word[4:1];
  endfunction

  always_comb begin
    cache_data_en    = 1'b0;
    cache_data_wen   = 1'b0;
    cache_data_addr  = '0;
    cache_data_wdata = '0;

    hit             = valid_mem[read_addr_q] && (cache_tag(cache_data_rdata) == req_index_q[12:9]);
    inserted_before = hit ? queued_mem[read_addr_q] : 1'b0;

    if (hit) begin
      probability_new = cache_probability(cache_data_rdata) + req_probability_q;
      seq_timestamp   = cache_timestamp(cache_data_rdata);
      updated_word    = {cache_tag(cache_data_rdata), probability_new, cache_timestamp(cache_data_rdata), 1'b1};
    end else begin
      probability_new = req_probability_q;
      seq_timestamp   = req_timestamp_q;
      updated_word    = {req_index_q[12:9], req_probability_q, req_timestamp_q, 1'b1};
    end

    case (state_q)
      IDLE: begin
        if (update_valid_i && !clear_i) begin
          cache_data_en   = 1'b1;
          cache_data_wen  = 1'b0;
          cache_data_addr = index_i[8:0];
        end
      end

      LOOKUP_UPDATE: begin
        cache_data_en    = 1'b1;
        cache_data_wen   = 1'b1;
        cache_data_addr  = read_addr_q;
        cache_data_wdata = updated_word;
      end

      default: begin
      end
    endcase
  end

  assign busy_o      = (state_q != IDLE);
  assign seq_full_o  = (seq_count_q == 6'd32);
  assign seq_count_o = seq_count_q;

  always_comb begin
    if ({1'b0, seq_read_addr_i} < seq_count_q) begin
      seq_read_data_o = seq_mem[seq_read_addr_i];
    end else begin
      seq_read_data_o = 17'd0;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q           <= IDLE;
      req_index_q       <= '0;
      req_probability_q <= '0;
      req_timestamp_q   <= '0;
      req_threshold_q   <= '0;
      read_addr_q       <= '0;
      seq_count_q       <= '0;
      clear_idx_q       <= '0;

      for (line_idx = 0; line_idx < CACHE_DEPTH; line_idx++) begin
        valid_mem[line_idx]  <= 1'b0;
        queued_mem[line_idx] <= 1'b0;
      end

      for (seq_idx = 0; seq_idx < SEQ_DEPTH; seq_idx++) begin
        seq_mem[seq_idx] <= '0;
      end
    end else begin
      case (state_q)
        IDLE: begin
          if (clear_i) begin
            state_q     <= CLEAR;
            clear_idx_q <= 9'd0;
            seq_count_q <= 6'd0;

            for (seq_idx = 0; seq_idx < SEQ_DEPTH; seq_idx++) begin
              seq_mem[seq_idx] <= '0;
            end
          end else if (update_valid_i) begin
            req_index_q       <= index_i;
            req_probability_q <= probability_i;
            req_timestamp_q   <= timestamp_i;
            req_threshold_q   <= threshold_i;
            read_addr_q       <= index_i[8:0];
            state_q           <= LOOKUP_UPDATE;
          end
        end

        LOOKUP_UPDATE: begin
          valid_mem[read_addr_q] <= 1'b1;

          if ((probability_new > req_threshold_q) && !inserted_before && !seq_full_o) begin
            seq_mem[seq_count_q]    <= {req_index_q, seq_timestamp};
            queued_mem[read_addr_q] <= 1'b1;
            seq_count_q             <= seq_count_q + 6'd1;
          end else begin
            queued_mem[read_addr_q] <= inserted_before;
          end

          state_q <= IDLE;
        end

        CLEAR: begin
          valid_mem[clear_idx_q]  <= 1'b0;
          queued_mem[clear_idx_q] <= 1'b0;

          if (clear_idx_q == 9'd511) begin
            state_q     <= IDLE;
            clear_idx_q <= 9'd0;
          end else begin
            clear_idx_q <= clear_idx_q + 9'd1;
          end
        end

        default: begin
          state_q <= IDLE;
        end
      endcase
    end
  end

  sram512x17 i_cache_data_sram (
    .clk   (clk_i),
    .cen   (~cache_data_en),
    .wen   (~cache_data_wen),
    .a     (cache_data_addr),
    .d     (cache_data_wdata),
    .q     (cache_data_rdata),
    .ema   (3'b100),
    .emaw  (2'b00),
    .emas  (1'b0),
    .ret1n (1'b1),
    .rawl  (1'b0),
    .rawlm (2'b00),
    .wabl  (1'b1),
    .wablm (2'b01)
  );

endmodule
