// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Add two BF16 operands with a pure combinational datapath.
//
// Key I/O semantics:
// `a_i` and `b_i` are treated as BF16 zero or normalized values.
// The result uses truncation when alignment or normalization discards low bits.
//
// Timing / latency:
// Pure combinational add/subtract and normalization.

module bf16_add (
  input  logic [15:0] a_i,
  input  logic [15:0] b_i,
  output logic [15:0] sum_o
);

  localparam int FRAC_W      = 7;
  localparam int MANT_W      = FRAC_W + 1;
  localparam int EXTRA_W     = 7;
  localparam int EXT_MANT_W  = MANT_W + EXTRA_W;
  localparam int ADD_MANT_W  = EXT_MANT_W + 1;

  logic        a_is_zero;
  logic        b_is_zero;
  logic        a_sign;
  logic        b_sign;
  logic  [7:0] a_exp;
  logic  [7:0] b_exp;
  logic  [6:0] a_frac;
  logic  [6:0] b_frac;

  logic        large_sign;
  logic        small_sign;
  logic  [7:0] large_exp;
  logic  [7:0] small_exp;
  logic [14:0] large_mantissa;
  logic [14:0] small_mantissa;
  logic  [7:0] exp_diff;
  logic [14:0] aligned_small_mantissa;

  logic [15:0] add_result;
  logic [14:0] sub_result;
  logic [14:0] normalized_mantissa;
  logic        result_sign;
  logic        found_leading_one;
  logic  [8:0] result_exp;
  logic  [3:0] left_shift;
  integer      shift_idx;

  always_comb begin
    a_is_zero             = (a_i[14:7] == 8'h00);
    b_is_zero             = (b_i[14:7] == 8'h00);
    a_sign                = a_i[15];
    b_sign                = b_i[15];
    a_exp                 = a_i[14:7];
    b_exp                 = b_i[14:7];
    a_frac                = a_i[6:0];
    b_frac                = b_i[6:0];
    large_sign            = a_sign;
    small_sign            = b_sign;
    large_exp             = a_exp;
    small_exp             = b_exp;
    large_mantissa        = {1'b1, a_frac, {EXTRA_W{1'b0}}};
    small_mantissa        = {1'b1, b_frac, {EXTRA_W{1'b0}}};
    exp_diff              = '0;
    aligned_small_mantissa = '0;
    add_result            = '0;
    sub_result            = '0;
    normalized_mantissa   = '0;
    result_sign           = 1'b0;
    found_leading_one     = 1'b0;
    result_exp            = '0;
    left_shift            = '0;
    sum_o                 = 16'h0000;

    if (a_is_zero && b_is_zero) begin
      sum_o = 16'h0000;
    end else if (a_is_zero) begin
      sum_o = b_i;
    end else if (b_is_zero) begin
      sum_o = a_i;
    end else begin
      if ((a_exp < b_exp) || ((a_exp == b_exp) && ({1'b1, a_frac} < {1'b1, b_frac}))) begin
        large_sign     = b_sign;
        small_sign     = a_sign;
        large_exp      = b_exp;
        small_exp      = a_exp;
        large_mantissa = {1'b1, b_frac, {EXTRA_W{1'b0}}};
        small_mantissa = {1'b1, a_frac, {EXTRA_W{1'b0}}};
      end

      exp_diff = large_exp - small_exp;
      if (exp_diff >= EXT_MANT_W) begin
        aligned_small_mantissa = '0;
      end else begin
        aligned_small_mantissa = small_mantissa >> exp_diff;
      end

      result_sign = large_sign;
      result_exp  = {1'b0, large_exp};

      if (large_sign == small_sign) begin
        add_result = {1'b0, large_mantissa} + {1'b0, aligned_small_mantissa};

        if (add_result[15]) begin
          normalized_mantissa = add_result[15:1];
          result_exp          = result_exp + 9'd1;
        end else begin
          normalized_mantissa = add_result[14:0];
        end
      end else begin
        if (large_mantissa >= aligned_small_mantissa) begin
          sub_result  = large_mantissa - aligned_small_mantissa;
          result_sign = large_sign;
        end else begin
          sub_result  = aligned_small_mantissa - large_mantissa;
          result_sign = small_sign;
        end

        if (sub_result == '0) begin
          result_exp          = '0;
          normalized_mantissa = '0;
          result_sign         = 1'b0;
        end else begin
          found_leading_one = 1'b0;
          left_shift = '0;
          for (shift_idx = EXT_MANT_W - 1; shift_idx >= 0; shift_idx = shift_idx - 1) begin
            if ((sub_result[shift_idx] == 1'b1) && !found_leading_one) begin
              left_shift = (EXT_MANT_W - 1) - shift_idx;
              found_leading_one = 1'b1;
            end
          end

          if (result_exp > left_shift) begin
            normalized_mantissa = sub_result << left_shift;
            result_exp          = result_exp - left_shift;
          end else begin
            normalized_mantissa = '0;
            result_exp          = '0;
            result_sign         = 1'b0;
          end
        end
      end

      if ((normalized_mantissa == '0) || (result_exp == '0)) begin
        sum_o = 16'h0000;
      end else if (result_exp >= 9'd255) begin
        sum_o = {result_sign, 8'hfe, 7'h7f};
      end else begin
        sum_o = {result_sign, result_exp[7:0], normalized_mantissa[13:7]};
      end
    end
  end

endmodule
