// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Convert a signed INT12 plus signed scaling factor pair into IEEE bfloat16.
//
// Key I/O semantics:
// `int12_i` and `scaling_factor_i` represent the numeric value
// `int12_i * 2^scaling_factor_i`.
//
// Timing / latency:
// Pure combinational conversion with truncation and zero special handling.

module int12_scale_to_bf16 (
  input  logic signed [11:0] int12_i,
  input  logic signed  [4:0] scaling_factor_i,
  output logic        [15:0] bf16_o
);

  logic        is_zero;
  logic        sign_bit;
  logic [12:0] abs_ext;
  logic [11:0] abs_int;
  logic  [3:0] msb_index;
  logic  [3:0] norm_shift;
  logic [11:0] normalized_abs;
  logic  [6:0] fraction_bits;
  logic signed [8:0] biased_exp;

  always_comb begin
    is_zero        = (int12_i == 12'sd0);
    sign_bit       = int12_i[11];
    abs_ext        = sign_bit ? $unsigned(-$signed({int12_i[11], int12_i})) : $unsigned({1'b0, int12_i});
    abs_int        = abs_ext[11:0];
    msb_index      = '0;
    norm_shift     = '0;
    normalized_abs = '0;
    fraction_bits  = '0;
    biased_exp     = '0;
    bf16_o         = 16'h0000;

    if (!is_zero) begin
      unique casez (abs_int)
        12'b1???_????_????: msb_index = 4'd11;
        12'b01??_????_????: msb_index = 4'd10;
        12'b001?_????_????: msb_index = 4'd9;
        12'b0001_????_????: msb_index = 4'd8;
        12'b0000_1???_????: msb_index = 4'd7;
        12'b0000_01??_????: msb_index = 4'd6;
        12'b0000_001?_????: msb_index = 4'd5;
        12'b0000_0001_????: msb_index = 4'd4;
        12'b0000_0000_1???: msb_index = 4'd3;
        12'b0000_0000_01??: msb_index = 4'd2;
        12'b0000_0000_001?: msb_index = 4'd1;
        12'b0000_0000_0001: msb_index = 4'd0;
        default:            msb_index = 4'd0;
      endcase

      norm_shift     = 4'd11 - msb_index;
      normalized_abs = abs_int << norm_shift;
      fraction_bits  = normalized_abs[10:4];
      biased_exp     = $signed({1'b0, msb_index}) + $signed(scaling_factor_i) + 9'sd127;
      bf16_o         = {sign_bit, biased_exp[7:0], fraction_bits};
    end
  end

endmodule
