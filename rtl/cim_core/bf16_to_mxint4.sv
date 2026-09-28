// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

// Module purpose:
// Quantize 8 packed BF16 lanes into 8 packed signed MXINT4 lanes with one shared
// signed scaling factor derived from the maximum BF16 exponent field.
//
// Key I/O semantics:
// - Lane packing:
//   - `bf16_vec_i[lane_idx * 16 +: 16]` is BF16 lane `lane_idx`
//   - `mxint4_vec_o[lane_idx *  4 +:  4]` is MXINT4 lane `lane_idx`
// - When `valid_i = 0`, outputs are forced to zero.
// - Input assumptions (per design spec):
//   - each lane is either +0/-0 or a normal finite BF16 value
//   - no NaN, Inf, or subnormal lanes are present
//
// Timing / latency:
// Pure combinational quantizer with truncation-toward-zero (via magnitude right
// shift). No rounding or saturation logic is implemented.

module bf16_to_mxint4 (
  input  logic         valid_i,
  input  logic [127:0] bf16_vec_i,
  output logic  [31:0] mxint4_vec_o,
  output logic signed [4:0] scaling_factor_o
);

  logic [15:0] bf16_lane  [8];
  logic        is_zero    [8];
  logic        sign_bit   [8];
  logic  [7:0] exp_field  [8];
  logic  [6:0] frac_field [8];
  logic  [7:0] mant       [8];

  logic        any_nonzero;
  logic  [7:0] max_exp_field;
  logic signed [8:0] scaling_factor_wide;

  logic  [7:0] exp_delta [8];
  logic  [8:0] shift_amt [8];
  logic  [7:0] mag_q     [8];
  logic signed [8:0] lane_q    [8];

  always_comb begin
    mxint4_vec_o         = 32'b0;
    scaling_factor_o     = 5'sd0;

    any_nonzero          = 1'b0;
    max_exp_field        = 8'h00;
    scaling_factor_wide  = 9'sd0;

    for (int unsigned lane_idx = 0; lane_idx < 8; lane_idx++) begin
      bf16_lane[lane_idx]   = 16'h0000;
      is_zero[lane_idx]     = 1'b1;
      sign_bit[lane_idx]    = 1'b0;
      exp_field[lane_idx]   = 8'h00;
      frac_field[lane_idx]  = 7'h00;
      mant[lane_idx]        = 8'h00;

      exp_delta[lane_idx]   = 8'h00;
      shift_amt[lane_idx]   = 9'h000;
      mag_q[lane_idx]       = 8'h00;
      lane_q[lane_idx]      = 9'sd0;
    end

    // Early return when invalid.
    if (!valid_i) begin
      // Outputs are already defaulted to zero.
    end else begin
      // 1) Input unpack and per-lane parse.
      for (int unsigned lane_idx = 0; lane_idx < 8; lane_idx++) begin
        bf16_lane[lane_idx]  = bf16_vec_i[lane_idx * 16 +: 16];
        is_zero[lane_idx]    = (bf16_lane[lane_idx][14:0] == 15'b0);
        sign_bit[lane_idx]   = bf16_lane[lane_idx][15];
        exp_field[lane_idx]  = bf16_lane[lane_idx][14:7];
        frac_field[lane_idx] = bf16_lane[lane_idx][6:0];
        mant[lane_idx]       = is_zero[lane_idx] ? 8'h00 : {1'b1, frac_field[lane_idx]};
      end

      // 2) Maximum exponent reduction across non-zero lanes.
      for (int unsigned lane_idx = 0; lane_idx < 8; lane_idx++) begin
        if (!is_zero[lane_idx]) begin
          any_nonzero = 1'b1;
          if (exp_field[lane_idx] > max_exp_field) begin
            max_exp_field = exp_field[lane_idx];
          end
        end
      end

      // 3) Shared scaling factor generation.
      if (!any_nonzero) begin
        scaling_factor_o = 5'sd0;
        // `mxint4_vec_o` remains all-zero.
      end else begin
        scaling_factor_wide = $signed({1'b0, max_exp_field}) - 9'sd129;
        scaling_factor_o    = scaling_factor_wide[4:0];

        // 4) Per-lane magnitude shift, sign application, and output pack.
        for (int unsigned lane_idx = 0; lane_idx < 8; lane_idx++) begin
          if (is_zero[lane_idx]) begin
            mxint4_vec_o[lane_idx * 4 +: 4] = 4'sd0;
          end else begin
            exp_delta[lane_idx] = max_exp_field - exp_field[lane_idx];
            shift_amt[lane_idx] = {1'b0, exp_delta[lane_idx]} + 9'd5;
            mag_q[lane_idx]     = mant[lane_idx] >> shift_amt[lane_idx];

            if (mag_q[lane_idx] == 8'h00) begin
              mxint4_vec_o[lane_idx * 4 +: 4] = 4'sd0;
            end else begin
              lane_q[lane_idx] = sign_bit[lane_idx]
                ? -$signed({1'b0, mag_q[lane_idx]})
                :  $signed({1'b0, mag_q[lane_idx]});
              mxint4_vec_o[lane_idx * 4 +: 4] = lane_q[lane_idx][3:0];
            end
          end
        end
      end
    end
  end

endmodule

