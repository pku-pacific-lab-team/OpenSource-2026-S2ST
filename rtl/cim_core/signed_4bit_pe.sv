// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module signed_4bit_pe (
  input  logic              clk_i,
  input  logic signed [3:0] activation_i,
  input  logic              weight_we_i,
  input  logic        [1:0] weight_wsel_i,
  input  logic signed [3:0] weight_d_i,
  input  logic        [1:0] weight_rsel_i,
  output logic signed [7:0] product_o
);

  logic [3:0] weight_slot_d [0:3];
  logic [3:0] weight_slot_q [0:3];
  logic signed [3:0] selected_weight;

  genvar idx;
  generate
    for (idx = 0; idx < 4; idx++) begin : gen_weight_slots
      localparam logic [1:0] SLOT_SEL = idx;

      assign weight_slot_d[idx] = (weight_we_i && (weight_wsel_i == SLOT_SEL))
        ? weight_d_i
        : weight_slot_q[idx];

      packed_4bit_register weight_reg (
        .clk_i (clk_i),
        .d_i   (weight_slot_d[idx]),
        .q_o   (weight_slot_q[idx])
      );
    end
  endgenerate

  always_comb begin
    case (weight_rsel_i)
      2'd0: selected_weight = $signed(weight_slot_q[0]);
      2'd1: selected_weight = $signed(weight_slot_q[1]);
      2'd2: selected_weight = $signed(weight_slot_q[2]);
      2'd3: selected_weight = $signed(weight_slot_q[3]);
      default: selected_weight = 'x;
    endcase
  end

  assign product_o = activation_i * selected_weight;

endmodule
