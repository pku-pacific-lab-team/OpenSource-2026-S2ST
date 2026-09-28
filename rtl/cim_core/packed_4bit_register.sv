// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module packed_4bit_register (
  input  logic       clk_i,
  input  logic [3:0] d_i,
  output logic [3:0] q_o
);

  // Minimal behavioral model for a fixed-width packed 4-bit register.
  logic [3:0] data_q;

  assign q_o = data_q;

  always_ff @(posedge clk_i) begin
    data_q <= d_i;
  end

endmodule
