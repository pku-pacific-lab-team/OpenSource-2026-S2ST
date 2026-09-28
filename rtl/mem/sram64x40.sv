// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

`ifndef PE_ARRAY_SRAM64X40_SV
`define PE_ARRAY_SRAM64X40_SV

// Module purpose:
// Behavioral single-port SRAM model for a `64 x 40` macro with one-cycle
// synchronous reads and active-low `cen` / `wen`.

module sram64x40 (
  output logic [39:0] q,
  input  logic        clk,
  input  logic        cen,
  input  logic        wen,
  input  logic  [5:0] a,
  input  logic [39:0] d,
  input  logic  [2:0] ema,
  input  logic  [1:0] emaw,
  input  logic        emas,
  input  logic        ret1n,
  input  logic        rawl,
  input  logic  [1:0] rawlm,
  input  logic        wabl,
  input  logic  [1:0] wablm
);

  logic [39:0] mem [0:63];
  integer      mem_idx;

  initial begin
    for (mem_idx = 0; mem_idx < 64; mem_idx++) begin
      mem[mem_idx] = '0;
    end
  end

  always_ff @(posedge clk) begin
    if (!cen) begin
      if (!wen) begin
        mem[a] <= d;
      end else begin
        q <= mem[a];
      end
    end
  end

endmodule

`endif
