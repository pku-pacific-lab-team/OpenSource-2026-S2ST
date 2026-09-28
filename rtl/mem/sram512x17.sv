// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

`ifndef PROB_CACHE_SRAM512X17_SV
`define PROB_CACHE_SRAM512X17_SV

// Module purpose:
// Behavioral single-port SRAM model for a `512 x 17` macro with one-cycle
// synchronous reads and active-low `cen` / `wen`.
//
// Timing / latency:
// - `cen = 0, wen = 0`: write `d` into `mem[a]` on `posedge clk`
// - `cen = 0, wen = 1`: update `q <= mem[a]` on `posedge clk`
// - `cen = 1`: hold `q`

module sram512x17 (
  output logic [16:0] q,
  input  logic        clk,
  input  logic        cen,
  input  logic        wen,
  input  logic  [8:0] a,
  input  logic [16:0] d,
  input  logic  [2:0] ema,
  input  logic  [1:0] emaw,
  input  logic        emas,
  input  logic        ret1n,
  input  logic        rawl,
  input  logic  [1:0] rawlm,
  input  logic        wabl,
  input  logic  [1:0] wablm
);

  logic [16:0] mem [0:511];
  integer      mem_idx;

  initial begin
    for (mem_idx = 0; mem_idx < 512; mem_idx++) begin
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
