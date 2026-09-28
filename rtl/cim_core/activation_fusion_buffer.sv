// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module activation_fusion_buffer (
  input  logic        clk_i,
  input  logic        clear_i,
  input  logic        slots_clear_i,
  input  logic        shift_refill_i,
  input  logic [31:0] refill_activation_i,
  input  logic signed [4:0] refill_activation_sf_i,
  input  logic  [7:0] refill_mask_i,
  output logic [31:0] fused_activation_vec_o,
  output logic signed [4:0] fused_activation_sf_o,
  output logic [15:0] fused_row_weight_rsel_o
);

  logic [31:0]        slot_activation_q    [0:3];
  logic signed  [4:0] slot_activation_sf_q [0:3];
  logic  [7:0]        slot_mask_q          [0:3];

  logic               lane_valid_d         [0:7];
  logic         [1:0] lane_slot_sel_d      [0:7];
  logic signed  [4:0] lane_selected_sf_d   [0:7];
  logic signed  [5:0] lane_sf_delta_d      [0:7];
  logic         [4:0] lane_shift_amt_d     [0:7];
  logic signed  [3:0] lane_nibble_d        [0:7];
  logic signed  [3:0] lane_aligned_d       [0:7];
  logic               any_valid_d;
  logic signed  [4:0] fused_sf_d;

  integer idx;
  integer lane_idx;
  integer slot_idx;
  logic lane_found;

  always_ff @(posedge clk_i) begin
    if (clear_i || slots_clear_i) begin
      for (idx = 0; idx < 4; idx++) begin
        slot_activation_q[idx]    <= '0;
        slot_activation_sf_q[idx] <= '0;
        slot_mask_q[idx]          <= 8'hFF;
      end
    end else if (shift_refill_i) begin
      slot_activation_q[0]    <= slot_activation_q[1];
      slot_activation_q[1]    <= slot_activation_q[2];
      slot_activation_q[2]    <= slot_activation_q[3];
      slot_activation_q[3]    <= refill_activation_i;

      slot_activation_sf_q[0] <= slot_activation_sf_q[1];
      slot_activation_sf_q[1] <= slot_activation_sf_q[2];
      slot_activation_sf_q[2] <= slot_activation_sf_q[3];
      slot_activation_sf_q[3] <= refill_activation_sf_i;

      slot_mask_q[0]          <= slot_mask_q[1];
      slot_mask_q[1]          <= slot_mask_q[2];
      slot_mask_q[2]          <= slot_mask_q[3];
      slot_mask_q[3]          <= refill_mask_i;
    end
  end

  always_comb begin
    for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
      lane_valid_d[lane_idx]       = 1'b0;
      lane_slot_sel_d[lane_idx]    = 2'd0;
      lane_selected_sf_d[lane_idx] = 5'd0;
      lane_sf_delta_d[lane_idx]    = 6'sd0;
      lane_shift_amt_d[lane_idx]   = 5'd0;
      lane_nibble_d[lane_idx]      = 4'sd0;
      lane_aligned_d[lane_idx]     = 4'sd0;

      lane_found = 1'b0;
      for (slot_idx = 0; slot_idx < 4; slot_idx++) begin
        if (!lane_found && (slot_mask_q[slot_idx][lane_idx] == 1'b0)) begin
          lane_valid_d[lane_idx]       = 1'b1;
          lane_slot_sel_d[lane_idx]    = slot_idx[1:0];
          lane_selected_sf_d[lane_idx] = slot_activation_sf_q[slot_idx];
          lane_nibble_d[lane_idx]      = $signed(slot_activation_q[slot_idx][(lane_idx * 4) +: 4]);
          lane_found                   = 1'b1;
        end
      end
    end

    any_valid_d = 1'b0;
    fused_sf_d  = 5'sd0;
    for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
      if (lane_valid_d[lane_idx] && (!any_valid_d || (lane_selected_sf_d[lane_idx] > fused_sf_d))) begin
        any_valid_d = 1'b1;
        fused_sf_d  = lane_selected_sf_d[lane_idx];
      end
    end

    fused_activation_vec_o    = 32'd0;
    fused_row_weight_rsel_o   = 16'd0;
    fused_activation_sf_o     = fused_sf_d;

    for (lane_idx = 0; lane_idx < 8; lane_idx++) begin
      if (lane_valid_d[lane_idx]) begin
        lane_sf_delta_d[lane_idx] = $signed({fused_sf_d[4], fused_sf_d})
                                  - $signed({lane_selected_sf_d[lane_idx][4], lane_selected_sf_d[lane_idx]});
        if (lane_sf_delta_d[lane_idx][5]) begin
          lane_shift_amt_d[lane_idx] = 5'd0;
        end else begin
          lane_shift_amt_d[lane_idx] = lane_sf_delta_d[lane_idx][4:0];
        end
        lane_aligned_d[lane_idx] = lane_nibble_d[lane_idx] >>> lane_shift_amt_d[lane_idx];
        fused_row_weight_rsel_o[(lane_idx * 2) +: 2] = lane_slot_sel_d[lane_idx];
      end else begin
        lane_sf_delta_d[lane_idx] = 6'sd0;
        lane_shift_amt_d[lane_idx] = 5'd0;
        lane_aligned_d[lane_idx]   = 4'sd0;
        fused_row_weight_rsel_o[(lane_idx * 2) +: 2] = 2'd0;
      end
      fused_activation_vec_o[(lane_idx * 4) +: 4] = lane_aligned_d[lane_idx][3:0];
    end
  end

endmodule
