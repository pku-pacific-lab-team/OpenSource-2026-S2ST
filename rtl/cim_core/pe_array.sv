// Copyright 2026 School of Integrated Circuits, Peking University
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module pe_array (
  input  logic        clk_i,
  input  logic        weight_we_i,
  input  logic  [4:0] weight_addr_i,
  input  logic [31:0] weight_row_i,
  input  logic [31:0] activation_vec_i,
  input  logic [15:0] row_weight_rsel_i,
  output logic [87:0] result_vec_o
);

  // 8x8 PE array. Each row shares one activation lane and one weight-slot select.
  // Weight writes target one row and one of the four internal slots per PE.
  logic signed [3:0]  activation_vec   [0:7];
  logic signed [3:0]  weight_row       [0:7];
  logic        [1:0]  row_weight_rsel  [0:7];
  logic               row_weight_we    [0:7];
  logic signed [7:0]  pe_product       [0:7][0:7];
  logic signed [10:0] col_sum_l1       [0:7][0:3];
  logic signed [10:0] col_sum_l2       [0:7][0:1];
  logic signed [10:0] col_sum          [0:7];

  genvar row_idx;
  generate
    for (row_idx = 0; row_idx < 8; row_idx++) begin : gen_input_unpack
      localparam logic [2:0] ROW_SEL = row_idx;

      assign activation_vec[row_idx] = $signed(activation_vec_i[row_idx * 4 +: 4]);
      assign row_weight_rsel[row_idx] = row_weight_rsel_i[row_idx * 2 +: 2];
      assign row_weight_we[row_idx] = weight_we_i && (weight_addr_i[4:2] == ROW_SEL);
    end
  endgenerate

  genvar col_idx;
  generate
    for (col_idx = 0; col_idx < 8; col_idx++) begin : gen_weight_unpack
      assign weight_row[col_idx] = $signed(weight_row_i[col_idx * 4 +: 4]);
    end
  endgenerate

  generate
    for (row_idx = 0; row_idx < 8; row_idx++) begin : gen_rows
      for (col_idx = 0; col_idx < 8; col_idx++) begin : gen_cols
        signed_4bit_pe pe_u (
          .clk_i        (clk_i),
          .activation_i (activation_vec[row_idx]),
          .weight_we_i  (row_weight_we[row_idx]),
          .weight_wsel_i(weight_addr_i[1:0]),
          .weight_d_i   (weight_row[col_idx]),
          .weight_rsel_i(row_weight_rsel[row_idx]),
          .product_o    (pe_product[row_idx][col_idx])
        );
      end
    end
  endgenerate

  generate
    for (col_idx = 0; col_idx < 8; col_idx++) begin : gen_column_reduce
      assign col_sum_l1[col_idx][0] = $signed({{3{pe_product[0][col_idx][7]}}, pe_product[0][col_idx]})
        + $signed({{3{pe_product[1][col_idx][7]}}, pe_product[1][col_idx]});
      assign col_sum_l1[col_idx][1] = $signed({{3{pe_product[2][col_idx][7]}}, pe_product[2][col_idx]})
        + $signed({{3{pe_product[3][col_idx][7]}}, pe_product[3][col_idx]});
      assign col_sum_l1[col_idx][2] = $signed({{3{pe_product[4][col_idx][7]}}, pe_product[4][col_idx]})
        + $signed({{3{pe_product[5][col_idx][7]}}, pe_product[5][col_idx]});
      assign col_sum_l1[col_idx][3] = $signed({{3{pe_product[6][col_idx][7]}}, pe_product[6][col_idx]})
        + $signed({{3{pe_product[7][col_idx][7]}}, pe_product[7][col_idx]});

      assign col_sum_l2[col_idx][0] = col_sum_l1[col_idx][0] + col_sum_l1[col_idx][1];
      assign col_sum_l2[col_idx][1] = col_sum_l1[col_idx][2] + col_sum_l1[col_idx][3];

      assign col_sum[col_idx] = col_sum_l2[col_idx][0] + col_sum_l2[col_idx][1];
      assign result_vec_o[col_idx * 11 +: 11] = col_sum[col_idx];
    end
  endgenerate

endmodule
