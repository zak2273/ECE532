// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.3
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module dense_array_ap_fixed_6u_array_ap_fixed_16_6_5_3_0_16u_config15_s (
        ap_clk,
        ap_rst,
        ap_start,
        start_full_n,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        start_out,
        start_write,
        data_stream_V_data_0_V_dout,
        data_stream_V_data_0_V_empty_n,
        data_stream_V_data_0_V_read,
        data_stream_V_data_1_V_dout,
        data_stream_V_data_1_V_empty_n,
        data_stream_V_data_1_V_read,
        data_stream_V_data_2_V_dout,
        data_stream_V_data_2_V_empty_n,
        data_stream_V_data_2_V_read,
        data_stream_V_data_3_V_dout,
        data_stream_V_data_3_V_empty_n,
        data_stream_V_data_3_V_read,
        data_stream_V_data_4_V_dout,
        data_stream_V_data_4_V_empty_n,
        data_stream_V_data_4_V_read,
        data_stream_V_data_5_V_dout,
        data_stream_V_data_5_V_empty_n,
        data_stream_V_data_5_V_read,
        res_stream_V_data_0_V_din,
        res_stream_V_data_0_V_full_n,
        res_stream_V_data_0_V_write,
        res_stream_V_data_1_V_din,
        res_stream_V_data_1_V_full_n,
        res_stream_V_data_1_V_write,
        res_stream_V_data_2_V_din,
        res_stream_V_data_2_V_full_n,
        res_stream_V_data_2_V_write,
        res_stream_V_data_3_V_din,
        res_stream_V_data_3_V_full_n,
        res_stream_V_data_3_V_write,
        res_stream_V_data_4_V_din,
        res_stream_V_data_4_V_full_n,
        res_stream_V_data_4_V_write,
        res_stream_V_data_5_V_din,
        res_stream_V_data_5_V_full_n,
        res_stream_V_data_5_V_write,
        res_stream_V_data_6_V_din,
        res_stream_V_data_6_V_full_n,
        res_stream_V_data_6_V_write,
        res_stream_V_data_7_V_din,
        res_stream_V_data_7_V_full_n,
        res_stream_V_data_7_V_write,
        res_stream_V_data_8_V_din,
        res_stream_V_data_8_V_full_n,
        res_stream_V_data_8_V_write,
        res_stream_V_data_9_V_din,
        res_stream_V_data_9_V_full_n,
        res_stream_V_data_9_V_write,
        res_stream_V_data_10_V_din,
        res_stream_V_data_10_V_full_n,
        res_stream_V_data_10_V_write,
        res_stream_V_data_11_V_din,
        res_stream_V_data_11_V_full_n,
        res_stream_V_data_11_V_write,
        res_stream_V_data_12_V_din,
        res_stream_V_data_12_V_full_n,
        res_stream_V_data_12_V_write,
        res_stream_V_data_13_V_din,
        res_stream_V_data_13_V_full_n,
        res_stream_V_data_13_V_write,
        res_stream_V_data_14_V_din,
        res_stream_V_data_14_V_full_n,
        res_stream_V_data_14_V_write,
        res_stream_V_data_15_V_din,
        res_stream_V_data_15_V_full_n,
        res_stream_V_data_15_V_write
);

parameter    ap_ST_fsm_state1 = 13'd1;
parameter    ap_ST_fsm_pp0_stage0 = 13'd2;
parameter    ap_ST_fsm_state4 = 13'd4;
parameter    ap_ST_fsm_state5 = 13'd8;
parameter    ap_ST_fsm_state6 = 13'd16;
parameter    ap_ST_fsm_state7 = 13'd32;
parameter    ap_ST_fsm_state8 = 13'd64;
parameter    ap_ST_fsm_state9 = 13'd128;
parameter    ap_ST_fsm_state10 = 13'd256;
parameter    ap_ST_fsm_state11 = 13'd512;
parameter    ap_ST_fsm_state12 = 13'd1024;
parameter    ap_ST_fsm_state13 = 13'd2048;
parameter    ap_ST_fsm_state14 = 13'd4096;

input   ap_clk;
input   ap_rst;
input   ap_start;
input   start_full_n;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
output   start_out;
output   start_write;
input  [15:0] data_stream_V_data_0_V_dout;
input   data_stream_V_data_0_V_empty_n;
output   data_stream_V_data_0_V_read;
input  [15:0] data_stream_V_data_1_V_dout;
input   data_stream_V_data_1_V_empty_n;
output   data_stream_V_data_1_V_read;
input  [15:0] data_stream_V_data_2_V_dout;
input   data_stream_V_data_2_V_empty_n;
output   data_stream_V_data_2_V_read;
input  [15:0] data_stream_V_data_3_V_dout;
input   data_stream_V_data_3_V_empty_n;
output   data_stream_V_data_3_V_read;
input  [15:0] data_stream_V_data_4_V_dout;
input   data_stream_V_data_4_V_empty_n;
output   data_stream_V_data_4_V_read;
input  [15:0] data_stream_V_data_5_V_dout;
input   data_stream_V_data_5_V_empty_n;
output   data_stream_V_data_5_V_read;
output  [15:0] res_stream_V_data_0_V_din;
input   res_stream_V_data_0_V_full_n;
output   res_stream_V_data_0_V_write;
output  [15:0] res_stream_V_data_1_V_din;
input   res_stream_V_data_1_V_full_n;
output   res_stream_V_data_1_V_write;
output  [15:0] res_stream_V_data_2_V_din;
input   res_stream_V_data_2_V_full_n;
output   res_stream_V_data_2_V_write;
output  [15:0] res_stream_V_data_3_V_din;
input   res_stream_V_data_3_V_full_n;
output   res_stream_V_data_3_V_write;
output  [15:0] res_stream_V_data_4_V_din;
input   res_stream_V_data_4_V_full_n;
output   res_stream_V_data_4_V_write;
output  [15:0] res_stream_V_data_5_V_din;
input   res_stream_V_data_5_V_full_n;
output   res_stream_V_data_5_V_write;
output  [15:0] res_stream_V_data_6_V_din;
input   res_stream_V_data_6_V_full_n;
output   res_stream_V_data_6_V_write;
output  [15:0] res_stream_V_data_7_V_din;
input   res_stream_V_data_7_V_full_n;
output   res_stream_V_data_7_V_write;
output  [15:0] res_stream_V_data_8_V_din;
input   res_stream_V_data_8_V_full_n;
output   res_stream_V_data_8_V_write;
output  [15:0] res_stream_V_data_9_V_din;
input   res_stream_V_data_9_V_full_n;
output   res_stream_V_data_9_V_write;
output  [15:0] res_stream_V_data_10_V_din;
input   res_stream_V_data_10_V_full_n;
output   res_stream_V_data_10_V_write;
output  [15:0] res_stream_V_data_11_V_din;
input   res_stream_V_data_11_V_full_n;
output   res_stream_V_data_11_V_write;
output  [15:0] res_stream_V_data_12_V_din;
input   res_stream_V_data_12_V_full_n;
output   res_stream_V_data_12_V_write;
output  [15:0] res_stream_V_data_13_V_din;
input   res_stream_V_data_13_V_full_n;
output   res_stream_V_data_13_V_write;
output  [15:0] res_stream_V_data_14_V_din;
input   res_stream_V_data_14_V_full_n;
output   res_stream_V_data_14_V_write;
output  [15:0] res_stream_V_data_15_V_din;
input   res_stream_V_data_15_V_full_n;
output   res_stream_V_data_15_V_write;

reg ap_done;
reg ap_idle;
reg start_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [12:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
reg    data_stream_V_data_0_V_blk_n;
wire    ap_CS_fsm_pp0_stage0;
reg    ap_enable_reg_pp0_iter1;
wire    ap_block_pp0_stage0;
reg   [0:0] tmp_reg_843;
reg    data_stream_V_data_1_V_blk_n;
reg    data_stream_V_data_2_V_blk_n;
reg    data_stream_V_data_3_V_blk_n;
reg    data_stream_V_data_4_V_blk_n;
reg    data_stream_V_data_5_V_blk_n;
reg    res_stream_V_data_0_V_blk_n;
wire    ap_CS_fsm_state14;
reg    res_stream_V_data_1_V_blk_n;
reg    res_stream_V_data_2_V_blk_n;
reg    res_stream_V_data_3_V_blk_n;
reg    res_stream_V_data_4_V_blk_n;
reg    res_stream_V_data_5_V_blk_n;
reg    res_stream_V_data_6_V_blk_n;
reg    res_stream_V_data_7_V_blk_n;
reg    res_stream_V_data_8_V_blk_n;
reg    res_stream_V_data_9_V_blk_n;
reg    res_stream_V_data_10_V_blk_n;
reg    res_stream_V_data_11_V_blk_n;
reg    res_stream_V_data_12_V_blk_n;
reg    res_stream_V_data_13_V_blk_n;
reg    res_stream_V_data_14_V_blk_n;
reg    res_stream_V_data_15_V_blk_n;
reg   [15:0] data_V_11_s_reg_430;
reg   [15:0] data_V_10_s_reg_442;
reg   [15:0] data_V_9_s_reg_454;
reg   [15:0] data_V_8_s_reg_466;
reg   [15:0] data_V_7_s_reg_478;
reg   [15:0] data_V_6_s_reg_490;
reg   [15:0] data_V_5_s_reg_502;
reg   [15:0] data_V_4_s_reg_514;
reg   [15:0] data_V_3_s_reg_526;
reg   [15:0] data_V_2_s_reg_538;
reg   [15:0] data_V_1_s_reg_550;
reg   [15:0] data_V_0_s_reg_562;
reg   [1:0] i_in_reg_574;
wire   [0:0] tmp_fu_613_p2;
wire    ap_block_state2_pp0_stage0_iter0;
wire    data_stream_V_data_0_V0_status;
reg    ap_block_state3_pp0_stage0_iter1;
reg    ap_block_pp0_stage0_11001;
wire   [1:0] i_in_1_fu_619_p2;
reg    ap_enable_reg_pp0_iter0;
wire   [0:0] cond_fu_653_p2;
reg   [0:0] cond_reg_852;
wire   [0:0] cond9_fu_665_p2;
reg   [0:0] cond9_reg_866;
wire   [15:0] data_6_V_1_fu_695_p3;
wire   [15:0] data_6_V_2_fu_702_p3;
wire   [15:0] data_7_V_1_fu_709_p3;
wire   [15:0] data_7_V_2_fu_716_p3;
wire   [15:0] data_8_V_1_fu_723_p3;
wire   [15:0] data_8_V_2_fu_730_p3;
wire   [15:0] data_9_V_1_fu_737_p3;
wire   [15:0] data_9_V_2_fu_744_p3;
wire   [15:0] data_10_V_1_fu_751_p3;
wire   [15:0] data_10_V_2_fu_758_p3;
wire   [15:0] data_11_V_1_fu_765_p3;
wire   [15:0] data_11_V_2_fu_772_p3;
reg   [15:0] tmp_data_0_V_reg_932;
wire    ap_CS_fsm_state13;
reg   [15:0] tmp_data_1_V_reg_937;
reg   [15:0] tmp_data_2_V_reg_942;
reg   [15:0] tmp_data_3_V_reg_947;
reg   [15:0] tmp_data_4_V_reg_952;
reg   [15:0] tmp_data_5_V_reg_957;
reg   [15:0] tmp_data_6_V_reg_962;
reg   [15:0] tmp_data_7_V_reg_967;
reg   [15:0] tmp_data_8_V_reg_972;
reg   [15:0] tmp_data_9_V_reg_977;
reg   [15:0] tmp_data_10_V_reg_982;
reg   [15:0] tmp_data_11_V_reg_987;
reg   [15:0] tmp_data_12_V_reg_992;
reg   [15:0] tmp_data_13_V_reg_997;
reg   [15:0] tmp_data_14_V_reg_1002;
reg   [15:0] tmp_data_15_V_reg_1007;
reg    ap_block_state1;
reg    ap_block_pp0_stage0_subdone;
reg    ap_condition_pp0_exit_iter0_state2;
wire    grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_start;
wire    grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_done;
wire    grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_idle;
wire    grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_ready;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_0;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_1;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_2;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_3;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_4;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_5;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_6;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_7;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_8;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_9;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_10;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_11;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_12;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_13;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_14;
wire   [15:0] grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_15;
reg    grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_start_reg;
reg   [12:0] ap_NS_fsm;
wire    ap_NS_fsm_state4;
wire    ap_CS_fsm_state4;
reg    data_stream_V_data_0_V0_update;
reg    res_stream_V_data_0_V1_update;
wire    res_stream_V_data_0_V1_status;
wire   [0:0] tmp_456_fu_625_p1;
wire   [1:0] tmp_457_fu_637_p2;
wire   [3:0] p_shl_fu_629_p3;
wire   [3:0] p_shl19_cast_fu_643_p1;
wire   [3:0] tmp_s_fu_647_p2;
wire   [3:0] tmp_346_122_t_fu_659_p2;
reg    ap_idle_pp0;
wire    ap_enable_pp0;

// power-on initialization
initial begin
#0 start_once_reg = 1'b0;
#0 ap_done_reg = 1'b0;
#0 ap_CS_fsm = 13'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter0 = 1'b0;
#0 grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_start_reg = 1'b0;
end

dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_start),
    .ap_done(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_done),
    .ap_idle(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_idle),
    .ap_ready(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_ready),
    .data_0_V_read(data_V_0_s_reg_562),
    .data_1_V_read(data_V_1_s_reg_550),
    .data_2_V_read(data_V_2_s_reg_538),
    .data_3_V_read(data_V_3_s_reg_526),
    .data_4_V_read(data_V_4_s_reg_514),
    .data_5_V_read(data_V_5_s_reg_502),
    .data_6_V_read(data_V_6_s_reg_490),
    .data_7_V_read(data_V_7_s_reg_478),
    .data_8_V_read(data_V_8_s_reg_466),
    .data_9_V_read(data_V_9_s_reg_454),
    .data_10_V_read(data_V_10_s_reg_442),
    .data_11_V_read(data_V_11_s_reg_430),
    .ap_return_0(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_0),
    .ap_return_1(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_1),
    .ap_return_2(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_2),
    .ap_return_3(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_3),
    .ap_return_4(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_4),
    .ap_return_5(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_5),
    .ap_return_6(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_6),
    .ap_return_7(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_7),
    .ap_return_8(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_8),
    .ap_return_9(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_9),
    .ap_return_10(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_10),
    .ap_return_11(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_11),
    .ap_return_12(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_12),
    .ap_return_13(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_13),
    .ap_return_14(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_14),
    .ap_return_15(grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_15)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((res_stream_V_data_0_V1_status == 1'b1) & (1'b1 == ap_CS_fsm_state14))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter0 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_condition_pp0_exit_iter0_state2) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_enable_reg_pp0_iter0 <= 1'b0;
        end else if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter0 <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_condition_pp0_exit_iter0_state2))) begin
            ap_enable_reg_pp0_iter1 <= (1'b1 ^ ap_condition_pp0_exit_iter0_state2);
        end else if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
        end else if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_NS_fsm_state4) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
            grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_start_reg <= 1'b1;
        end else if ((grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_ready == 1'b1)) begin
            grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        start_once_reg <= 1'b0;
    end else begin
        if (((internal_ap_ready == 1'b0) & (real_start == 1'b1))) begin
            start_once_reg <= 1'b1;
        end else if ((internal_ap_ready == 1'b1)) begin
            start_once_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_fu_613_p2 == 1'd0))) begin
        i_in_reg_574 <= i_in_1_fu_619_p2;
    end else if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        i_in_reg_574 <= 2'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_fu_613_p2 == 1'd0))) begin
        cond9_reg_866 <= cond9_fu_665_p2;
        cond_reg_852 <= cond_fu_653_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_843 == 1'd0))) begin
        data_V_0_s_reg_562 <= data_6_V_2_fu_702_p3;
        data_V_10_s_reg_442 <= data_10_V_1_fu_751_p3;
        data_V_11_s_reg_430 <= data_11_V_1_fu_765_p3;
        data_V_1_s_reg_550 <= data_7_V_2_fu_716_p3;
        data_V_2_s_reg_538 <= data_8_V_2_fu_730_p3;
        data_V_3_s_reg_526 <= data_9_V_2_fu_744_p3;
        data_V_4_s_reg_514 <= data_10_V_2_fu_758_p3;
        data_V_5_s_reg_502 <= data_11_V_2_fu_772_p3;
        data_V_6_s_reg_490 <= data_6_V_1_fu_695_p3;
        data_V_7_s_reg_478 <= data_7_V_1_fu_709_p3;
        data_V_8_s_reg_466 <= data_8_V_1_fu_723_p3;
        data_V_9_s_reg_454 <= data_9_V_1_fu_737_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state13)) begin
        tmp_data_0_V_reg_932 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_0;
        tmp_data_10_V_reg_982 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_10;
        tmp_data_11_V_reg_987 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_11;
        tmp_data_12_V_reg_992 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_12;
        tmp_data_13_V_reg_997 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_13;
        tmp_data_14_V_reg_1002 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_14;
        tmp_data_15_V_reg_1007 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_15;
        tmp_data_1_V_reg_937 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_1;
        tmp_data_2_V_reg_942 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_2;
        tmp_data_3_V_reg_947 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_3;
        tmp_data_4_V_reg_952 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_4;
        tmp_data_5_V_reg_957 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_5;
        tmp_data_6_V_reg_962 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_6;
        tmp_data_7_V_reg_967 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_7;
        tmp_data_8_V_reg_972 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_8;
        tmp_data_9_V_reg_977 <= grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_return_9;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        tmp_reg_843 <= tmp_fu_613_p2;
    end
end

always @ (*) begin
    if ((tmp_fu_613_p2 == 1'd1)) begin
        ap_condition_pp0_exit_iter0_state2 = 1'b1;
    end else begin
        ap_condition_pp0_exit_iter0_state2 = 1'b0;
    end
end

always @ (*) begin
    if (((res_stream_V_data_0_V1_status == 1'b1) & (1'b1 == ap_CS_fsm_state14))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
    end
end

always @ (*) begin
    if (((real_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_843 == 1'd0))) begin
        data_stream_V_data_0_V0_update = 1'b1;
    end else begin
        data_stream_V_data_0_V0_update = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_843 == 1'd0) & (1'b0 == ap_block_pp0_stage0))) begin
        data_stream_V_data_0_V_blk_n = data_stream_V_data_0_V_empty_n;
    end else begin
        data_stream_V_data_0_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_843 == 1'd0) & (1'b0 == ap_block_pp0_stage0))) begin
        data_stream_V_data_1_V_blk_n = data_stream_V_data_1_V_empty_n;
    end else begin
        data_stream_V_data_1_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_843 == 1'd0) & (1'b0 == ap_block_pp0_stage0))) begin
        data_stream_V_data_2_V_blk_n = data_stream_V_data_2_V_empty_n;
    end else begin
        data_stream_V_data_2_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_843 == 1'd0) & (1'b0 == ap_block_pp0_stage0))) begin
        data_stream_V_data_3_V_blk_n = data_stream_V_data_3_V_empty_n;
    end else begin
        data_stream_V_data_3_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_843 == 1'd0) & (1'b0 == ap_block_pp0_stage0))) begin
        data_stream_V_data_4_V_blk_n = data_stream_V_data_4_V_empty_n;
    end else begin
        data_stream_V_data_4_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_843 == 1'd0) & (1'b0 == ap_block_pp0_stage0))) begin
        data_stream_V_data_5_V_blk_n = data_stream_V_data_5_V_empty_n;
    end else begin
        data_stream_V_data_5_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((res_stream_V_data_0_V1_status == 1'b1) & (1'b1 == ap_CS_fsm_state14))) begin
        internal_ap_ready = 1'b1;
    end else begin
        internal_ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((start_once_reg == 1'b0) & (start_full_n == 1'b0))) begin
        real_start = 1'b0;
    end else begin
        real_start = ap_start;
    end
end

always @ (*) begin
    if (((res_stream_V_data_0_V1_status == 1'b1) & (1'b1 == ap_CS_fsm_state14))) begin
        res_stream_V_data_0_V1_update = 1'b1;
    end else begin
        res_stream_V_data_0_V1_update = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_0_V_blk_n = res_stream_V_data_0_V_full_n;
    end else begin
        res_stream_V_data_0_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_10_V_blk_n = res_stream_V_data_10_V_full_n;
    end else begin
        res_stream_V_data_10_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_11_V_blk_n = res_stream_V_data_11_V_full_n;
    end else begin
        res_stream_V_data_11_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_12_V_blk_n = res_stream_V_data_12_V_full_n;
    end else begin
        res_stream_V_data_12_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_13_V_blk_n = res_stream_V_data_13_V_full_n;
    end else begin
        res_stream_V_data_13_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_14_V_blk_n = res_stream_V_data_14_V_full_n;
    end else begin
        res_stream_V_data_14_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_15_V_blk_n = res_stream_V_data_15_V_full_n;
    end else begin
        res_stream_V_data_15_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_1_V_blk_n = res_stream_V_data_1_V_full_n;
    end else begin
        res_stream_V_data_1_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_2_V_blk_n = res_stream_V_data_2_V_full_n;
    end else begin
        res_stream_V_data_2_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_3_V_blk_n = res_stream_V_data_3_V_full_n;
    end else begin
        res_stream_V_data_3_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_4_V_blk_n = res_stream_V_data_4_V_full_n;
    end else begin
        res_stream_V_data_4_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_5_V_blk_n = res_stream_V_data_5_V_full_n;
    end else begin
        res_stream_V_data_5_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_6_V_blk_n = res_stream_V_data_6_V_full_n;
    end else begin
        res_stream_V_data_6_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_7_V_blk_n = res_stream_V_data_7_V_full_n;
    end else begin
        res_stream_V_data_7_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_8_V_blk_n = res_stream_V_data_8_V_full_n;
    end else begin
        res_stream_V_data_8_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        res_stream_V_data_9_V_blk_n = res_stream_V_data_9_V_full_n;
    end else begin
        res_stream_V_data_9_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((start_once_reg == 1'b0) & (real_start == 1'b1))) begin
        start_write = 1'b1;
    end else begin
        start_write = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_pp0_stage0 : begin
            if (~((1'b0 == ap_block_pp0_stage0_subdone) & (tmp_fu_613_p2 == 1'd1) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else if (((1'b0 == ap_block_pp0_stage0_subdone) & (tmp_fu_613_p2 == 1'd1) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            ap_NS_fsm = ap_ST_fsm_state11;
        end
        ap_ST_fsm_state11 : begin
            ap_NS_fsm = ap_ST_fsm_state12;
        end
        ap_ST_fsm_state12 : begin
            ap_NS_fsm = ap_ST_fsm_state13;
        end
        ap_ST_fsm_state13 : begin
            ap_NS_fsm = ap_ST_fsm_state14;
        end
        ap_ST_fsm_state14 : begin
            if (((res_stream_V_data_0_V1_status == 1'b1) & (1'b1 == ap_CS_fsm_state14))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state14;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state13 = ap_CS_fsm[32'd11];

assign ap_CS_fsm_state14 = ap_CS_fsm[32'd12];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd2];

assign ap_NS_fsm_state4 = ap_NS_fsm[32'd2];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_11001 = ((data_stream_V_data_0_V0_status == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (tmp_reg_843 == 1'd0));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = ((data_stream_V_data_0_V0_status == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (tmp_reg_843 == 1'd0));
end

always @ (*) begin
    ap_block_state1 = ((ap_done_reg == 1'b1) | (real_start == 1'b0));
end

assign ap_block_state2_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state3_pp0_stage0_iter1 = ((data_stream_V_data_0_V0_status == 1'b0) & (tmp_reg_843 == 1'd0));
end

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_ready = internal_ap_ready;

assign cond9_fu_665_p2 = ((tmp_346_122_t_fu_659_p2 == 4'd1) ? 1'b1 : 1'b0);

assign cond_fu_653_p2 = ((p_shl_fu_629_p3 == p_shl19_cast_fu_643_p1) ? 1'b1 : 1'b0);

assign data_10_V_1_fu_751_p3 = ((cond_reg_852[0:0] === 1'b1) ? data_V_10_s_reg_442 : data_stream_V_data_4_V_dout);

assign data_10_V_2_fu_758_p3 = ((cond_reg_852[0:0] === 1'b1) ? data_stream_V_data_4_V_dout : data_V_4_s_reg_514);

assign data_11_V_1_fu_765_p3 = ((cond_reg_852[0:0] === 1'b1) ? data_V_11_s_reg_430 : data_stream_V_data_5_V_dout);

assign data_11_V_2_fu_772_p3 = ((cond_reg_852[0:0] === 1'b1) ? data_stream_V_data_5_V_dout : data_V_5_s_reg_502);

assign data_6_V_1_fu_695_p3 = ((cond_reg_852[0:0] === 1'b1) ? data_V_6_s_reg_490 : data_stream_V_data_0_V_dout);

assign data_6_V_2_fu_702_p3 = ((cond_reg_852[0:0] === 1'b1) ? data_stream_V_data_0_V_dout : data_V_0_s_reg_562);

assign data_7_V_1_fu_709_p3 = ((cond9_reg_866[0:0] === 1'b1) ? data_V_7_s_reg_478 : data_stream_V_data_1_V_dout);

assign data_7_V_2_fu_716_p3 = ((cond9_reg_866[0:0] === 1'b1) ? data_stream_V_data_1_V_dout : data_V_1_s_reg_550);

assign data_8_V_1_fu_723_p3 = ((cond_reg_852[0:0] === 1'b1) ? data_V_8_s_reg_466 : data_stream_V_data_2_V_dout);

assign data_8_V_2_fu_730_p3 = ((cond_reg_852[0:0] === 1'b1) ? data_stream_V_data_2_V_dout : data_V_2_s_reg_538);

assign data_9_V_1_fu_737_p3 = ((cond_reg_852[0:0] === 1'b1) ? data_V_9_s_reg_454 : data_stream_V_data_3_V_dout);

assign data_9_V_2_fu_744_p3 = ((cond_reg_852[0:0] === 1'b1) ? data_stream_V_data_3_V_dout : data_V_3_s_reg_526);

assign data_stream_V_data_0_V0_status = (data_stream_V_data_5_V_empty_n & data_stream_V_data_4_V_empty_n & data_stream_V_data_3_V_empty_n & data_stream_V_data_2_V_empty_n & data_stream_V_data_1_V_empty_n & data_stream_V_data_0_V_empty_n);

assign data_stream_V_data_0_V_read = data_stream_V_data_0_V0_update;

assign data_stream_V_data_1_V_read = data_stream_V_data_0_V0_update;

assign data_stream_V_data_2_V_read = data_stream_V_data_0_V0_update;

assign data_stream_V_data_3_V_read = data_stream_V_data_0_V0_update;

assign data_stream_V_data_4_V_read = data_stream_V_data_0_V0_update;

assign data_stream_V_data_5_V_read = data_stream_V_data_0_V0_update;

assign grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_start = grp_dense_wrapper_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config15_s_fu_585_ap_start_reg;

assign i_in_1_fu_619_p2 = (i_in_reg_574 + 2'd1);

assign p_shl19_cast_fu_643_p1 = tmp_457_fu_637_p2;

assign p_shl_fu_629_p3 = {{tmp_456_fu_625_p1}, {3'd0}};

assign res_stream_V_data_0_V1_status = (res_stream_V_data_9_V_full_n & res_stream_V_data_8_V_full_n & res_stream_V_data_7_V_full_n & res_stream_V_data_6_V_full_n & res_stream_V_data_5_V_full_n & res_stream_V_data_4_V_full_n & res_stream_V_data_3_V_full_n & res_stream_V_data_2_V_full_n & res_stream_V_data_1_V_full_n & res_stream_V_data_15_V_full_n & res_stream_V_data_14_V_full_n & res_stream_V_data_13_V_full_n & res_stream_V_data_12_V_full_n & res_stream_V_data_11_V_full_n & res_stream_V_data_10_V_full_n & res_stream_V_data_0_V_full_n);

assign res_stream_V_data_0_V_din = tmp_data_0_V_reg_932;

assign res_stream_V_data_0_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_10_V_din = tmp_data_10_V_reg_982;

assign res_stream_V_data_10_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_11_V_din = tmp_data_11_V_reg_987;

assign res_stream_V_data_11_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_12_V_din = tmp_data_12_V_reg_992;

assign res_stream_V_data_12_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_13_V_din = tmp_data_13_V_reg_997;

assign res_stream_V_data_13_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_14_V_din = tmp_data_14_V_reg_1002;

assign res_stream_V_data_14_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_15_V_din = tmp_data_15_V_reg_1007;

assign res_stream_V_data_15_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_1_V_din = tmp_data_1_V_reg_937;

assign res_stream_V_data_1_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_2_V_din = tmp_data_2_V_reg_942;

assign res_stream_V_data_2_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_3_V_din = tmp_data_3_V_reg_947;

assign res_stream_V_data_3_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_4_V_din = tmp_data_4_V_reg_952;

assign res_stream_V_data_4_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_5_V_din = tmp_data_5_V_reg_957;

assign res_stream_V_data_5_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_6_V_din = tmp_data_6_V_reg_962;

assign res_stream_V_data_6_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_7_V_din = tmp_data_7_V_reg_967;

assign res_stream_V_data_7_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_8_V_din = tmp_data_8_V_reg_972;

assign res_stream_V_data_8_V_write = res_stream_V_data_0_V1_update;

assign res_stream_V_data_9_V_din = tmp_data_9_V_reg_977;

assign res_stream_V_data_9_V_write = res_stream_V_data_0_V1_update;

assign start_out = real_start;

assign tmp_346_122_t_fu_659_p2 = (tmp_s_fu_647_p2 | 4'd1);

assign tmp_456_fu_625_p1 = i_in_reg_574[0:0];

assign tmp_457_fu_637_p2 = i_in_reg_574 << 2'd1;

assign tmp_fu_613_p2 = ((i_in_reg_574 == 2'd2) ? 1'b1 : 1'b0);

assign tmp_s_fu_647_p2 = (p_shl_fu_629_p3 - p_shl19_cast_fu_643_p1);

endmodule //dense_array_ap_fixed_6u_array_ap_fixed_16_6_5_3_0_16u_config15_s
