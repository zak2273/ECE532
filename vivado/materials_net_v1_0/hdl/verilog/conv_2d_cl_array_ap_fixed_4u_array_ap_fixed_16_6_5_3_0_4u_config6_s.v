// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.3
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module conv_2d_cl_array_ap_fixed_4u_array_ap_fixed_16_6_5_3_0_4u_config6_s (
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
        data_V_data_0_V_dout,
        data_V_data_0_V_empty_n,
        data_V_data_0_V_read,
        data_V_data_1_V_dout,
        data_V_data_1_V_empty_n,
        data_V_data_1_V_read,
        data_V_data_2_V_dout,
        data_V_data_2_V_empty_n,
        data_V_data_2_V_read,
        data_V_data_3_V_dout,
        data_V_data_3_V_empty_n,
        data_V_data_3_V_read,
        res_V_data_0_V_din,
        res_V_data_0_V_full_n,
        res_V_data_0_V_write,
        res_V_data_1_V_din,
        res_V_data_1_V_full_n,
        res_V_data_1_V_write,
        res_V_data_2_V_din,
        res_V_data_2_V_full_n,
        res_V_data_2_V_write,
        res_V_data_3_V_din,
        res_V_data_3_V_full_n,
        res_V_data_3_V_write
);

parameter    ap_ST_fsm_state1 = 10'd1;
parameter    ap_ST_fsm_pp0_stage0 = 10'd2;
parameter    ap_ST_fsm_pp0_stage1 = 10'd4;
parameter    ap_ST_fsm_pp0_stage2 = 10'd8;
parameter    ap_ST_fsm_pp0_stage3 = 10'd16;
parameter    ap_ST_fsm_pp0_stage4 = 10'd32;
parameter    ap_ST_fsm_pp0_stage5 = 10'd64;
parameter    ap_ST_fsm_pp0_stage6 = 10'd128;
parameter    ap_ST_fsm_pp0_stage7 = 10'd256;
parameter    ap_ST_fsm_state11 = 10'd512;

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
input  [15:0] data_V_data_0_V_dout;
input   data_V_data_0_V_empty_n;
output   data_V_data_0_V_read;
input  [15:0] data_V_data_1_V_dout;
input   data_V_data_1_V_empty_n;
output   data_V_data_1_V_read;
input  [15:0] data_V_data_2_V_dout;
input   data_V_data_2_V_empty_n;
output   data_V_data_2_V_read;
input  [15:0] data_V_data_3_V_dout;
input   data_V_data_3_V_empty_n;
output   data_V_data_3_V_read;
output  [15:0] res_V_data_0_V_din;
input   res_V_data_0_V_full_n;
output   res_V_data_0_V_write;
output  [15:0] res_V_data_1_V_din;
input   res_V_data_1_V_full_n;
output   res_V_data_1_V_write;
output  [15:0] res_V_data_2_V_din;
input   res_V_data_2_V_full_n;
output   res_V_data_2_V_write;
output  [15:0] res_V_data_3_V_din;
input   res_V_data_3_V_full_n;
output   res_V_data_3_V_write;

reg ap_done;
reg ap_idle;
reg start_write;
reg res_V_data_0_V_write;
reg res_V_data_1_V_write;
reg res_V_data_2_V_write;
reg res_V_data_3_V_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [9:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
reg    data_V_data_0_V_blk_n;
wire    ap_CS_fsm_pp0_stage1;
reg    ap_enable_reg_pp0_iter0;
wire    ap_block_pp0_stage1;
reg   [0:0] exitcond_flatten_reg_367;
reg    data_V_data_1_V_blk_n;
reg    data_V_data_2_V_blk_n;
reg    data_V_data_3_V_blk_n;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_blk_n;
reg    res_V_data_0_V_blk_n;
wire    ap_CS_fsm_pp0_stage0;
reg    ap_enable_reg_pp0_iter1;
wire    ap_block_pp0_stage0;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_blk_n;
reg    res_V_data_1_V_blk_n;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_blk_n;
reg    res_V_data_2_V_blk_n;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_blk_n;
reg    res_V_data_3_V_blk_n;
reg   [10:0] indvar_flatten_reg_236;
wire   [0:0] exitcond_flatten_fu_335_p2;
wire    ap_block_state2_pp0_stage0_iter0;
reg    ap_block_state10_pp0_stage0_iter1;
reg    ap_block_pp0_stage0_11001;
wire   [10:0] indvar_flatten_next_fu_341_p2;
reg   [10:0] indvar_flatten_next_reg_371;
reg    ap_block_state1;
reg    ap_block_pp0_stage0_subdone;
reg    ap_condition_pp0_exit_iter0_state2;
wire    ap_block_state9_pp0_stage7_iter0;
wire    ap_block_pp0_stage7_subdone;
wire    ap_CS_fsm_pp0_stage7;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_start;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_done;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_idle;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_ready;
wire   [15:0] grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_din;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_write;
wire   [15:0] grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_din;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_write;
wire   [15:0] grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_din;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_write;
wire   [15:0] grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_din;
wire    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_write;
reg    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_ce;
wire    data_V_data_0_V0_status;
reg    ap_block_state3_pp0_stage1_iter0_ignore_call10;
reg    ap_block_pp0_stage1_11001_ignoreCallOp30;
wire    ap_block_state4_pp0_stage2_iter0_ignore_call10;
wire    ap_block_pp0_stage2_11001_ignoreCallOp31;
wire    ap_CS_fsm_pp0_stage2;
wire    ap_block_state5_pp0_stage3_iter0_ignore_call10;
wire    ap_block_pp0_stage3_11001_ignoreCallOp32;
wire    ap_CS_fsm_pp0_stage3;
wire    ap_block_state6_pp0_stage4_iter0_ignore_call10;
wire    ap_block_pp0_stage4_11001_ignoreCallOp33;
wire    ap_CS_fsm_pp0_stage4;
wire    ap_block_state7_pp0_stage5_iter0_ignore_call10;
wire    ap_block_pp0_stage5_11001_ignoreCallOp34;
wire    ap_CS_fsm_pp0_stage5;
wire    ap_block_state8_pp0_stage6_iter0_ignore_call10;
wire    ap_block_pp0_stage6_11001_ignoreCallOp35;
wire    ap_CS_fsm_pp0_stage6;
wire    ap_block_state9_pp0_stage7_iter0_ignore_call10;
wire    ap_block_pp0_stage7_11001_ignoreCallOp36;
wire    ap_block_state2_pp0_stage0_iter0_ignore_call10;
wire    ap_block_state10_pp0_stage0_iter1_ignore_call10;
wire    ap_block_pp0_stage0_11001_ignoreCallOp42;
reg   [10:0] ap_phi_mux_indvar_flatten_phi_fu_240_p4;
reg    grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_start_reg;
reg    ap_block_state3_pp0_stage1_iter0;
reg    ap_block_pp0_stage1_11001;
wire    ap_block_state4_pp0_stage2_iter0;
wire    ap_block_pp0_stage2_11001;
wire    ap_block_state5_pp0_stage3_iter0;
wire    ap_block_pp0_stage3_11001;
reg    data_V_data_0_V0_update;
wire    ap_CS_fsm_state11;
reg   [9:0] ap_NS_fsm;
reg    ap_block_pp0_stage1_subdone;
wire    ap_block_pp0_stage2_subdone;
wire    ap_block_pp0_stage3_subdone;
wire    ap_block_state6_pp0_stage4_iter0;
wire    ap_block_pp0_stage4_subdone;
wire    ap_block_state7_pp0_stage5_iter0;
wire    ap_block_pp0_stage5_subdone;
wire    ap_block_state8_pp0_stage6_iter0;
wire    ap_block_pp0_stage6_subdone;
reg    ap_idle_pp0;
wire    ap_enable_pp0;

// power-on initialization
initial begin
#0 start_once_reg = 1'b0;
#0 ap_done_reg = 1'b0;
#0 ap_CS_fsm = 10'd1;
#0 ap_enable_reg_pp0_iter0 = 1'b0;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_start_reg = 1'b0;
end

compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_start),
    .ap_done(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_done),
    .ap_idle(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_idle),
    .ap_ready(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_ready),
    .in_elem_data_0_V_read(data_V_data_0_V_dout),
    .in_elem_data_1_V_read(data_V_data_1_V_dout),
    .in_elem_data_2_V_read(data_V_data_2_V_dout),
    .in_elem_data_3_V_read(data_V_data_3_V_dout),
    .res_stream_V_data_0_V_din(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_din),
    .res_stream_V_data_0_V_full_n(res_V_data_0_V_full_n),
    .res_stream_V_data_0_V_write(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_write),
    .res_stream_V_data_1_V_din(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_din),
    .res_stream_V_data_1_V_full_n(res_V_data_1_V_full_n),
    .res_stream_V_data_1_V_write(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_write),
    .res_stream_V_data_2_V_din(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_din),
    .res_stream_V_data_2_V_full_n(res_V_data_2_V_full_n),
    .res_stream_V_data_2_V_write(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_write),
    .res_stream_V_data_3_V_din(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_din),
    .res_stream_V_data_3_V_full_n(res_V_data_3_V_full_n),
    .res_stream_V_data_3_V_write(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_write),
    .res_stream_V_data_0_V_blk_n(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_blk_n),
    .res_stream_V_data_1_V_blk_n(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_blk_n),
    .res_stream_V_data_2_V_blk_n(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_blk_n),
    .res_stream_V_data_3_V_blk_n(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_blk_n),
    .ap_ce(grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_ce)
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
        end else if ((1'b1 == ap_CS_fsm_state11)) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter0 <= 1'b0;
    end else begin
        if (((1'b1 == ap_condition_pp0_exit_iter0_state2) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
            ap_enable_reg_pp0_iter0 <= 1'b0;
        end else if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter0 <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if ((((1'b1 == ap_CS_fsm_pp0_stage7) & (1'b0 == ap_block_pp0_stage7_subdone)) | ((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_subdone)))) begin
            ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
        end else if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_start_reg <= 1'b0;
    end else begin
        if (((exitcond_flatten_fu_335_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
            grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_start_reg <= 1'b1;
        end else if ((grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_ready == 1'b1)) begin
            grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_start_reg <= 1'b0;
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
    if (((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        indvar_flatten_reg_236 <= indvar_flatten_next_reg_371;
    end else if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        indvar_flatten_reg_236 <= 11'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        exitcond_flatten_reg_367 <= exitcond_flatten_fu_335_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        indvar_flatten_next_reg_371 <= indvar_flatten_next_fu_341_p2;
    end
end

always @ (*) begin
    if ((exitcond_flatten_fu_335_p2 == 1'd1)) begin
        ap_condition_pp0_exit_iter0_state2 = 1'b1;
    end else begin
        ap_condition_pp0_exit_iter0_state2 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state11)) begin
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
    if (((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
        ap_phi_mux_indvar_flatten_phi_fu_240_p4 = indvar_flatten_next_reg_371;
    end else begin
        ap_phi_mux_indvar_flatten_phi_fu_240_p4 = indvar_flatten_reg_236;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (1'b0 == ap_block_pp0_stage1_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1))) begin
        data_V_data_0_V0_update = 1'b1;
    end else begin
        data_V_data_0_V0_update = 1'b0;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (1'b0 == ap_block_pp0_stage1) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1))) begin
        data_V_data_0_V_blk_n = data_V_data_0_V_empty_n;
    end else begin
        data_V_data_0_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (1'b0 == ap_block_pp0_stage1) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1))) begin
        data_V_data_1_V_blk_n = data_V_data_1_V_empty_n;
    end else begin
        data_V_data_1_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (1'b0 == ap_block_pp0_stage1) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1))) begin
        data_V_data_2_V_blk_n = data_V_data_2_V_empty_n;
    end else begin
        data_V_data_2_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (1'b0 == ap_block_pp0_stage1) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1))) begin
        data_V_data_3_V_blk_n = data_V_data_3_V_empty_n;
    end else begin
        data_V_data_3_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001_ignoreCallOp30)) | ((1'b0 == ap_block_pp0_stage6_11001_ignoreCallOp35) & (1'b1 == ap_CS_fsm_pp0_stage6)) | ((1'b0 == ap_block_pp0_stage5_11001_ignoreCallOp34) & (1'b1 == ap_CS_fsm_pp0_stage5)) | ((1'b0 == ap_block_pp0_stage4_11001_ignoreCallOp33) & (1'b1 == ap_CS_fsm_pp0_stage4)) | ((1'b0 == ap_block_pp0_stage3_11001_ignoreCallOp32) & (1'b1 == ap_CS_fsm_pp0_stage3)) | ((1'b0 == ap_block_pp0_stage2_11001_ignoreCallOp31) & (1'b1 == ap_CS_fsm_pp0_stage2)) | ((1'b0 == ap_block_pp0_stage7_11001_ignoreCallOp36) & (1'b1 == ap_CS_fsm_pp0_stage7)) | ((1'b0 == ap_block_pp0_stage0_11001_ignoreCallOp42) & (1'b1 == ap_CS_fsm_pp0_stage0)))) begin
        grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_ce = 1'b1;
    end else begin
        grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_ce = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state11)) begin
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
    if (((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
        res_V_data_0_V_blk_n = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_blk_n;
    end else begin
        res_V_data_0_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
        res_V_data_0_V_write = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_write;
    end else begin
        res_V_data_0_V_write = 1'b0;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
        res_V_data_1_V_blk_n = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_blk_n;
    end else begin
        res_V_data_1_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
        res_V_data_1_V_write = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_write;
    end else begin
        res_V_data_1_V_write = 1'b0;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
        res_V_data_2_V_blk_n = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_blk_n;
    end else begin
        res_V_data_2_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
        res_V_data_2_V_write = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_write;
    end else begin
        res_V_data_2_V_write = 1'b0;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
        res_V_data_3_V_blk_n = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_blk_n;
    end else begin
        res_V_data_3_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
        res_V_data_3_V_write = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_write;
    end else begin
        res_V_data_3_V_write = 1'b0;
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
            if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_pp0_stage0 : begin
            if ((~((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone) & (exitcond_flatten_fu_335_p2 == 1'd1)) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage1;
            end else if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone) & (exitcond_flatten_fu_335_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end
        end
        ap_ST_fsm_pp0_stage1 : begin
            if ((1'b0 == ap_block_pp0_stage1_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage1;
            end
        end
        ap_ST_fsm_pp0_stage2 : begin
            if ((1'b0 == ap_block_pp0_stage2_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage2;
            end
        end
        ap_ST_fsm_pp0_stage3 : begin
            if ((1'b0 == ap_block_pp0_stage3_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage3;
            end
        end
        ap_ST_fsm_pp0_stage4 : begin
            if ((1'b0 == ap_block_pp0_stage4_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage4;
            end
        end
        ap_ST_fsm_pp0_stage5 : begin
            if ((1'b0 == ap_block_pp0_stage5_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage6;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage5;
            end
        end
        ap_ST_fsm_pp0_stage6 : begin
            if ((1'b0 == ap_block_pp0_stage6_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage7;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage6;
            end
        end
        ap_ST_fsm_pp0_stage7 : begin
            if ((1'b0 == ap_block_pp0_stage7_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage7;
            end
        end
        ap_ST_fsm_state11 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_pp0_stage1 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_pp0_stage2 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_pp0_stage3 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_pp0_stage4 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_pp0_stage5 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_pp0_stage6 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_pp0_stage7 = ap_CS_fsm[32'd8];

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state11 = ap_CS_fsm[32'd9];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_11001 = ((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & ((grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_blk_n == 1'b0) | (grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_blk_n == 1'b0) | (grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_blk_n == 1'b0) | (grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_blk_n == 1'b0)));
end

assign ap_block_pp0_stage0_11001_ignoreCallOp42 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_subdone = ((exitcond_flatten_reg_367 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & ((grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_blk_n == 1'b0) | (grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_blk_n == 1'b0) | (grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_blk_n == 1'b0) | (grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_blk_n == 1'b0)));
end

assign ap_block_pp0_stage1 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage1_11001 = ((exitcond_flatten_reg_367 == 1'd0) & (data_V_data_0_V0_status == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage1_11001_ignoreCallOp30 = ((exitcond_flatten_reg_367 == 1'd0) & (data_V_data_0_V0_status == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage1_subdone = ((exitcond_flatten_reg_367 == 1'd0) & (data_V_data_0_V0_status == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b1));
end

assign ap_block_pp0_stage2_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage2_11001_ignoreCallOp31 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage2_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage3_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage3_11001_ignoreCallOp32 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage3_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage4_11001_ignoreCallOp33 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage4_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage5_11001_ignoreCallOp34 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage5_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage6_11001_ignoreCallOp35 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage6_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage7_11001_ignoreCallOp36 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage7_subdone = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state1 = ((real_start == 1'b0) | (ap_done_reg == 1'b1));
end

always @ (*) begin
    ap_block_state10_pp0_stage0_iter1 = ((exitcond_flatten_reg_367 == 1'd0) & ((grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_blk_n == 1'b0) | (grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_blk_n == 1'b0) | (grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_blk_n == 1'b0) | (grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_blk_n == 1'b0)));
end

assign ap_block_state10_pp0_stage0_iter1_ignore_call10 = ~(1'b1 == 1'b1);

assign ap_block_state2_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state2_pp0_stage0_iter0_ignore_call10 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state3_pp0_stage1_iter0 = ((exitcond_flatten_reg_367 == 1'd0) & (data_V_data_0_V0_status == 1'b0));
end

always @ (*) begin
    ap_block_state3_pp0_stage1_iter0_ignore_call10 = ((exitcond_flatten_reg_367 == 1'd0) & (data_V_data_0_V0_status == 1'b0));
end

assign ap_block_state4_pp0_stage2_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state4_pp0_stage2_iter0_ignore_call10 = ~(1'b1 == 1'b1);

assign ap_block_state5_pp0_stage3_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state5_pp0_stage3_iter0_ignore_call10 = ~(1'b1 == 1'b1);

assign ap_block_state6_pp0_stage4_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state6_pp0_stage4_iter0_ignore_call10 = ~(1'b1 == 1'b1);

assign ap_block_state7_pp0_stage5_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state7_pp0_stage5_iter0_ignore_call10 = ~(1'b1 == 1'b1);

assign ap_block_state8_pp0_stage6_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state8_pp0_stage6_iter0_ignore_call10 = ~(1'b1 == 1'b1);

assign ap_block_state9_pp0_stage7_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state9_pp0_stage7_iter0_ignore_call10 = ~(1'b1 == 1'b1);

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_ready = internal_ap_ready;

assign data_V_data_0_V0_status = (data_V_data_3_V_empty_n & data_V_data_2_V_empty_n & data_V_data_1_V_empty_n & data_V_data_0_V_empty_n);

assign data_V_data_0_V_read = data_V_data_0_V0_update;

assign data_V_data_1_V_read = data_V_data_0_V0_update;

assign data_V_data_2_V_read = data_V_data_0_V0_update;

assign data_V_data_3_V_read = data_V_data_0_V0_update;

assign exitcond_flatten_fu_335_p2 = ((ap_phi_mux_indvar_flatten_phi_fu_240_p4 == 11'd1131) ? 1'b1 : 1'b0);

assign grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_start = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_ap_start_reg;

assign indvar_flatten_next_fu_341_p2 = (ap_phi_mux_indvar_flatten_phi_fu_240_p4 + 11'd1);

assign res_V_data_0_V_din = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_0_V_din;

assign res_V_data_1_V_din = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_1_V_din;

assign res_V_data_2_V_din = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_2_V_din;

assign res_V_data_3_V_din = grp_compute_output_buffer_2d_array_array_ap_fixed_16_6_5_3_0_4u_config6_s_fu_247_res_stream_V_data_3_V_din;

assign start_out = real_start;

endmodule //conv_2d_cl_array_ap_fixed_4u_array_ap_fixed_16_6_5_3_0_4u_config6_s
