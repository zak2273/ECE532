// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.3
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module relu_array_ap_fixed_6u_array_ap_ufixed_6_0_4_0_0_6u_relu_config12_s (
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
        data_V_data_4_V_dout,
        data_V_data_4_V_empty_n,
        data_V_data_4_V_read,
        data_V_data_5_V_dout,
        data_V_data_5_V_empty_n,
        data_V_data_5_V_read,
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
        res_V_data_3_V_write,
        res_V_data_4_V_din,
        res_V_data_4_V_full_n,
        res_V_data_4_V_write,
        res_V_data_5_V_din,
        res_V_data_5_V_full_n,
        res_V_data_5_V_write
);

parameter    ap_ST_fsm_state1 = 3'd1;
parameter    ap_ST_fsm_pp0_stage0 = 3'd2;
parameter    ap_ST_fsm_state6 = 3'd4;

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
input  [15:0] data_V_data_4_V_dout;
input   data_V_data_4_V_empty_n;
output   data_V_data_4_V_read;
input  [15:0] data_V_data_5_V_dout;
input   data_V_data_5_V_empty_n;
output   data_V_data_5_V_read;
output  [5:0] res_V_data_0_V_din;
input   res_V_data_0_V_full_n;
output   res_V_data_0_V_write;
output  [5:0] res_V_data_1_V_din;
input   res_V_data_1_V_full_n;
output   res_V_data_1_V_write;
output  [5:0] res_V_data_2_V_din;
input   res_V_data_2_V_full_n;
output   res_V_data_2_V_write;
output  [5:0] res_V_data_3_V_din;
input   res_V_data_3_V_full_n;
output   res_V_data_3_V_write;
output  [5:0] res_V_data_4_V_din;
input   res_V_data_4_V_full_n;
output   res_V_data_4_V_write;
output  [5:0] res_V_data_5_V_din;
input   res_V_data_5_V_full_n;
output   res_V_data_5_V_write;

reg ap_done;
reg ap_idle;
reg start_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
reg    data_V_data_0_V_blk_n;
wire    ap_CS_fsm_pp0_stage0;
reg    ap_enable_reg_pp0_iter1;
wire    ap_block_pp0_stage0;
reg   [0:0] tmp_reg_1101;
reg    data_V_data_1_V_blk_n;
reg    data_V_data_2_V_blk_n;
reg    data_V_data_3_V_blk_n;
reg    data_V_data_4_V_blk_n;
reg    data_V_data_5_V_blk_n;
reg    res_V_data_0_V_blk_n;
reg    ap_enable_reg_pp0_iter3;
reg   [0:0] tmp_reg_1101_pp0_iter2_reg;
reg    res_V_data_1_V_blk_n;
reg    res_V_data_2_V_blk_n;
reg    res_V_data_3_V_blk_n;
reg    res_V_data_4_V_blk_n;
reg    res_V_data_5_V_blk_n;
reg   [3:0] i_reg_274;
wire   [0:0] tmp_fu_285_p2;
wire    ap_block_state2_pp0_stage0_iter0;
wire    data_V_data_0_V0_status;
reg    ap_block_state3_pp0_stage0_iter1;
wire    ap_block_state4_pp0_stage0_iter2;
wire    res_V_data_0_V1_status;
reg    ap_block_state5_pp0_stage0_iter3;
reg    ap_block_pp0_stage0_11001;
reg   [0:0] tmp_reg_1101_pp0_iter1_reg;
wire   [3:0] i_1_fu_291_p2;
reg    ap_enable_reg_pp0_iter0;
reg   [15:0] tmp_data_V_0_reg_1110;
reg   [15:0] tmp_data_V_1_reg_1119;
reg   [15:0] tmp_data_V_215_reg_1128;
reg   [15:0] tmp_data_V_3_reg_1137;
reg   [15:0] tmp_data_V_4_reg_1146;
reg   [15:0] tmp_data_V_5_reg_1155;
wire   [0:0] r_2_fu_325_p2;
reg   [0:0] r_2_reg_1164;
wire   [0:0] Range1_all_ones_fu_341_p2;
reg   [0:0] Range1_all_ones_reg_1169;
wire   [0:0] Range1_all_zeros_fu_347_p2;
reg   [0:0] Range1_all_zeros_reg_1174;
wire   [0:0] r_2_1_fu_357_p2;
reg   [0:0] r_2_1_reg_1179;
wire   [0:0] Range1_all_ones_1_fu_373_p2;
reg   [0:0] Range1_all_ones_1_reg_1184;
wire   [0:0] Range1_all_zeros_1_fu_379_p2;
reg   [0:0] Range1_all_zeros_1_reg_1189;
wire   [0:0] r_2_2_fu_389_p2;
reg   [0:0] r_2_2_reg_1194;
wire   [0:0] Range1_all_ones_2_fu_405_p2;
reg   [0:0] Range1_all_ones_2_reg_1199;
wire   [0:0] Range1_all_zeros_2_fu_411_p2;
reg   [0:0] Range1_all_zeros_2_reg_1204;
wire   [0:0] r_2_3_fu_421_p2;
reg   [0:0] r_2_3_reg_1209;
wire   [0:0] Range1_all_ones_3_fu_437_p2;
reg   [0:0] Range1_all_ones_3_reg_1214;
wire   [0:0] Range1_all_zeros_3_fu_443_p2;
reg   [0:0] Range1_all_zeros_3_reg_1219;
wire   [0:0] r_2_4_fu_453_p2;
reg   [0:0] r_2_4_reg_1224;
wire   [0:0] Range1_all_ones_4_fu_469_p2;
reg   [0:0] Range1_all_ones_4_reg_1229;
wire   [0:0] Range1_all_zeros_4_fu_475_p2;
reg   [0:0] Range1_all_zeros_4_reg_1234;
wire   [0:0] r_2_5_fu_485_p2;
reg   [0:0] r_2_5_reg_1239;
wire   [0:0] Range1_all_ones_5_fu_501_p2;
reg   [0:0] Range1_all_ones_5_reg_1244;
wire   [0:0] Range1_all_zeros_5_fu_507_p2;
reg   [0:0] Range1_all_zeros_5_reg_1249;
wire   [5:0] tmp_data_0_V_fu_603_p3;
reg   [5:0] tmp_data_0_V_reg_1254;
wire   [5:0] tmp_data_1_V_fu_701_p3;
reg   [5:0] tmp_data_1_V_reg_1259;
wire   [5:0] tmp_data_2_V_fu_799_p3;
reg   [5:0] tmp_data_2_V_reg_1264;
wire   [5:0] tmp_data_3_V_fu_897_p3;
reg   [5:0] tmp_data_3_V_reg_1269;
wire   [5:0] tmp_data_4_V_fu_995_p3;
reg   [5:0] tmp_data_4_V_reg_1274;
wire   [5:0] tmp_data_5_V_fu_1093_p3;
reg   [5:0] tmp_data_5_V_reg_1279;
reg    ap_block_state1;
reg    ap_block_pp0_stage0_subdone;
reg    ap_condition_pp0_exit_iter0_state2;
reg    ap_enable_reg_pp0_iter2;
reg    data_V_data_0_V0_update;
reg    res_V_data_0_V1_update;
reg    ap_block_pp0_stage0_01001;
wire   [2:0] tmp_35_fu_321_p1;
wire   [5:0] p_Result_s_fu_331_p4;
wire   [2:0] tmp_40_fu_353_p1;
wire   [5:0] p_Result_10_1_fu_363_p4;
wire   [2:0] tmp_45_fu_385_p1;
wire   [5:0] p_Result_10_2_fu_395_p4;
wire   [2:0] tmp_50_fu_417_p1;
wire   [5:0] p_Result_10_3_fu_427_p4;
wire   [2:0] tmp_55_fu_449_p1;
wire   [5:0] p_Result_10_4_fu_459_p4;
wire   [2:0] tmp_60_fu_481_p1;
wire   [5:0] p_Result_10_5_fu_491_p4;
wire   [0:0] tmp_34_fu_527_p3;
wire   [0:0] r_fu_541_p2;
wire   [0:0] tmp_37_fu_546_p3;
wire   [0:0] tmp_5_fu_553_p2;
wire   [5:0] tmp_84_cast_fu_559_p1;
wire   [5:0] p_Val2_8_fu_518_p4;
wire   [5:0] p_Val2_9_fu_563_p2;
wire   [0:0] tmp_38_fu_569_p3;
wire   [0:0] tmp_36_fu_534_p3;
wire   [0:0] rev_fu_577_p2;
wire   [0:0] carry_1_fu_583_p2;
wire   [0:0] deleted_zeros_fu_589_p3;
wire   [0:0] tmp_s_fu_513_p2;
wire   [5:0] p_mux_fu_595_p3;
wire   [0:0] tmp_39_fu_625_p3;
wire   [0:0] r_1_fu_639_p2;
wire   [0:0] tmp_42_fu_644_p3;
wire   [0:0] tmp_84_1_fu_651_p2;
wire   [5:0] tmp_84_1_cast_fu_657_p1;
wire   [5:0] p_Val2_8_1_fu_616_p4;
wire   [5:0] p_Val2_9_1_fu_661_p2;
wire   [0:0] tmp_43_fu_667_p3;
wire   [0:0] tmp_41_fu_632_p3;
wire   [0:0] rev8_fu_675_p2;
wire   [0:0] carry_1_1_fu_681_p2;
wire   [0:0] deleted_zeros_1_fu_687_p3;
wire   [0:0] tmp_78_1_fu_611_p2;
wire   [5:0] p_mux_1_fu_693_p3;
wire   [0:0] tmp_44_fu_723_p3;
wire   [0:0] r_s_fu_737_p2;
wire   [0:0] tmp_47_fu_742_p3;
wire   [0:0] tmp_84_2_fu_749_p2;
wire   [5:0] tmp_84_2_cast_fu_755_p1;
wire   [5:0] p_Val2_8_2_fu_714_p4;
wire   [5:0] p_Val2_9_2_fu_759_p2;
wire   [0:0] tmp_48_fu_765_p3;
wire   [0:0] tmp_46_fu_730_p3;
wire   [0:0] rev1_fu_773_p2;
wire   [0:0] carry_1_2_fu_779_p2;
wire   [0:0] deleted_zeros_2_fu_785_p3;
wire   [0:0] tmp_78_2_fu_709_p2;
wire   [5:0] p_mux_2_fu_791_p3;
wire   [0:0] tmp_49_fu_821_p3;
wire   [0:0] r_3_fu_835_p2;
wire   [0:0] tmp_52_fu_840_p3;
wire   [0:0] tmp_84_3_fu_847_p2;
wire   [5:0] tmp_84_3_cast_fu_853_p1;
wire   [5:0] p_Val2_8_3_fu_812_p4;
wire   [5:0] p_Val2_9_3_fu_857_p2;
wire   [0:0] tmp_53_fu_863_p3;
wire   [0:0] tmp_51_fu_828_p3;
wire   [0:0] rev2_fu_871_p2;
wire   [0:0] carry_1_3_fu_877_p2;
wire   [0:0] deleted_zeros_3_fu_883_p3;
wire   [0:0] tmp_78_3_fu_807_p2;
wire   [5:0] p_mux_3_fu_889_p3;
wire   [0:0] tmp_54_fu_919_p3;
wire   [0:0] r_4_fu_933_p2;
wire   [0:0] tmp_57_fu_938_p3;
wire   [0:0] tmp_84_4_fu_945_p2;
wire   [5:0] tmp_84_4_cast_fu_951_p1;
wire   [5:0] p_Val2_8_4_fu_910_p4;
wire   [5:0] p_Val2_9_4_fu_955_p2;
wire   [0:0] tmp_58_fu_961_p3;
wire   [0:0] tmp_56_fu_926_p3;
wire   [0:0] rev3_fu_969_p2;
wire   [0:0] carry_1_4_fu_975_p2;
wire   [0:0] deleted_zeros_4_fu_981_p3;
wire   [0:0] tmp_78_4_fu_905_p2;
wire   [5:0] p_mux_4_fu_987_p3;
wire   [0:0] tmp_59_fu_1017_p3;
wire   [0:0] r_5_fu_1031_p2;
wire   [0:0] tmp_62_fu_1036_p3;
wire   [0:0] tmp_84_5_fu_1043_p2;
wire   [5:0] tmp_84_5_cast_fu_1049_p1;
wire   [5:0] p_Val2_8_5_fu_1008_p4;
wire   [5:0] p_Val2_9_5_fu_1053_p2;
wire   [0:0] tmp_63_fu_1059_p3;
wire   [0:0] tmp_61_fu_1024_p3;
wire   [0:0] rev4_fu_1067_p2;
wire   [0:0] carry_1_5_fu_1073_p2;
wire   [0:0] deleted_zeros_5_fu_1079_p3;
wire   [0:0] tmp_78_5_fu_1003_p2;
wire   [5:0] p_mux_5_fu_1085_p3;
wire    ap_CS_fsm_state6;
reg   [2:0] ap_NS_fsm;
reg    ap_idle_pp0;
wire    ap_enable_pp0;

// power-on initialization
initial begin
#0 start_once_reg = 1'b0;
#0 ap_done_reg = 1'b0;
#0 ap_CS_fsm = 3'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter3 = 1'b0;
#0 ap_enable_reg_pp0_iter0 = 1'b0;
#0 ap_enable_reg_pp0_iter2 = 1'b0;
end

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
        end else if ((1'b1 == ap_CS_fsm_state6)) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter0 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b1 == ap_condition_pp0_exit_iter0_state2))) begin
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
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            if ((1'b1 == ap_condition_pp0_exit_iter0_state2)) begin
                ap_enable_reg_pp0_iter1 <= (1'b1 ^ ap_condition_pp0_exit_iter0_state2);
            end else if ((1'b1 == 1'b1)) begin
                ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
            end
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter2 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter2 <= ap_enable_reg_pp0_iter1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter3 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter3 <= ap_enable_reg_pp0_iter2;
        end else if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter3 <= 1'b0;
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
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001) & (tmp_fu_285_p2 == 1'd0))) begin
        i_reg_274 <= i_1_fu_291_p2;
    end else if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        i_reg_274 <= 4'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001) & (tmp_reg_1101 == 1'd0))) begin
        Range1_all_ones_1_reg_1184 <= Range1_all_ones_1_fu_373_p2;
        Range1_all_ones_2_reg_1199 <= Range1_all_ones_2_fu_405_p2;
        Range1_all_ones_3_reg_1214 <= Range1_all_ones_3_fu_437_p2;
        Range1_all_ones_4_reg_1229 <= Range1_all_ones_4_fu_469_p2;
        Range1_all_ones_5_reg_1244 <= Range1_all_ones_5_fu_501_p2;
        Range1_all_ones_reg_1169 <= Range1_all_ones_fu_341_p2;
        Range1_all_zeros_1_reg_1189 <= Range1_all_zeros_1_fu_379_p2;
        Range1_all_zeros_2_reg_1204 <= Range1_all_zeros_2_fu_411_p2;
        Range1_all_zeros_3_reg_1219 <= Range1_all_zeros_3_fu_443_p2;
        Range1_all_zeros_4_reg_1234 <= Range1_all_zeros_4_fu_475_p2;
        Range1_all_zeros_5_reg_1249 <= Range1_all_zeros_5_fu_507_p2;
        Range1_all_zeros_reg_1174 <= Range1_all_zeros_fu_347_p2;
        r_2_1_reg_1179 <= r_2_1_fu_357_p2;
        r_2_2_reg_1194 <= r_2_2_fu_389_p2;
        r_2_3_reg_1209 <= r_2_3_fu_421_p2;
        r_2_4_reg_1224 <= r_2_4_fu_453_p2;
        r_2_5_reg_1239 <= r_2_5_fu_485_p2;
        r_2_reg_1164 <= r_2_fu_325_p2;
        tmp_data_V_0_reg_1110 <= data_V_data_0_V_dout;
        tmp_data_V_1_reg_1119 <= data_V_data_1_V_dout;
        tmp_data_V_215_reg_1128 <= data_V_data_2_V_dout;
        tmp_data_V_3_reg_1137 <= data_V_data_3_V_dout;
        tmp_data_V_4_reg_1146 <= data_V_data_4_V_dout;
        tmp_data_V_5_reg_1155 <= data_V_data_5_V_dout;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (tmp_reg_1101_pp0_iter1_reg == 1'd0))) begin
        tmp_data_0_V_reg_1254 <= tmp_data_0_V_fu_603_p3;
        tmp_data_1_V_reg_1259 <= tmp_data_1_V_fu_701_p3;
        tmp_data_2_V_reg_1264 <= tmp_data_2_V_fu_799_p3;
        tmp_data_3_V_reg_1269 <= tmp_data_3_V_fu_897_p3;
        tmp_data_4_V_reg_1274 <= tmp_data_4_V_fu_995_p3;
        tmp_data_5_V_reg_1279 <= tmp_data_5_V_fu_1093_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        tmp_reg_1101 <= tmp_fu_285_p2;
        tmp_reg_1101_pp0_iter1_reg <= tmp_reg_1101;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b0 == ap_block_pp0_stage0_11001)) begin
        tmp_reg_1101_pp0_iter2_reg <= tmp_reg_1101_pp0_iter1_reg;
    end
end

always @ (*) begin
    if ((tmp_fu_285_p2 == 1'd1)) begin
        ap_condition_pp0_exit_iter0_state2 = 1'b1;
    end else begin
        ap_condition_pp0_exit_iter0_state2 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
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
    if (((ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001) & (tmp_reg_1101 == 1'd0))) begin
        data_V_data_0_V0_update = 1'b1;
    end else begin
        data_V_data_0_V0_update = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_1101 == 1'd0))) begin
        data_V_data_0_V_blk_n = data_V_data_0_V_empty_n;
    end else begin
        data_V_data_0_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_1101 == 1'd0))) begin
        data_V_data_1_V_blk_n = data_V_data_1_V_empty_n;
    end else begin
        data_V_data_1_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_1101 == 1'd0))) begin
        data_V_data_2_V_blk_n = data_V_data_2_V_empty_n;
    end else begin
        data_V_data_2_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_1101 == 1'd0))) begin
        data_V_data_3_V_blk_n = data_V_data_3_V_empty_n;
    end else begin
        data_V_data_3_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_1101 == 1'd0))) begin
        data_V_data_4_V_blk_n = data_V_data_4_V_empty_n;
    end else begin
        data_V_data_4_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_reg_1101 == 1'd0))) begin
        data_V_data_5_V_blk_n = data_V_data_5_V_empty_n;
    end else begin
        data_V_data_5_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
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
    if (((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001) & (tmp_reg_1101_pp0_iter2_reg == 1'd0))) begin
        res_V_data_0_V1_update = 1'b1;
    end else begin
        res_V_data_0_V1_update = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (tmp_reg_1101_pp0_iter2_reg == 1'd0))) begin
        res_V_data_0_V_blk_n = res_V_data_0_V_full_n;
    end else begin
        res_V_data_0_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (tmp_reg_1101_pp0_iter2_reg == 1'd0))) begin
        res_V_data_1_V_blk_n = res_V_data_1_V_full_n;
    end else begin
        res_V_data_1_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (tmp_reg_1101_pp0_iter2_reg == 1'd0))) begin
        res_V_data_2_V_blk_n = res_V_data_2_V_full_n;
    end else begin
        res_V_data_2_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (tmp_reg_1101_pp0_iter2_reg == 1'd0))) begin
        res_V_data_3_V_blk_n = res_V_data_3_V_full_n;
    end else begin
        res_V_data_3_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (tmp_reg_1101_pp0_iter2_reg == 1'd0))) begin
        res_V_data_4_V_blk_n = res_V_data_4_V_full_n;
    end else begin
        res_V_data_4_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (tmp_reg_1101_pp0_iter2_reg == 1'd0))) begin
        res_V_data_5_V_blk_n = res_V_data_5_V_full_n;
    end else begin
        res_V_data_5_V_blk_n = 1'b1;
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
            if ((~((ap_enable_reg_pp0_iter1 == 1'b0) & (1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (tmp_fu_285_p2 == 1'd1)) & ~((ap_enable_reg_pp0_iter2 == 1'b0) & (1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter3 == 1'b1)))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else if ((((ap_enable_reg_pp0_iter1 == 1'b0) & (1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (tmp_fu_285_p2 == 1'd1)) | ((ap_enable_reg_pp0_iter2 == 1'b0) & (1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter3 == 1'b1)))) begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign Range1_all_ones_1_fu_373_p2 = ((p_Result_10_1_fu_363_p4 == 6'd63) ? 1'b1 : 1'b0);

assign Range1_all_ones_2_fu_405_p2 = ((p_Result_10_2_fu_395_p4 == 6'd63) ? 1'b1 : 1'b0);

assign Range1_all_ones_3_fu_437_p2 = ((p_Result_10_3_fu_427_p4 == 6'd63) ? 1'b1 : 1'b0);

assign Range1_all_ones_4_fu_469_p2 = ((p_Result_10_4_fu_459_p4 == 6'd63) ? 1'b1 : 1'b0);

assign Range1_all_ones_5_fu_501_p2 = ((p_Result_10_5_fu_491_p4 == 6'd63) ? 1'b1 : 1'b0);

assign Range1_all_ones_fu_341_p2 = ((p_Result_s_fu_331_p4 == 6'd63) ? 1'b1 : 1'b0);

assign Range1_all_zeros_1_fu_379_p2 = ((p_Result_10_1_fu_363_p4 == 6'd0) ? 1'b1 : 1'b0);

assign Range1_all_zeros_2_fu_411_p2 = ((p_Result_10_2_fu_395_p4 == 6'd0) ? 1'b1 : 1'b0);

assign Range1_all_zeros_3_fu_443_p2 = ((p_Result_10_3_fu_427_p4 == 6'd0) ? 1'b1 : 1'b0);

assign Range1_all_zeros_4_fu_475_p2 = ((p_Result_10_4_fu_459_p4 == 6'd0) ? 1'b1 : 1'b0);

assign Range1_all_zeros_5_fu_507_p2 = ((p_Result_10_5_fu_491_p4 == 6'd0) ? 1'b1 : 1'b0);

assign Range1_all_zeros_fu_347_p2 = ((p_Result_s_fu_331_p4 == 6'd0) ? 1'b1 : 1'b0);

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd2];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_01001 = (((data_V_data_0_V0_status == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (tmp_reg_1101 == 1'd0)) | ((res_V_data_0_V1_status == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (tmp_reg_1101_pp0_iter2_reg == 1'd0)));
end

always @ (*) begin
    ap_block_pp0_stage0_11001 = (((data_V_data_0_V0_status == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (tmp_reg_1101 == 1'd0)) | ((res_V_data_0_V1_status == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (tmp_reg_1101_pp0_iter2_reg == 1'd0)));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = (((data_V_data_0_V0_status == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (tmp_reg_1101 == 1'd0)) | ((res_V_data_0_V1_status == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (tmp_reg_1101_pp0_iter2_reg == 1'd0)));
end

always @ (*) begin
    ap_block_state1 = ((real_start == 1'b0) | (ap_done_reg == 1'b1));
end

assign ap_block_state2_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state3_pp0_stage0_iter1 = ((data_V_data_0_V0_status == 1'b0) & (tmp_reg_1101 == 1'd0));
end

assign ap_block_state4_pp0_stage0_iter2 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state5_pp0_stage0_iter3 = ((res_V_data_0_V1_status == 1'b0) & (tmp_reg_1101_pp0_iter2_reg == 1'd0));
end

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_ready = internal_ap_ready;

assign carry_1_1_fu_681_p2 = (tmp_41_fu_632_p3 & rev8_fu_675_p2);

assign carry_1_2_fu_779_p2 = (tmp_46_fu_730_p3 & rev1_fu_773_p2);

assign carry_1_3_fu_877_p2 = (tmp_51_fu_828_p3 & rev2_fu_871_p2);

assign carry_1_4_fu_975_p2 = (tmp_56_fu_926_p3 & rev3_fu_969_p2);

assign carry_1_5_fu_1073_p2 = (tmp_61_fu_1024_p3 & rev4_fu_1067_p2);

assign carry_1_fu_583_p2 = (tmp_36_fu_534_p3 & rev_fu_577_p2);

assign data_V_data_0_V0_status = (data_V_data_5_V_empty_n & data_V_data_4_V_empty_n & data_V_data_3_V_empty_n & data_V_data_2_V_empty_n & data_V_data_1_V_empty_n & data_V_data_0_V_empty_n);

assign data_V_data_0_V_read = data_V_data_0_V0_update;

assign data_V_data_1_V_read = data_V_data_0_V0_update;

assign data_V_data_2_V_read = data_V_data_0_V0_update;

assign data_V_data_3_V_read = data_V_data_0_V0_update;

assign data_V_data_4_V_read = data_V_data_0_V0_update;

assign data_V_data_5_V_read = data_V_data_0_V0_update;

assign deleted_zeros_1_fu_687_p3 = ((carry_1_1_fu_681_p2[0:0] === 1'b1) ? Range1_all_ones_1_reg_1184 : Range1_all_zeros_1_reg_1189);

assign deleted_zeros_2_fu_785_p3 = ((carry_1_2_fu_779_p2[0:0] === 1'b1) ? Range1_all_ones_2_reg_1199 : Range1_all_zeros_2_reg_1204);

assign deleted_zeros_3_fu_883_p3 = ((carry_1_3_fu_877_p2[0:0] === 1'b1) ? Range1_all_ones_3_reg_1214 : Range1_all_zeros_3_reg_1219);

assign deleted_zeros_4_fu_981_p3 = ((carry_1_4_fu_975_p2[0:0] === 1'b1) ? Range1_all_ones_4_reg_1229 : Range1_all_zeros_4_reg_1234);

assign deleted_zeros_5_fu_1079_p3 = ((carry_1_5_fu_1073_p2[0:0] === 1'b1) ? Range1_all_ones_5_reg_1244 : Range1_all_zeros_5_reg_1249);

assign deleted_zeros_fu_589_p3 = ((carry_1_fu_583_p2[0:0] === 1'b1) ? Range1_all_ones_reg_1169 : Range1_all_zeros_reg_1174);

assign i_1_fu_291_p2 = (i_reg_274 + 4'd1);

assign p_Result_10_1_fu_363_p4 = {{data_V_data_1_V_dout[15:10]}};

assign p_Result_10_2_fu_395_p4 = {{data_V_data_2_V_dout[15:10]}};

assign p_Result_10_3_fu_427_p4 = {{data_V_data_3_V_dout[15:10]}};

assign p_Result_10_4_fu_459_p4 = {{data_V_data_4_V_dout[15:10]}};

assign p_Result_10_5_fu_491_p4 = {{data_V_data_5_V_dout[15:10]}};

assign p_Result_s_fu_331_p4 = {{data_V_data_0_V_dout[15:10]}};

assign p_Val2_8_1_fu_616_p4 = {{tmp_data_V_1_reg_1119[9:4]}};

assign p_Val2_8_2_fu_714_p4 = {{tmp_data_V_215_reg_1128[9:4]}};

assign p_Val2_8_3_fu_812_p4 = {{tmp_data_V_3_reg_1137[9:4]}};

assign p_Val2_8_4_fu_910_p4 = {{tmp_data_V_4_reg_1146[9:4]}};

assign p_Val2_8_5_fu_1008_p4 = {{tmp_data_V_5_reg_1155[9:4]}};

assign p_Val2_8_fu_518_p4 = {{tmp_data_V_0_reg_1110[9:4]}};

assign p_Val2_9_1_fu_661_p2 = (tmp_84_1_cast_fu_657_p1 + p_Val2_8_1_fu_616_p4);

assign p_Val2_9_2_fu_759_p2 = (tmp_84_2_cast_fu_755_p1 + p_Val2_8_2_fu_714_p4);

assign p_Val2_9_3_fu_857_p2 = (tmp_84_3_cast_fu_853_p1 + p_Val2_8_3_fu_812_p4);

assign p_Val2_9_4_fu_955_p2 = (tmp_84_4_cast_fu_951_p1 + p_Val2_8_4_fu_910_p4);

assign p_Val2_9_5_fu_1053_p2 = (tmp_84_5_cast_fu_1049_p1 + p_Val2_8_5_fu_1008_p4);

assign p_Val2_9_fu_563_p2 = (tmp_84_cast_fu_559_p1 + p_Val2_8_fu_518_p4);

assign p_mux_1_fu_693_p3 = ((deleted_zeros_1_fu_687_p3[0:0] === 1'b1) ? p_Val2_9_1_fu_661_p2 : 6'd63);

assign p_mux_2_fu_791_p3 = ((deleted_zeros_2_fu_785_p3[0:0] === 1'b1) ? p_Val2_9_2_fu_759_p2 : 6'd63);

assign p_mux_3_fu_889_p3 = ((deleted_zeros_3_fu_883_p3[0:0] === 1'b1) ? p_Val2_9_3_fu_857_p2 : 6'd63);

assign p_mux_4_fu_987_p3 = ((deleted_zeros_4_fu_981_p3[0:0] === 1'b1) ? p_Val2_9_4_fu_955_p2 : 6'd63);

assign p_mux_5_fu_1085_p3 = ((deleted_zeros_5_fu_1079_p3[0:0] === 1'b1) ? p_Val2_9_5_fu_1053_p2 : 6'd63);

assign p_mux_fu_595_p3 = ((deleted_zeros_fu_589_p3[0:0] === 1'b1) ? p_Val2_9_fu_563_p2 : 6'd63);

assign r_1_fu_639_p2 = (tmp_39_fu_625_p3 | r_2_1_reg_1179);

assign r_2_1_fu_357_p2 = ((tmp_40_fu_353_p1 != 3'd0) ? 1'b1 : 1'b0);

assign r_2_2_fu_389_p2 = ((tmp_45_fu_385_p1 != 3'd0) ? 1'b1 : 1'b0);

assign r_2_3_fu_421_p2 = ((tmp_50_fu_417_p1 != 3'd0) ? 1'b1 : 1'b0);

assign r_2_4_fu_453_p2 = ((tmp_55_fu_449_p1 != 3'd0) ? 1'b1 : 1'b0);

assign r_2_5_fu_485_p2 = ((tmp_60_fu_481_p1 != 3'd0) ? 1'b1 : 1'b0);

assign r_2_fu_325_p2 = ((tmp_35_fu_321_p1 != 3'd0) ? 1'b1 : 1'b0);

assign r_3_fu_835_p2 = (tmp_49_fu_821_p3 | r_2_3_reg_1209);

assign r_4_fu_933_p2 = (tmp_54_fu_919_p3 | r_2_4_reg_1224);

assign r_5_fu_1031_p2 = (tmp_59_fu_1017_p3 | r_2_5_reg_1239);

assign r_fu_541_p2 = (tmp_34_fu_527_p3 | r_2_reg_1164);

assign r_s_fu_737_p2 = (tmp_44_fu_723_p3 | r_2_2_reg_1194);

assign res_V_data_0_V1_status = (res_V_data_5_V_full_n & res_V_data_4_V_full_n & res_V_data_3_V_full_n & res_V_data_2_V_full_n & res_V_data_1_V_full_n & res_V_data_0_V_full_n);

assign res_V_data_0_V_din = tmp_data_0_V_reg_1254;

assign res_V_data_0_V_write = res_V_data_0_V1_update;

assign res_V_data_1_V_din = tmp_data_1_V_reg_1259;

assign res_V_data_1_V_write = res_V_data_0_V1_update;

assign res_V_data_2_V_din = tmp_data_2_V_reg_1264;

assign res_V_data_2_V_write = res_V_data_0_V1_update;

assign res_V_data_3_V_din = tmp_data_3_V_reg_1269;

assign res_V_data_3_V_write = res_V_data_0_V1_update;

assign res_V_data_4_V_din = tmp_data_4_V_reg_1274;

assign res_V_data_4_V_write = res_V_data_0_V1_update;

assign res_V_data_5_V_din = tmp_data_5_V_reg_1279;

assign res_V_data_5_V_write = res_V_data_0_V1_update;

assign rev1_fu_773_p2 = (tmp_48_fu_765_p3 ^ 1'd1);

assign rev2_fu_871_p2 = (tmp_53_fu_863_p3 ^ 1'd1);

assign rev3_fu_969_p2 = (tmp_58_fu_961_p3 ^ 1'd1);

assign rev4_fu_1067_p2 = (tmp_63_fu_1059_p3 ^ 1'd1);

assign rev8_fu_675_p2 = (tmp_43_fu_667_p3 ^ 1'd1);

assign rev_fu_577_p2 = (tmp_38_fu_569_p3 ^ 1'd1);

assign start_out = real_start;

assign tmp_34_fu_527_p3 = tmp_data_V_0_reg_1110[32'd4];

assign tmp_35_fu_321_p1 = data_V_data_0_V_dout[2:0];

assign tmp_36_fu_534_p3 = tmp_data_V_0_reg_1110[32'd9];

assign tmp_37_fu_546_p3 = tmp_data_V_0_reg_1110[32'd3];

assign tmp_38_fu_569_p3 = p_Val2_9_fu_563_p2[32'd5];

assign tmp_39_fu_625_p3 = tmp_data_V_1_reg_1119[32'd4];

assign tmp_40_fu_353_p1 = data_V_data_1_V_dout[2:0];

assign tmp_41_fu_632_p3 = tmp_data_V_1_reg_1119[32'd9];

assign tmp_42_fu_644_p3 = tmp_data_V_1_reg_1119[32'd3];

assign tmp_43_fu_667_p3 = p_Val2_9_1_fu_661_p2[32'd5];

assign tmp_44_fu_723_p3 = tmp_data_V_215_reg_1128[32'd4];

assign tmp_45_fu_385_p1 = data_V_data_2_V_dout[2:0];

assign tmp_46_fu_730_p3 = tmp_data_V_215_reg_1128[32'd9];

assign tmp_47_fu_742_p3 = tmp_data_V_215_reg_1128[32'd3];

assign tmp_48_fu_765_p3 = p_Val2_9_2_fu_759_p2[32'd5];

assign tmp_49_fu_821_p3 = tmp_data_V_3_reg_1137[32'd4];

assign tmp_50_fu_417_p1 = data_V_data_3_V_dout[2:0];

assign tmp_51_fu_828_p3 = tmp_data_V_3_reg_1137[32'd9];

assign tmp_52_fu_840_p3 = tmp_data_V_3_reg_1137[32'd3];

assign tmp_53_fu_863_p3 = p_Val2_9_3_fu_857_p2[32'd5];

assign tmp_54_fu_919_p3 = tmp_data_V_4_reg_1146[32'd4];

assign tmp_55_fu_449_p1 = data_V_data_4_V_dout[2:0];

assign tmp_56_fu_926_p3 = tmp_data_V_4_reg_1146[32'd9];

assign tmp_57_fu_938_p3 = tmp_data_V_4_reg_1146[32'd3];

assign tmp_58_fu_961_p3 = p_Val2_9_4_fu_955_p2[32'd5];

assign tmp_59_fu_1017_p3 = tmp_data_V_5_reg_1155[32'd4];

assign tmp_5_fu_553_p2 = (tmp_37_fu_546_p3 & r_fu_541_p2);

assign tmp_60_fu_481_p1 = data_V_data_5_V_dout[2:0];

assign tmp_61_fu_1024_p3 = tmp_data_V_5_reg_1155[32'd9];

assign tmp_62_fu_1036_p3 = tmp_data_V_5_reg_1155[32'd3];

assign tmp_63_fu_1059_p3 = p_Val2_9_5_fu_1053_p2[32'd5];

assign tmp_78_1_fu_611_p2 = (($signed(tmp_data_V_1_reg_1119) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign tmp_78_2_fu_709_p2 = (($signed(tmp_data_V_215_reg_1128) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign tmp_78_3_fu_807_p2 = (($signed(tmp_data_V_3_reg_1137) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign tmp_78_4_fu_905_p2 = (($signed(tmp_data_V_4_reg_1146) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign tmp_78_5_fu_1003_p2 = (($signed(tmp_data_V_5_reg_1155) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign tmp_84_1_cast_fu_657_p1 = tmp_84_1_fu_651_p2;

assign tmp_84_1_fu_651_p2 = (tmp_42_fu_644_p3 & r_1_fu_639_p2);

assign tmp_84_2_cast_fu_755_p1 = tmp_84_2_fu_749_p2;

assign tmp_84_2_fu_749_p2 = (tmp_47_fu_742_p3 & r_s_fu_737_p2);

assign tmp_84_3_cast_fu_853_p1 = tmp_84_3_fu_847_p2;

assign tmp_84_3_fu_847_p2 = (tmp_52_fu_840_p3 & r_3_fu_835_p2);

assign tmp_84_4_cast_fu_951_p1 = tmp_84_4_fu_945_p2;

assign tmp_84_4_fu_945_p2 = (tmp_57_fu_938_p3 & r_4_fu_933_p2);

assign tmp_84_5_cast_fu_1049_p1 = tmp_84_5_fu_1043_p2;

assign tmp_84_5_fu_1043_p2 = (tmp_62_fu_1036_p3 & r_5_fu_1031_p2);

assign tmp_84_cast_fu_559_p1 = tmp_5_fu_553_p2;

assign tmp_data_0_V_fu_603_p3 = ((tmp_s_fu_513_p2[0:0] === 1'b1) ? p_mux_fu_595_p3 : 6'd0);

assign tmp_data_1_V_fu_701_p3 = ((tmp_78_1_fu_611_p2[0:0] === 1'b1) ? p_mux_1_fu_693_p3 : 6'd0);

assign tmp_data_2_V_fu_799_p3 = ((tmp_78_2_fu_709_p2[0:0] === 1'b1) ? p_mux_2_fu_791_p3 : 6'd0);

assign tmp_data_3_V_fu_897_p3 = ((tmp_78_3_fu_807_p2[0:0] === 1'b1) ? p_mux_3_fu_889_p3 : 6'd0);

assign tmp_data_4_V_fu_995_p3 = ((tmp_78_4_fu_905_p2[0:0] === 1'b1) ? p_mux_4_fu_987_p3 : 6'd0);

assign tmp_data_5_V_fu_1093_p3 = ((tmp_78_5_fu_1003_p2[0:0] === 1'b1) ? p_mux_5_fu_1085_p3 : 6'd0);

assign tmp_fu_285_p2 = ((i_reg_274 == 4'd12) ? 1'b1 : 1'b0);

assign tmp_s_fu_513_p2 = (($signed(tmp_data_V_0_reg_1110) > $signed(16'd0)) ? 1'b1 : 1'b0);

endmodule //relu_array_ap_fixed_6u_array_ap_ufixed_6_0_4_0_0_6u_relu_config12_s
