// ==============================================================
// File generated on Wed Mar 20 19:27:57 EDT 2024
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2018.3 (64-bit)
// SW Build 2405991 on Thu Dec  6 23:36:41 MST 2018
// IP Build 2404404 on Fri Dec  7 01:43:56 MST 2018
// Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module softmax_stable_array_array_ap_fixed_16_6_5_3_0_4u_softmax_config21_s_invert_tHfu_rom (
addr0, ce0, q0, clk);

parameter DWIDTH = 18;
parameter AWIDTH = 10;
parameter MEM_SIZE = 1024;

input[AWIDTH-1:0] addr0;
input ce0;
output reg[DWIDTH-1:0] q0;
input clk;

reg [DWIDTH-1:0] ram[0:MEM_SIZE-1];

initial begin
    $readmemh("./softmax_stable_array_array_ap_fixed_16_6_5_3_0_4u_softmax_config21_s_invert_tHfu_rom.dat", ram);
end



always @(posedge clk)  
begin 
    if (ce0) 
    begin
        q0 <= ram[addr0];
    end
end



endmodule

`timescale 1 ns / 1 ps
module softmax_stable_array_array_ap_fixed_16_6_5_3_0_4u_softmax_config21_s_invert_tHfu(
    reset,
    clk,
    address0,
    ce0,
    q0);

parameter DataWidth = 32'd18;
parameter AddressRange = 32'd1024;
parameter AddressWidth = 32'd10;
input reset;
input clk;
input[AddressWidth - 1:0] address0;
input ce0;
output[DataWidth - 1:0] q0;



softmax_stable_array_array_ap_fixed_16_6_5_3_0_4u_softmax_config21_s_invert_tHfu_rom softmax_stable_array_array_ap_fixed_16_6_5_3_0_4u_softmax_config21_s_invert_tHfu_rom_U(
    .clk( clk ),
    .addr0( address0 ),
    .ce0( ce0 ),
    .q0( q0 ));

endmodule

