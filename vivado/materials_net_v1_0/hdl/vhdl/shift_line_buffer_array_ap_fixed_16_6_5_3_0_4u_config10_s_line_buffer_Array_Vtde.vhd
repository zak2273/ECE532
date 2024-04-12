-- ==============================================================
-- File generated on Wed Mar 20 19:28:04 EDT 2024
-- Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2018.3 (64-bit)
-- SW Build 2405991 on Thu Dec  6 23:36:41 MST 2018
-- IP Build 2404404 on Fri Dec  7 01:43:56 MST 2018
-- Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
-- ==============================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use IEEE.std_logic_unsigned.all;

entity shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config10_s_line_buffer_Array_Vtde_core is
    generic (
        DATA_WIDTH : integer := 16;
        ADDR_WIDTH : integer := 4;
        DEPTH : integer := 9);
    port (
        clk : in std_logic;
        ce : in std_logic;
        din : in std_logic_vector(DATA_WIDTH-1 downto 0);
        addr : in std_logic_vector(ADDR_WIDTH-1 downto 0);
        dout : out std_logic_vector(DATA_WIDTH-1 downto 0));

    attribute keep_hierarchy : string;
    attribute keep_hierarchy of shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config10_s_line_buffer_Array_Vtde_core : entity is "yes";

end entity;

architecture rtl of shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config10_s_line_buffer_Array_Vtde_core is
type SRL_ARRAY is array (0 to DEPTH-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
signal ShiftRegMem : SRL_ARRAY := (others=>(others=>'0'));


begin
p_shift: process (clk)
    variable i: integer;
begin
    if (clk'event and clk = '1') then
        if (ce = '1') then
            for i in 0 to DEPTH - 2 loop
                ShiftRegMem(i+1) <= ShiftRegMem(i);
            end loop;
            ShiftRegMem(0) <= din;
        end if;
    end if;
end process;

dout <= ShiftRegMem(CONV_INTEGER(addr));

end rtl;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_unsigned.all;

entity shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config10_s_line_buffer_Array_Vtde is
    generic (
        DataWidth : integer := 16;
        AddressWidth : integer := 4;
        AddressRange : integer := 9);
    port (
        clk : in std_logic;
        reset : in std_logic;
        address0 : in std_logic_vector(AddressWidth-1 downto 0);
        ce0 : in std_logic;
        we0 : in std_logic;
        d0 : in std_logic_vector(DataWidth-1 downto 0);
        q0 : out std_logic_vector(DataWidth-1 downto 0));
end;

architecture behav of shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config10_s_line_buffer_Array_Vtde is

    component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config10_s_line_buffer_Array_Vtde_core is
        generic (
            DATA_WIDTH : integer;
            ADDR_WIDTH : integer;
            DEPTH : integer);
        port (
            clk : in std_logic;
            ce : in std_logic;
            din : in std_logic_vector(DATA_WIDTH-1 downto 0);
            addr : in std_logic_vector(ADDR_WIDTH-1 downto 0);
            dout : out std_logic_vector(DATA_WIDTH-1 downto 0));
    end component;

begin
    shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config10_s_line_buffer_Array_Vtde_core_U : component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config10_s_line_buffer_Array_Vtde_core
    generic map (
        DATA_WIDTH => DataWidth,
        ADDR_WIDTH => AddressWidth,
        DEPTH => AddressRange)
    port map (
        clk => clk,
        ce => we0,
        din => d0,
        addr => address0,
        dout => q0);

end behav;

