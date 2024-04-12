-- ==============================================================
-- RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
-- Version: 2018.3
-- Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
-- 
-- ===========================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s is
port (
    ap_clk : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC;
    ap_start : IN STD_LOGIC;
    ap_done : OUT STD_LOGIC;
    ap_idle : OUT STD_LOGIC;
    ap_ready : OUT STD_LOGIC;
    in_elem_data_0_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    in_elem_data_1_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    in_elem_data_2_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    in_elem_data_3_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_4_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_5_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_6_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_7_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_8_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_9_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_10_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_11_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_16_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_17_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_18_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_19_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_20_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_21_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_22_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_23_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_28_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_29_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_30_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_31_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_32_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_33_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_34_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    kernel_window_35_V_read : IN STD_LOGIC_VECTOR (15 downto 0);
    ap_return_0 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_1 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_2 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_3 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_4 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_5 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_6 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_7 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_8 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_9 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_10 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_11 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_12 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_13 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_14 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_15 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_16 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_17 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_18 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_19 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_20 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_21 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_22 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_23 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_24 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_25 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_26 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_27 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_28 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_29 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_30 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_31 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_32 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_33 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_34 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_return_35 : OUT STD_LOGIC_VECTOR (15 downto 0);
    ap_ce : IN STD_LOGIC );
end;


architecture behav of shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_logic_0 : STD_LOGIC := '0';
    constant ap_ST_fsm_state1 : STD_LOGIC_VECTOR (0 downto 0) := "1";
    constant ap_const_lv32_0 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000000";
    constant ap_const_lv1_1 : STD_LOGIC_VECTOR (0 downto 0) := "1";
    constant ap_const_lv6_26 : STD_LOGIC_VECTOR (5 downto 0) := "100110";
    constant ap_const_boolean_1 : BOOLEAN := true;

    signal ap_CS_fsm : STD_LOGIC_VECTOR (0 downto 0) := "1";
    attribute fsm_encoding : string;
    attribute fsm_encoding of ap_CS_fsm : signal is "none";
    signal ap_CS_fsm_state1 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state1 : signal is "none";
    signal line_buffer_Array_V_1_0_0_ce0 : STD_LOGIC;
    signal line_buffer_Array_V_1_0_0_we0 : STD_LOGIC;
    signal line_buffer_Array_V_1_0_0_q0 : STD_LOGIC_VECTOR (15 downto 0);
    signal line_buffer_Array_V_1_1_0_ce0 : STD_LOGIC;
    signal line_buffer_Array_V_1_1_0_we0 : STD_LOGIC;
    signal line_buffer_Array_V_1_1_0_q0 : STD_LOGIC_VECTOR (15 downto 0);
    signal line_buffer_Array_V_1_0_1_ce0 : STD_LOGIC;
    signal line_buffer_Array_V_1_0_1_we0 : STD_LOGIC;
    signal line_buffer_Array_V_1_0_1_q0 : STD_LOGIC_VECTOR (15 downto 0);
    signal line_buffer_Array_V_1_1_1_ce0 : STD_LOGIC;
    signal line_buffer_Array_V_1_1_1_we0 : STD_LOGIC;
    signal line_buffer_Array_V_1_1_1_q0 : STD_LOGIC_VECTOR (15 downto 0);
    signal line_buffer_Array_V_1_0_2_ce0 : STD_LOGIC;
    signal line_buffer_Array_V_1_0_2_we0 : STD_LOGIC;
    signal line_buffer_Array_V_1_0_2_q0 : STD_LOGIC_VECTOR (15 downto 0);
    signal line_buffer_Array_V_1_1_2_ce0 : STD_LOGIC;
    signal line_buffer_Array_V_1_1_2_we0 : STD_LOGIC;
    signal line_buffer_Array_V_1_1_2_q0 : STD_LOGIC_VECTOR (15 downto 0);
    signal line_buffer_Array_V_1_0_3_ce0 : STD_LOGIC;
    signal line_buffer_Array_V_1_0_3_we0 : STD_LOGIC;
    signal line_buffer_Array_V_1_0_3_q0 : STD_LOGIC_VECTOR (15 downto 0);
    signal line_buffer_Array_V_1_1_3_ce0 : STD_LOGIC;
    signal line_buffer_Array_V_1_1_3_we0 : STD_LOGIC;
    signal line_buffer_Array_V_1_1_3_q0 : STD_LOGIC_VECTOR (15 downto 0);
    signal ap_NS_fsm : STD_LOGIC_VECTOR (0 downto 0);

    component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s_line_buffer_Array_V_hbi IS
    generic (
        DataWidth : INTEGER;
        AddressRange : INTEGER;
        AddressWidth : INTEGER );
    port (
        clk : IN STD_LOGIC;
        reset : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR (5 downto 0);
        ce0 : IN STD_LOGIC;
        we0 : IN STD_LOGIC;
        d0 : IN STD_LOGIC_VECTOR (15 downto 0);
        q0 : OUT STD_LOGIC_VECTOR (15 downto 0) );
    end component;



begin
    line_buffer_Array_V_1_0_0_U : component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s_line_buffer_Array_V_hbi
    generic map (
        DataWidth => 16,
        AddressRange => 39,
        AddressWidth => 6)
    port map (
        clk => ap_clk,
        reset => ap_rst,
        address0 => ap_const_lv6_26,
        ce0 => line_buffer_Array_V_1_0_0_ce0,
        we0 => line_buffer_Array_V_1_0_0_we0,
        d0 => in_elem_data_0_V_read,
        q0 => line_buffer_Array_V_1_0_0_q0);

    line_buffer_Array_V_1_1_0_U : component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s_line_buffer_Array_V_hbi
    generic map (
        DataWidth => 16,
        AddressRange => 39,
        AddressWidth => 6)
    port map (
        clk => ap_clk,
        reset => ap_rst,
        address0 => ap_const_lv6_26,
        ce0 => line_buffer_Array_V_1_1_0_ce0,
        we0 => line_buffer_Array_V_1_1_0_we0,
        d0 => line_buffer_Array_V_1_0_0_q0,
        q0 => line_buffer_Array_V_1_1_0_q0);

    line_buffer_Array_V_1_0_1_U : component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s_line_buffer_Array_V_hbi
    generic map (
        DataWidth => 16,
        AddressRange => 39,
        AddressWidth => 6)
    port map (
        clk => ap_clk,
        reset => ap_rst,
        address0 => ap_const_lv6_26,
        ce0 => line_buffer_Array_V_1_0_1_ce0,
        we0 => line_buffer_Array_V_1_0_1_we0,
        d0 => in_elem_data_1_V_read,
        q0 => line_buffer_Array_V_1_0_1_q0);

    line_buffer_Array_V_1_1_1_U : component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s_line_buffer_Array_V_hbi
    generic map (
        DataWidth => 16,
        AddressRange => 39,
        AddressWidth => 6)
    port map (
        clk => ap_clk,
        reset => ap_rst,
        address0 => ap_const_lv6_26,
        ce0 => line_buffer_Array_V_1_1_1_ce0,
        we0 => line_buffer_Array_V_1_1_1_we0,
        d0 => line_buffer_Array_V_1_0_1_q0,
        q0 => line_buffer_Array_V_1_1_1_q0);

    line_buffer_Array_V_1_0_2_U : component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s_line_buffer_Array_V_hbi
    generic map (
        DataWidth => 16,
        AddressRange => 39,
        AddressWidth => 6)
    port map (
        clk => ap_clk,
        reset => ap_rst,
        address0 => ap_const_lv6_26,
        ce0 => line_buffer_Array_V_1_0_2_ce0,
        we0 => line_buffer_Array_V_1_0_2_we0,
        d0 => in_elem_data_2_V_read,
        q0 => line_buffer_Array_V_1_0_2_q0);

    line_buffer_Array_V_1_1_2_U : component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s_line_buffer_Array_V_hbi
    generic map (
        DataWidth => 16,
        AddressRange => 39,
        AddressWidth => 6)
    port map (
        clk => ap_clk,
        reset => ap_rst,
        address0 => ap_const_lv6_26,
        ce0 => line_buffer_Array_V_1_1_2_ce0,
        we0 => line_buffer_Array_V_1_1_2_we0,
        d0 => line_buffer_Array_V_1_0_2_q0,
        q0 => line_buffer_Array_V_1_1_2_q0);

    line_buffer_Array_V_1_0_3_U : component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s_line_buffer_Array_V_hbi
    generic map (
        DataWidth => 16,
        AddressRange => 39,
        AddressWidth => 6)
    port map (
        clk => ap_clk,
        reset => ap_rst,
        address0 => ap_const_lv6_26,
        ce0 => line_buffer_Array_V_1_0_3_ce0,
        we0 => line_buffer_Array_V_1_0_3_we0,
        d0 => in_elem_data_3_V_read,
        q0 => line_buffer_Array_V_1_0_3_q0);

    line_buffer_Array_V_1_1_3_U : component shift_line_buffer_array_ap_fixed_16_6_5_3_0_4u_config6_s_line_buffer_Array_V_hbi
    generic map (
        DataWidth => 16,
        AddressRange => 39,
        AddressWidth => 6)
    port map (
        clk => ap_clk,
        reset => ap_rst,
        address0 => ap_const_lv6_26,
        ce0 => line_buffer_Array_V_1_1_3_ce0,
        we0 => line_buffer_Array_V_1_1_3_we0,
        d0 => line_buffer_Array_V_1_0_3_q0,
        q0 => line_buffer_Array_V_1_1_3_q0);





    ap_CS_fsm_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_CS_fsm <= ap_ST_fsm_state1;
            else
                ap_CS_fsm <= ap_NS_fsm;
            end if;
        end if;
    end process;


    ap_NS_fsm_assign_proc : process (ap_start, ap_CS_fsm, ap_CS_fsm_state1, ap_ce)
    begin
        case ap_CS_fsm is
            when ap_ST_fsm_state1 => 
                ap_NS_fsm <= ap_ST_fsm_state1;
            when others =>  
                ap_NS_fsm <= "X";
        end case;
    end process;
    ap_CS_fsm_state1 <= ap_CS_fsm(0);

    ap_done_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if ((((ap_start = ap_const_logic_0) and (ap_const_logic_1 = ap_CS_fsm_state1)) or ((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce)))) then 
            ap_done <= ap_const_logic_1;
        else 
            ap_done <= ap_const_logic_0;
        end if; 
    end process;


    ap_idle_assign_proc : process(ap_start, ap_CS_fsm_state1)
    begin
        if (((ap_start = ap_const_logic_0) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            ap_idle <= ap_const_logic_1;
        else 
            ap_idle <= ap_const_logic_0;
        end if; 
    end process;


    ap_ready_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            ap_ready <= ap_const_logic_1;
        else 
            ap_ready <= ap_const_logic_0;
        end if; 
    end process;

    ap_return_0 <= kernel_window_4_V_read;
    ap_return_1 <= kernel_window_5_V_read;
    ap_return_10 <= kernel_window_30_V_read;
    ap_return_11 <= kernel_window_31_V_read;
    ap_return_12 <= kernel_window_8_V_read;
    ap_return_13 <= kernel_window_9_V_read;
    ap_return_14 <= kernel_window_10_V_read;
    ap_return_15 <= kernel_window_11_V_read;
    ap_return_16 <= line_buffer_Array_V_1_1_0_q0;
    ap_return_17 <= line_buffer_Array_V_1_1_1_q0;
    ap_return_18 <= line_buffer_Array_V_1_1_2_q0;
    ap_return_19 <= line_buffer_Array_V_1_1_3_q0;
    ap_return_2 <= kernel_window_6_V_read;
    ap_return_20 <= kernel_window_20_V_read;
    ap_return_21 <= kernel_window_21_V_read;
    ap_return_22 <= kernel_window_22_V_read;
    ap_return_23 <= kernel_window_23_V_read;
    ap_return_24 <= line_buffer_Array_V_1_0_0_q0;
    ap_return_25 <= line_buffer_Array_V_1_0_1_q0;
    ap_return_26 <= line_buffer_Array_V_1_0_2_q0;
    ap_return_27 <= line_buffer_Array_V_1_0_3_q0;
    ap_return_28 <= kernel_window_32_V_read;
    ap_return_29 <= kernel_window_33_V_read;
    ap_return_3 <= kernel_window_7_V_read;
    ap_return_30 <= kernel_window_34_V_read;
    ap_return_31 <= kernel_window_35_V_read;
    ap_return_32 <= in_elem_data_0_V_read;
    ap_return_33 <= in_elem_data_1_V_read;
    ap_return_34 <= in_elem_data_2_V_read;
    ap_return_35 <= in_elem_data_3_V_read;
    ap_return_4 <= kernel_window_16_V_read;
    ap_return_5 <= kernel_window_17_V_read;
    ap_return_6 <= kernel_window_18_V_read;
    ap_return_7 <= kernel_window_19_V_read;
    ap_return_8 <= kernel_window_28_V_read;
    ap_return_9 <= kernel_window_29_V_read;

    line_buffer_Array_V_1_0_0_ce0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_0_0_ce0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_0_0_ce0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_0_0_we0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_0_0_we0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_0_0_we0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_0_1_ce0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_0_1_ce0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_0_1_ce0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_0_1_we0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_0_1_we0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_0_1_we0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_0_2_ce0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_0_2_ce0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_0_2_ce0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_0_2_we0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_0_2_we0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_0_2_we0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_0_3_ce0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_0_3_ce0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_0_3_ce0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_0_3_we0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_0_3_we0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_0_3_we0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_1_0_ce0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_1_0_ce0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_1_0_ce0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_1_0_we0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_1_0_we0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_1_0_we0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_1_1_ce0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_1_1_ce0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_1_1_ce0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_1_1_we0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_1_1_we0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_1_1_we0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_1_2_ce0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_1_2_ce0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_1_2_ce0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_1_2_we0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_1_2_we0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_1_2_we0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_1_3_ce0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_1_3_ce0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_1_3_ce0 <= ap_const_logic_0;
        end if; 
    end process;


    line_buffer_Array_V_1_1_3_we0_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_ce)
    begin
        if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1) and (ap_const_logic_1 = ap_ce))) then 
            line_buffer_Array_V_1_1_3_we0 <= ap_const_logic_1;
        else 
            line_buffer_Array_V_1_1_3_we0 <= ap_const_logic_0;
        end if; 
    end process;

end behav;
