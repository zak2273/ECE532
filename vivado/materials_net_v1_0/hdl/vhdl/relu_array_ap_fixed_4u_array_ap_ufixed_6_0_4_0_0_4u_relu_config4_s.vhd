-- ==============================================================
-- RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
-- Version: 2018.3
-- Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
-- 
-- ===========================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity relu_array_ap_fixed_4u_array_ap_ufixed_6_0_4_0_0_4u_relu_config4_s is
port (
    ap_clk : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC;
    ap_start : IN STD_LOGIC;
    start_full_n : IN STD_LOGIC;
    ap_done : OUT STD_LOGIC;
    ap_continue : IN STD_LOGIC;
    ap_idle : OUT STD_LOGIC;
    ap_ready : OUT STD_LOGIC;
    start_out : OUT STD_LOGIC;
    start_write : OUT STD_LOGIC;
    data_V_data_0_V_dout : IN STD_LOGIC_VECTOR (15 downto 0);
    data_V_data_0_V_empty_n : IN STD_LOGIC;
    data_V_data_0_V_read : OUT STD_LOGIC;
    data_V_data_1_V_dout : IN STD_LOGIC_VECTOR (15 downto 0);
    data_V_data_1_V_empty_n : IN STD_LOGIC;
    data_V_data_1_V_read : OUT STD_LOGIC;
    data_V_data_2_V_dout : IN STD_LOGIC_VECTOR (15 downto 0);
    data_V_data_2_V_empty_n : IN STD_LOGIC;
    data_V_data_2_V_read : OUT STD_LOGIC;
    data_V_data_3_V_dout : IN STD_LOGIC_VECTOR (15 downto 0);
    data_V_data_3_V_empty_n : IN STD_LOGIC;
    data_V_data_3_V_read : OUT STD_LOGIC;
    res_V_data_0_V_din : OUT STD_LOGIC_VECTOR (5 downto 0);
    res_V_data_0_V_full_n : IN STD_LOGIC;
    res_V_data_0_V_write : OUT STD_LOGIC;
    res_V_data_1_V_din : OUT STD_LOGIC_VECTOR (5 downto 0);
    res_V_data_1_V_full_n : IN STD_LOGIC;
    res_V_data_1_V_write : OUT STD_LOGIC;
    res_V_data_2_V_din : OUT STD_LOGIC_VECTOR (5 downto 0);
    res_V_data_2_V_full_n : IN STD_LOGIC;
    res_V_data_2_V_write : OUT STD_LOGIC;
    res_V_data_3_V_din : OUT STD_LOGIC_VECTOR (5 downto 0);
    res_V_data_3_V_full_n : IN STD_LOGIC;
    res_V_data_3_V_write : OUT STD_LOGIC );
end;


architecture behav of relu_array_ap_fixed_4u_array_ap_ufixed_6_0_4_0_0_4u_relu_config4_s is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_logic_0 : STD_LOGIC := '0';
    constant ap_ST_fsm_state1 : STD_LOGIC_VECTOR (2 downto 0) := "001";
    constant ap_ST_fsm_pp0_stage0 : STD_LOGIC_VECTOR (2 downto 0) := "010";
    constant ap_ST_fsm_state6 : STD_LOGIC_VECTOR (2 downto 0) := "100";
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_lv32_0 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000000";
    constant ap_const_lv32_1 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000001";
    constant ap_const_boolean_0 : BOOLEAN := false;
    constant ap_const_lv1_0 : STD_LOGIC_VECTOR (0 downto 0) := "0";
    constant ap_const_lv1_1 : STD_LOGIC_VECTOR (0 downto 0) := "1";
    constant ap_const_lv13_0 : STD_LOGIC_VECTOR (12 downto 0) := "0000000000000";
    constant ap_const_lv13_1235 : STD_LOGIC_VECTOR (12 downto 0) := "1001000110101";
    constant ap_const_lv13_1 : STD_LOGIC_VECTOR (12 downto 0) := "0000000000001";
    constant ap_const_lv3_0 : STD_LOGIC_VECTOR (2 downto 0) := "000";
    constant ap_const_lv32_A : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000001010";
    constant ap_const_lv32_F : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000001111";
    constant ap_const_lv6_3F : STD_LOGIC_VECTOR (5 downto 0) := "111111";
    constant ap_const_lv6_0 : STD_LOGIC_VECTOR (5 downto 0) := "000000";
    constant ap_const_lv16_0 : STD_LOGIC_VECTOR (15 downto 0) := "0000000000000000";
    constant ap_const_lv32_4 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000100";
    constant ap_const_lv32_9 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000001001";
    constant ap_const_lv32_3 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000011";
    constant ap_const_lv32_5 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000101";
    constant ap_const_lv32_2 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000010";

    signal real_start : STD_LOGIC;
    signal start_once_reg : STD_LOGIC := '0';
    signal ap_done_reg : STD_LOGIC := '0';
    signal ap_CS_fsm : STD_LOGIC_VECTOR (2 downto 0) := "001";
    attribute fsm_encoding : string;
    attribute fsm_encoding of ap_CS_fsm : signal is "none";
    signal ap_CS_fsm_state1 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state1 : signal is "none";
    signal internal_ap_ready : STD_LOGIC;
    signal data_V_data_0_V_blk_n : STD_LOGIC;
    signal ap_CS_fsm_pp0_stage0 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_pp0_stage0 : signal is "none";
    signal ap_enable_reg_pp0_iter1 : STD_LOGIC := '0';
    signal ap_block_pp0_stage0 : BOOLEAN;
    signal tmp_reg_767 : STD_LOGIC_VECTOR (0 downto 0);
    signal data_V_data_1_V_blk_n : STD_LOGIC;
    signal data_V_data_2_V_blk_n : STD_LOGIC;
    signal data_V_data_3_V_blk_n : STD_LOGIC;
    signal res_V_data_0_V_blk_n : STD_LOGIC;
    signal ap_enable_reg_pp0_iter3 : STD_LOGIC := '0';
    signal tmp_reg_767_pp0_iter2_reg : STD_LOGIC_VECTOR (0 downto 0);
    signal res_V_data_1_V_blk_n : STD_LOGIC;
    signal res_V_data_2_V_blk_n : STD_LOGIC;
    signal res_V_data_3_V_blk_n : STD_LOGIC;
    signal i_reg_208 : STD_LOGIC_VECTOR (12 downto 0);
    signal tmp_fu_219_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal ap_block_state2_pp0_stage0_iter0 : BOOLEAN;
    signal data_V_data_0_V0_status : STD_LOGIC;
    signal ap_block_state3_pp0_stage0_iter1 : BOOLEAN;
    signal ap_block_state4_pp0_stage0_iter2 : BOOLEAN;
    signal res_V_data_0_V1_status : STD_LOGIC;
    signal ap_block_state5_pp0_stage0_iter3 : BOOLEAN;
    signal ap_block_pp0_stage0_11001 : BOOLEAN;
    signal tmp_reg_767_pp0_iter1_reg : STD_LOGIC_VECTOR (0 downto 0);
    signal i_3_fu_225_p2 : STD_LOGIC_VECTOR (12 downto 0);
    signal ap_enable_reg_pp0_iter0 : STD_LOGIC := '0';
    signal tmp_data_0_V_reg_776 : STD_LOGIC_VECTOR (15 downto 0);
    signal tmp_data_1_V_reg_785 : STD_LOGIC_VECTOR (15 downto 0);
    signal tmp_data_2_V_reg_794 : STD_LOGIC_VECTOR (15 downto 0);
    signal tmp_data_3_V_reg_803 : STD_LOGIC_VECTOR (15 downto 0);
    signal r_4_fu_251_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_4_reg_812 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_ones_fu_267_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_ones_reg_817 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_zeros_fu_273_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_zeros_reg_822 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_4_1_fu_283_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_4_1_reg_827 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_ones_1_fu_299_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_ones_1_reg_832 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_zeros_1_fu_305_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_zeros_1_reg_837 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_4_2_fu_315_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_4_2_reg_842 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_ones_2_fu_331_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_ones_2_reg_847 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_zeros_2_fu_337_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_zeros_2_reg_852 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_4_3_fu_347_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_4_3_reg_857 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_ones_3_fu_363_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_ones_3_reg_862 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_zeros_3_fu_369_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal Range1_all_zeros_3_reg_867 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_data_0_V_3_fu_465_p3 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_data_0_V_3_reg_872 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_data_1_V_3_fu_563_p3 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_data_1_V_3_reg_877 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_data_2_V_3_fu_661_p3 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_data_2_V_3_reg_882 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_data_3_V_3_fu_759_p3 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_data_3_V_3_reg_887 : STD_LOGIC_VECTOR (5 downto 0);
    signal ap_block_state1 : BOOLEAN;
    signal ap_block_pp0_stage0_subdone : BOOLEAN;
    signal ap_condition_pp0_exit_iter0_state2 : STD_LOGIC;
    signal ap_enable_reg_pp0_iter2 : STD_LOGIC := '0';
    signal data_V_data_0_V0_update : STD_LOGIC;
    signal res_V_data_0_V1_update : STD_LOGIC;
    signal ap_block_pp0_stage0_01001 : BOOLEAN;
    signal tmp_85_fu_247_p1 : STD_LOGIC_VECTOR (2 downto 0);
    signal p_Result_5_fu_257_p4 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_90_fu_279_p1 : STD_LOGIC_VECTOR (2 downto 0);
    signal p_Result_24_1_fu_289_p4 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_95_fu_311_p1 : STD_LOGIC_VECTOR (2 downto 0);
    signal p_Result_24_2_fu_321_p4 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_100_fu_343_p1 : STD_LOGIC_VECTOR (2 downto 0);
    signal p_Result_24_3_fu_353_p4 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_84_fu_389_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_fu_403_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_87_fu_408_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_9_fu_415_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_100_cast_fu_421_p1 : STD_LOGIC_VECTOR (5 downto 0);
    signal p_Val2_s_fu_380_p4 : STD_LOGIC_VECTOR (5 downto 0);
    signal p_Val2_2_fu_425_p2 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_88_fu_431_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_86_fu_396_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal rev_fu_439_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal carry_3_fu_445_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal deleted_zeros_fu_451_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_6_fu_375_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal p_mux_fu_457_p3 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_89_fu_487_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_1_fu_501_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_92_fu_506_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_100_1_fu_513_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_100_1_cast_fu_519_p1 : STD_LOGIC_VECTOR (5 downto 0);
    signal p_Val2_12_1_fu_478_p4 : STD_LOGIC_VECTOR (5 downto 0);
    signal p_Val2_13_1_fu_523_p2 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_93_fu_529_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_91_fu_494_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal rev8_fu_537_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal carry_3_1_fu_543_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal deleted_zeros_1_fu_549_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_94_1_fu_473_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal p_mux_1_fu_555_p3 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_94_fu_585_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_2_fu_599_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_97_fu_604_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_100_2_fu_611_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_100_2_cast_fu_617_p1 : STD_LOGIC_VECTOR (5 downto 0);
    signal p_Val2_12_2_fu_576_p4 : STD_LOGIC_VECTOR (5 downto 0);
    signal p_Val2_13_2_fu_621_p2 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_98_fu_627_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_96_fu_592_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal rev7_fu_635_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal carry_3_2_fu_641_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal deleted_zeros_2_fu_647_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_94_2_fu_571_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal p_mux_2_fu_653_p3 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_99_fu_683_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_3_fu_697_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_102_fu_702_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_100_3_fu_709_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_100_3_cast_fu_715_p1 : STD_LOGIC_VECTOR (5 downto 0);
    signal p_Val2_12_3_fu_674_p4 : STD_LOGIC_VECTOR (5 downto 0);
    signal p_Val2_13_3_fu_719_p2 : STD_LOGIC_VECTOR (5 downto 0);
    signal tmp_103_fu_725_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_101_fu_690_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal rev9_fu_733_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal carry_3_3_fu_739_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal deleted_zeros_3_fu_745_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_94_3_fu_669_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal p_mux_3_fu_751_p3 : STD_LOGIC_VECTOR (5 downto 0);
    signal ap_CS_fsm_state6 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state6 : signal is "none";
    signal ap_NS_fsm : STD_LOGIC_VECTOR (2 downto 0);
    signal ap_idle_pp0 : STD_LOGIC;
    signal ap_enable_pp0 : STD_LOGIC;


begin




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


    ap_done_reg_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_done_reg <= ap_const_logic_0;
            else
                if ((ap_continue = ap_const_logic_1)) then 
                    ap_done_reg <= ap_const_logic_0;
                elsif ((ap_const_logic_1 = ap_CS_fsm_state6)) then 
                    ap_done_reg <= ap_const_logic_1;
                end if; 
            end if;
        end if;
    end process;


    ap_enable_reg_pp0_iter0_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_enable_reg_pp0_iter0 <= ap_const_logic_0;
            else
                if (((ap_const_logic_1 = ap_CS_fsm_pp0_stage0) and (ap_const_logic_1 = ap_condition_pp0_exit_iter0_state2) and (ap_const_boolean_0 = ap_block_pp0_stage0_subdone))) then 
                    ap_enable_reg_pp0_iter0 <= ap_const_logic_0;
                elsif ((not(((real_start = ap_const_logic_0) or (ap_done_reg = ap_const_logic_1))) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
                    ap_enable_reg_pp0_iter0 <= ap_const_logic_1;
                end if; 
            end if;
        end if;
    end process;


    ap_enable_reg_pp0_iter1_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_enable_reg_pp0_iter1 <= ap_const_logic_0;
            else
                if ((ap_const_boolean_0 = ap_block_pp0_stage0_subdone)) then
                    if ((ap_const_logic_1 = ap_condition_pp0_exit_iter0_state2)) then 
                        ap_enable_reg_pp0_iter1 <= (ap_const_logic_1 xor ap_condition_pp0_exit_iter0_state2);
                    elsif ((ap_const_boolean_1 = ap_const_boolean_1)) then 
                        ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
                    end if;
                end if; 
            end if;
        end if;
    end process;


    ap_enable_reg_pp0_iter2_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_enable_reg_pp0_iter2 <= ap_const_logic_0;
            else
                if ((ap_const_boolean_0 = ap_block_pp0_stage0_subdone)) then 
                    ap_enable_reg_pp0_iter2 <= ap_enable_reg_pp0_iter1;
                end if; 
            end if;
        end if;
    end process;


    ap_enable_reg_pp0_iter3_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_enable_reg_pp0_iter3 <= ap_const_logic_0;
            else
                if ((ap_const_boolean_0 = ap_block_pp0_stage0_subdone)) then 
                    ap_enable_reg_pp0_iter3 <= ap_enable_reg_pp0_iter2;
                elsif ((not(((real_start = ap_const_logic_0) or (ap_done_reg = ap_const_logic_1))) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
                    ap_enable_reg_pp0_iter3 <= ap_const_logic_0;
                end if; 
            end if;
        end if;
    end process;


    start_once_reg_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                start_once_reg <= ap_const_logic_0;
            else
                if (((internal_ap_ready = ap_const_logic_0) and (real_start = ap_const_logic_1))) then 
                    start_once_reg <= ap_const_logic_1;
                elsif ((internal_ap_ready = ap_const_logic_1)) then 
                    start_once_reg <= ap_const_logic_0;
                end if; 
            end if;
        end if;
    end process;


    i_reg_208_assign_proc : process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((tmp_fu_219_p2 = ap_const_lv1_0) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0) and (ap_enable_reg_pp0_iter0 = ap_const_logic_1) and (ap_const_boolean_0 = ap_block_pp0_stage0_11001))) then 
                i_reg_208 <= i_3_fu_225_p2;
            elsif ((not(((real_start = ap_const_logic_0) or (ap_done_reg = ap_const_logic_1))) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
                i_reg_208 <= ap_const_lv13_0;
            end if; 
        end if;
    end process;
    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((tmp_reg_767 = ap_const_lv1_0) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0) and (ap_const_boolean_0 = ap_block_pp0_stage0_11001))) then
                Range1_all_ones_1_reg_832 <= Range1_all_ones_1_fu_299_p2;
                Range1_all_ones_2_reg_847 <= Range1_all_ones_2_fu_331_p2;
                Range1_all_ones_3_reg_862 <= Range1_all_ones_3_fu_363_p2;
                Range1_all_ones_reg_817 <= Range1_all_ones_fu_267_p2;
                Range1_all_zeros_1_reg_837 <= Range1_all_zeros_1_fu_305_p2;
                Range1_all_zeros_2_reg_852 <= Range1_all_zeros_2_fu_337_p2;
                Range1_all_zeros_3_reg_867 <= Range1_all_zeros_3_fu_369_p2;
                Range1_all_zeros_reg_822 <= Range1_all_zeros_fu_273_p2;
                r_4_1_reg_827 <= r_4_1_fu_283_p2;
                r_4_2_reg_842 <= r_4_2_fu_315_p2;
                r_4_3_reg_857 <= r_4_3_fu_347_p2;
                r_4_reg_812 <= r_4_fu_251_p2;
                tmp_data_0_V_reg_776 <= data_V_data_0_V_dout;
                tmp_data_1_V_reg_785 <= data_V_data_1_V_dout;
                tmp_data_2_V_reg_794 <= data_V_data_2_V_dout;
                tmp_data_3_V_reg_803 <= data_V_data_3_V_dout;
            end if;
        end if;
    end process;
    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((tmp_reg_767_pp0_iter1_reg = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0_11001))) then
                tmp_data_0_V_3_reg_872 <= tmp_data_0_V_3_fu_465_p3;
                tmp_data_1_V_3_reg_877 <= tmp_data_1_V_3_fu_563_p3;
                tmp_data_2_V_3_reg_882 <= tmp_data_2_V_3_fu_661_p3;
                tmp_data_3_V_3_reg_887 <= tmp_data_3_V_3_fu_759_p3;
            end if;
        end if;
    end process;
    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((ap_const_logic_1 = ap_CS_fsm_pp0_stage0) and (ap_const_boolean_0 = ap_block_pp0_stage0_11001))) then
                tmp_reg_767 <= tmp_fu_219_p2;
                tmp_reg_767_pp0_iter1_reg <= tmp_reg_767;
            end if;
        end if;
    end process;
    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if ((ap_const_boolean_0 = ap_block_pp0_stage0_11001)) then
                tmp_reg_767_pp0_iter2_reg <= tmp_reg_767_pp0_iter1_reg;
            end if;
        end if;
    end process;

    ap_NS_fsm_assign_proc : process (real_start, ap_done_reg, ap_CS_fsm, ap_CS_fsm_state1, ap_enable_reg_pp0_iter1, ap_enable_reg_pp0_iter3, tmp_fu_219_p2, ap_enable_reg_pp0_iter0, ap_block_pp0_stage0_subdone, ap_enable_reg_pp0_iter2)
    begin
        case ap_CS_fsm is
            when ap_ST_fsm_state1 => 
                if ((not(((real_start = ap_const_logic_0) or (ap_done_reg = ap_const_logic_1))) and (ap_const_logic_1 = ap_CS_fsm_state1))) then
                    ap_NS_fsm <= ap_ST_fsm_pp0_stage0;
                else
                    ap_NS_fsm <= ap_ST_fsm_state1;
                end if;
            when ap_ST_fsm_pp0_stage0 => 
                if ((not(((ap_enable_reg_pp0_iter1 = ap_const_logic_0) and (ap_enable_reg_pp0_iter0 = ap_const_logic_1) and (ap_const_boolean_0 = ap_block_pp0_stage0_subdone) and (tmp_fu_219_p2 = ap_const_lv1_1))) and not(((ap_enable_reg_pp0_iter3 = ap_const_logic_1) and (ap_enable_reg_pp0_iter2 = ap_const_logic_0) and (ap_const_boolean_0 = ap_block_pp0_stage0_subdone))))) then
                    ap_NS_fsm <= ap_ST_fsm_pp0_stage0;
                elsif ((((ap_enable_reg_pp0_iter3 = ap_const_logic_1) and (ap_enable_reg_pp0_iter2 = ap_const_logic_0) and (ap_const_boolean_0 = ap_block_pp0_stage0_subdone)) or ((ap_enable_reg_pp0_iter1 = ap_const_logic_0) and (ap_enable_reg_pp0_iter0 = ap_const_logic_1) and (ap_const_boolean_0 = ap_block_pp0_stage0_subdone) and (tmp_fu_219_p2 = ap_const_lv1_1)))) then
                    ap_NS_fsm <= ap_ST_fsm_state6;
                else
                    ap_NS_fsm <= ap_ST_fsm_pp0_stage0;
                end if;
            when ap_ST_fsm_state6 => 
                ap_NS_fsm <= ap_ST_fsm_state1;
            when others =>  
                ap_NS_fsm <= "XXX";
        end case;
    end process;
    Range1_all_ones_1_fu_299_p2 <= "1" when (p_Result_24_1_fu_289_p4 = ap_const_lv6_3F) else "0";
    Range1_all_ones_2_fu_331_p2 <= "1" when (p_Result_24_2_fu_321_p4 = ap_const_lv6_3F) else "0";
    Range1_all_ones_3_fu_363_p2 <= "1" when (p_Result_24_3_fu_353_p4 = ap_const_lv6_3F) else "0";
    Range1_all_ones_fu_267_p2 <= "1" when (p_Result_5_fu_257_p4 = ap_const_lv6_3F) else "0";
    Range1_all_zeros_1_fu_305_p2 <= "1" when (p_Result_24_1_fu_289_p4 = ap_const_lv6_0) else "0";
    Range1_all_zeros_2_fu_337_p2 <= "1" when (p_Result_24_2_fu_321_p4 = ap_const_lv6_0) else "0";
    Range1_all_zeros_3_fu_369_p2 <= "1" when (p_Result_24_3_fu_353_p4 = ap_const_lv6_0) else "0";
    Range1_all_zeros_fu_273_p2 <= "1" when (p_Result_5_fu_257_p4 = ap_const_lv6_0) else "0";
    ap_CS_fsm_pp0_stage0 <= ap_CS_fsm(1);
    ap_CS_fsm_state1 <= ap_CS_fsm(0);
    ap_CS_fsm_state6 <= ap_CS_fsm(2);
        ap_block_pp0_stage0 <= not((ap_const_boolean_1 = ap_const_boolean_1));

    ap_block_pp0_stage0_01001_assign_proc : process(ap_enable_reg_pp0_iter1, tmp_reg_767, ap_enable_reg_pp0_iter3, tmp_reg_767_pp0_iter2_reg, data_V_data_0_V0_status, res_V_data_0_V1_status)
    begin
                ap_block_pp0_stage0_01001 <= (((tmp_reg_767_pp0_iter2_reg = ap_const_lv1_0) and (res_V_data_0_V1_status = ap_const_logic_0) and (ap_enable_reg_pp0_iter3 = ap_const_logic_1)) or ((tmp_reg_767 = ap_const_lv1_0) and (data_V_data_0_V0_status = ap_const_logic_0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1)));
    end process;


    ap_block_pp0_stage0_11001_assign_proc : process(ap_enable_reg_pp0_iter1, tmp_reg_767, ap_enable_reg_pp0_iter3, tmp_reg_767_pp0_iter2_reg, data_V_data_0_V0_status, res_V_data_0_V1_status)
    begin
                ap_block_pp0_stage0_11001 <= (((tmp_reg_767_pp0_iter2_reg = ap_const_lv1_0) and (res_V_data_0_V1_status = ap_const_logic_0) and (ap_enable_reg_pp0_iter3 = ap_const_logic_1)) or ((tmp_reg_767 = ap_const_lv1_0) and (data_V_data_0_V0_status = ap_const_logic_0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1)));
    end process;


    ap_block_pp0_stage0_subdone_assign_proc : process(ap_enable_reg_pp0_iter1, tmp_reg_767, ap_enable_reg_pp0_iter3, tmp_reg_767_pp0_iter2_reg, data_V_data_0_V0_status, res_V_data_0_V1_status)
    begin
                ap_block_pp0_stage0_subdone <= (((tmp_reg_767_pp0_iter2_reg = ap_const_lv1_0) and (res_V_data_0_V1_status = ap_const_logic_0) and (ap_enable_reg_pp0_iter3 = ap_const_logic_1)) or ((tmp_reg_767 = ap_const_lv1_0) and (data_V_data_0_V0_status = ap_const_logic_0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1)));
    end process;


    ap_block_state1_assign_proc : process(real_start, ap_done_reg)
    begin
                ap_block_state1 <= ((real_start = ap_const_logic_0) or (ap_done_reg = ap_const_logic_1));
    end process;

        ap_block_state2_pp0_stage0_iter0 <= not((ap_const_boolean_1 = ap_const_boolean_1));

    ap_block_state3_pp0_stage0_iter1_assign_proc : process(tmp_reg_767, data_V_data_0_V0_status)
    begin
                ap_block_state3_pp0_stage0_iter1 <= ((tmp_reg_767 = ap_const_lv1_0) and (data_V_data_0_V0_status = ap_const_logic_0));
    end process;

        ap_block_state4_pp0_stage0_iter2 <= not((ap_const_boolean_1 = ap_const_boolean_1));

    ap_block_state5_pp0_stage0_iter3_assign_proc : process(tmp_reg_767_pp0_iter2_reg, res_V_data_0_V1_status)
    begin
                ap_block_state5_pp0_stage0_iter3 <= ((tmp_reg_767_pp0_iter2_reg = ap_const_lv1_0) and (res_V_data_0_V1_status = ap_const_logic_0));
    end process;


    ap_condition_pp0_exit_iter0_state2_assign_proc : process(tmp_fu_219_p2)
    begin
        if ((tmp_fu_219_p2 = ap_const_lv1_1)) then 
            ap_condition_pp0_exit_iter0_state2 <= ap_const_logic_1;
        else 
            ap_condition_pp0_exit_iter0_state2 <= ap_const_logic_0;
        end if; 
    end process;


    ap_done_assign_proc : process(ap_done_reg, ap_CS_fsm_state6)
    begin
        if ((ap_const_logic_1 = ap_CS_fsm_state6)) then 
            ap_done <= ap_const_logic_1;
        else 
            ap_done <= ap_done_reg;
        end if; 
    end process;

    ap_enable_pp0 <= (ap_idle_pp0 xor ap_const_logic_1);

    ap_idle_assign_proc : process(real_start, ap_CS_fsm_state1)
    begin
        if (((real_start = ap_const_logic_0) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            ap_idle <= ap_const_logic_1;
        else 
            ap_idle <= ap_const_logic_0;
        end if; 
    end process;


    ap_idle_pp0_assign_proc : process(ap_enable_reg_pp0_iter1, ap_enable_reg_pp0_iter3, ap_enable_reg_pp0_iter0, ap_enable_reg_pp0_iter2)
    begin
        if (((ap_enable_reg_pp0_iter3 = ap_const_logic_0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_0) and (ap_enable_reg_pp0_iter2 = ap_const_logic_0) and (ap_enable_reg_pp0_iter0 = ap_const_logic_0))) then 
            ap_idle_pp0 <= ap_const_logic_1;
        else 
            ap_idle_pp0 <= ap_const_logic_0;
        end if; 
    end process;

    ap_ready <= internal_ap_ready;
    carry_3_1_fu_543_p2 <= (tmp_91_fu_494_p3 and rev8_fu_537_p2);
    carry_3_2_fu_641_p2 <= (tmp_96_fu_592_p3 and rev7_fu_635_p2);
    carry_3_3_fu_739_p2 <= (tmp_101_fu_690_p3 and rev9_fu_733_p2);
    carry_3_fu_445_p2 <= (tmp_86_fu_396_p3 and rev_fu_439_p2);
    data_V_data_0_V0_status <= (data_V_data_3_V_empty_n and data_V_data_2_V_empty_n and data_V_data_1_V_empty_n and data_V_data_0_V_empty_n);

    data_V_data_0_V0_update_assign_proc : process(ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, tmp_reg_767, ap_block_pp0_stage0_11001)
    begin
        if (((tmp_reg_767 = ap_const_lv1_0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0) and (ap_const_boolean_0 = ap_block_pp0_stage0_11001))) then 
            data_V_data_0_V0_update <= ap_const_logic_1;
        else 
            data_V_data_0_V0_update <= ap_const_logic_0;
        end if; 
    end process;


    data_V_data_0_V_blk_n_assign_proc : process(data_V_data_0_V_empty_n, ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, ap_block_pp0_stage0, tmp_reg_767)
    begin
        if (((tmp_reg_767 = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            data_V_data_0_V_blk_n <= data_V_data_0_V_empty_n;
        else 
            data_V_data_0_V_blk_n <= ap_const_logic_1;
        end if; 
    end process;

    data_V_data_0_V_read <= data_V_data_0_V0_update;

    data_V_data_1_V_blk_n_assign_proc : process(data_V_data_1_V_empty_n, ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, ap_block_pp0_stage0, tmp_reg_767)
    begin
        if (((tmp_reg_767 = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            data_V_data_1_V_blk_n <= data_V_data_1_V_empty_n;
        else 
            data_V_data_1_V_blk_n <= ap_const_logic_1;
        end if; 
    end process;

    data_V_data_1_V_read <= data_V_data_0_V0_update;

    data_V_data_2_V_blk_n_assign_proc : process(data_V_data_2_V_empty_n, ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, ap_block_pp0_stage0, tmp_reg_767)
    begin
        if (((tmp_reg_767 = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            data_V_data_2_V_blk_n <= data_V_data_2_V_empty_n;
        else 
            data_V_data_2_V_blk_n <= ap_const_logic_1;
        end if; 
    end process;

    data_V_data_2_V_read <= data_V_data_0_V0_update;

    data_V_data_3_V_blk_n_assign_proc : process(data_V_data_3_V_empty_n, ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, ap_block_pp0_stage0, tmp_reg_767)
    begin
        if (((tmp_reg_767 = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            data_V_data_3_V_blk_n <= data_V_data_3_V_empty_n;
        else 
            data_V_data_3_V_blk_n <= ap_const_logic_1;
        end if; 
    end process;

    data_V_data_3_V_read <= data_V_data_0_V0_update;
    deleted_zeros_1_fu_549_p3 <= 
        Range1_all_ones_1_reg_832 when (carry_3_1_fu_543_p2(0) = '1') else 
        Range1_all_zeros_1_reg_837;
    deleted_zeros_2_fu_647_p3 <= 
        Range1_all_ones_2_reg_847 when (carry_3_2_fu_641_p2(0) = '1') else 
        Range1_all_zeros_2_reg_852;
    deleted_zeros_3_fu_745_p3 <= 
        Range1_all_ones_3_reg_862 when (carry_3_3_fu_739_p2(0) = '1') else 
        Range1_all_zeros_3_reg_867;
    deleted_zeros_fu_451_p3 <= 
        Range1_all_ones_reg_817 when (carry_3_fu_445_p2(0) = '1') else 
        Range1_all_zeros_reg_822;
    i_3_fu_225_p2 <= std_logic_vector(unsigned(i_reg_208) + unsigned(ap_const_lv13_1));

    internal_ap_ready_assign_proc : process(ap_CS_fsm_state6)
    begin
        if ((ap_const_logic_1 = ap_CS_fsm_state6)) then 
            internal_ap_ready <= ap_const_logic_1;
        else 
            internal_ap_ready <= ap_const_logic_0;
        end if; 
    end process;

    p_Result_24_1_fu_289_p4 <= data_V_data_1_V_dout(15 downto 10);
    p_Result_24_2_fu_321_p4 <= data_V_data_2_V_dout(15 downto 10);
    p_Result_24_3_fu_353_p4 <= data_V_data_3_V_dout(15 downto 10);
    p_Result_5_fu_257_p4 <= data_V_data_0_V_dout(15 downto 10);
    p_Val2_12_1_fu_478_p4 <= tmp_data_1_V_reg_785(9 downto 4);
    p_Val2_12_2_fu_576_p4 <= tmp_data_2_V_reg_794(9 downto 4);
    p_Val2_12_3_fu_674_p4 <= tmp_data_3_V_reg_803(9 downto 4);
    p_Val2_13_1_fu_523_p2 <= std_logic_vector(unsigned(tmp_100_1_cast_fu_519_p1) + unsigned(p_Val2_12_1_fu_478_p4));
    p_Val2_13_2_fu_621_p2 <= std_logic_vector(unsigned(tmp_100_2_cast_fu_617_p1) + unsigned(p_Val2_12_2_fu_576_p4));
    p_Val2_13_3_fu_719_p2 <= std_logic_vector(unsigned(tmp_100_3_cast_fu_715_p1) + unsigned(p_Val2_12_3_fu_674_p4));
    p_Val2_2_fu_425_p2 <= std_logic_vector(unsigned(tmp_100_cast_fu_421_p1) + unsigned(p_Val2_s_fu_380_p4));
    p_Val2_s_fu_380_p4 <= tmp_data_0_V_reg_776(9 downto 4);
    p_mux_1_fu_555_p3 <= 
        p_Val2_13_1_fu_523_p2 when (deleted_zeros_1_fu_549_p3(0) = '1') else 
        ap_const_lv6_3F;
    p_mux_2_fu_653_p3 <= 
        p_Val2_13_2_fu_621_p2 when (deleted_zeros_2_fu_647_p3(0) = '1') else 
        ap_const_lv6_3F;
    p_mux_3_fu_751_p3 <= 
        p_Val2_13_3_fu_719_p2 when (deleted_zeros_3_fu_745_p3(0) = '1') else 
        ap_const_lv6_3F;
    p_mux_fu_457_p3 <= 
        p_Val2_2_fu_425_p2 when (deleted_zeros_fu_451_p3(0) = '1') else 
        ap_const_lv6_3F;
    r_1_fu_501_p2 <= (tmp_89_fu_487_p3 or r_4_1_reg_827);
    r_2_fu_599_p2 <= (tmp_94_fu_585_p3 or r_4_2_reg_842);
    r_3_fu_697_p2 <= (tmp_99_fu_683_p3 or r_4_3_reg_857);
    r_4_1_fu_283_p2 <= "0" when (tmp_90_fu_279_p1 = ap_const_lv3_0) else "1";
    r_4_2_fu_315_p2 <= "0" when (tmp_95_fu_311_p1 = ap_const_lv3_0) else "1";
    r_4_3_fu_347_p2 <= "0" when (tmp_100_fu_343_p1 = ap_const_lv3_0) else "1";
    r_4_fu_251_p2 <= "0" when (tmp_85_fu_247_p1 = ap_const_lv3_0) else "1";
    r_fu_403_p2 <= (tmp_84_fu_389_p3 or r_4_reg_812);

    real_start_assign_proc : process(ap_start, start_full_n, start_once_reg)
    begin
        if (((start_once_reg = ap_const_logic_0) and (start_full_n = ap_const_logic_0))) then 
            real_start <= ap_const_logic_0;
        else 
            real_start <= ap_start;
        end if; 
    end process;

    res_V_data_0_V1_status <= (res_V_data_3_V_full_n and res_V_data_2_V_full_n and res_V_data_1_V_full_n and res_V_data_0_V_full_n);

    res_V_data_0_V1_update_assign_proc : process(ap_enable_reg_pp0_iter3, tmp_reg_767_pp0_iter2_reg, ap_block_pp0_stage0_11001)
    begin
        if (((tmp_reg_767_pp0_iter2_reg = ap_const_lv1_0) and (ap_enable_reg_pp0_iter3 = ap_const_logic_1) and (ap_const_boolean_0 = ap_block_pp0_stage0_11001))) then 
            res_V_data_0_V1_update <= ap_const_logic_1;
        else 
            res_V_data_0_V1_update <= ap_const_logic_0;
        end if; 
    end process;


    res_V_data_0_V_blk_n_assign_proc : process(res_V_data_0_V_full_n, ap_block_pp0_stage0, ap_enable_reg_pp0_iter3, tmp_reg_767_pp0_iter2_reg)
    begin
        if (((tmp_reg_767_pp0_iter2_reg = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0) and (ap_enable_reg_pp0_iter3 = ap_const_logic_1))) then 
            res_V_data_0_V_blk_n <= res_V_data_0_V_full_n;
        else 
            res_V_data_0_V_blk_n <= ap_const_logic_1;
        end if; 
    end process;

    res_V_data_0_V_din <= tmp_data_0_V_3_reg_872;
    res_V_data_0_V_write <= res_V_data_0_V1_update;

    res_V_data_1_V_blk_n_assign_proc : process(res_V_data_1_V_full_n, ap_block_pp0_stage0, ap_enable_reg_pp0_iter3, tmp_reg_767_pp0_iter2_reg)
    begin
        if (((tmp_reg_767_pp0_iter2_reg = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0) and (ap_enable_reg_pp0_iter3 = ap_const_logic_1))) then 
            res_V_data_1_V_blk_n <= res_V_data_1_V_full_n;
        else 
            res_V_data_1_V_blk_n <= ap_const_logic_1;
        end if; 
    end process;

    res_V_data_1_V_din <= tmp_data_1_V_3_reg_877;
    res_V_data_1_V_write <= res_V_data_0_V1_update;

    res_V_data_2_V_blk_n_assign_proc : process(res_V_data_2_V_full_n, ap_block_pp0_stage0, ap_enable_reg_pp0_iter3, tmp_reg_767_pp0_iter2_reg)
    begin
        if (((tmp_reg_767_pp0_iter2_reg = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0) and (ap_enable_reg_pp0_iter3 = ap_const_logic_1))) then 
            res_V_data_2_V_blk_n <= res_V_data_2_V_full_n;
        else 
            res_V_data_2_V_blk_n <= ap_const_logic_1;
        end if; 
    end process;

    res_V_data_2_V_din <= tmp_data_2_V_3_reg_882;
    res_V_data_2_V_write <= res_V_data_0_V1_update;

    res_V_data_3_V_blk_n_assign_proc : process(res_V_data_3_V_full_n, ap_block_pp0_stage0, ap_enable_reg_pp0_iter3, tmp_reg_767_pp0_iter2_reg)
    begin
        if (((tmp_reg_767_pp0_iter2_reg = ap_const_lv1_0) and (ap_const_boolean_0 = ap_block_pp0_stage0) and (ap_enable_reg_pp0_iter3 = ap_const_logic_1))) then 
            res_V_data_3_V_blk_n <= res_V_data_3_V_full_n;
        else 
            res_V_data_3_V_blk_n <= ap_const_logic_1;
        end if; 
    end process;

    res_V_data_3_V_din <= tmp_data_3_V_3_reg_887;
    res_V_data_3_V_write <= res_V_data_0_V1_update;
    rev7_fu_635_p2 <= (tmp_98_fu_627_p3 xor ap_const_lv1_1);
    rev8_fu_537_p2 <= (tmp_93_fu_529_p3 xor ap_const_lv1_1);
    rev9_fu_733_p2 <= (tmp_103_fu_725_p3 xor ap_const_lv1_1);
    rev_fu_439_p2 <= (tmp_88_fu_431_p3 xor ap_const_lv1_1);
    start_out <= real_start;

    start_write_assign_proc : process(real_start, start_once_reg)
    begin
        if (((start_once_reg = ap_const_logic_0) and (real_start = ap_const_logic_1))) then 
            start_write <= ap_const_logic_1;
        else 
            start_write <= ap_const_logic_0;
        end if; 
    end process;

    tmp_100_1_cast_fu_519_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(tmp_100_1_fu_513_p2),6));
    tmp_100_1_fu_513_p2 <= (tmp_92_fu_506_p3 and r_1_fu_501_p2);
    tmp_100_2_cast_fu_617_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(tmp_100_2_fu_611_p2),6));
    tmp_100_2_fu_611_p2 <= (tmp_97_fu_604_p3 and r_2_fu_599_p2);
    tmp_100_3_cast_fu_715_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(tmp_100_3_fu_709_p2),6));
    tmp_100_3_fu_709_p2 <= (tmp_102_fu_702_p3 and r_3_fu_697_p2);
    tmp_100_cast_fu_421_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(tmp_9_fu_415_p2),6));
    tmp_100_fu_343_p1 <= data_V_data_3_V_dout(3 - 1 downto 0);
    tmp_101_fu_690_p3 <= tmp_data_3_V_reg_803(9 downto 9);
    tmp_102_fu_702_p3 <= tmp_data_3_V_reg_803(3 downto 3);
    tmp_103_fu_725_p3 <= p_Val2_13_3_fu_719_p2(5 downto 5);
    tmp_6_fu_375_p2 <= "1" when (signed(tmp_data_0_V_reg_776) > signed(ap_const_lv16_0)) else "0";
    tmp_84_fu_389_p3 <= tmp_data_0_V_reg_776(4 downto 4);
    tmp_85_fu_247_p1 <= data_V_data_0_V_dout(3 - 1 downto 0);
    tmp_86_fu_396_p3 <= tmp_data_0_V_reg_776(9 downto 9);
    tmp_87_fu_408_p3 <= tmp_data_0_V_reg_776(3 downto 3);
    tmp_88_fu_431_p3 <= p_Val2_2_fu_425_p2(5 downto 5);
    tmp_89_fu_487_p3 <= tmp_data_1_V_reg_785(4 downto 4);
    tmp_90_fu_279_p1 <= data_V_data_1_V_dout(3 - 1 downto 0);
    tmp_91_fu_494_p3 <= tmp_data_1_V_reg_785(9 downto 9);
    tmp_92_fu_506_p3 <= tmp_data_1_V_reg_785(3 downto 3);
    tmp_93_fu_529_p3 <= p_Val2_13_1_fu_523_p2(5 downto 5);
    tmp_94_1_fu_473_p2 <= "1" when (signed(tmp_data_1_V_reg_785) > signed(ap_const_lv16_0)) else "0";
    tmp_94_2_fu_571_p2 <= "1" when (signed(tmp_data_2_V_reg_794) > signed(ap_const_lv16_0)) else "0";
    tmp_94_3_fu_669_p2 <= "1" when (signed(tmp_data_3_V_reg_803) > signed(ap_const_lv16_0)) else "0";
    tmp_94_fu_585_p3 <= tmp_data_2_V_reg_794(4 downto 4);
    tmp_95_fu_311_p1 <= data_V_data_2_V_dout(3 - 1 downto 0);
    tmp_96_fu_592_p3 <= tmp_data_2_V_reg_794(9 downto 9);
    tmp_97_fu_604_p3 <= tmp_data_2_V_reg_794(3 downto 3);
    tmp_98_fu_627_p3 <= p_Val2_13_2_fu_621_p2(5 downto 5);
    tmp_99_fu_683_p3 <= tmp_data_3_V_reg_803(4 downto 4);
    tmp_9_fu_415_p2 <= (tmp_87_fu_408_p3 and r_fu_403_p2);
    tmp_data_0_V_3_fu_465_p3 <= 
        p_mux_fu_457_p3 when (tmp_6_fu_375_p2(0) = '1') else 
        ap_const_lv6_0;
    tmp_data_1_V_3_fu_563_p3 <= 
        p_mux_1_fu_555_p3 when (tmp_94_1_fu_473_p2(0) = '1') else 
        ap_const_lv6_0;
    tmp_data_2_V_3_fu_661_p3 <= 
        p_mux_2_fu_653_p3 when (tmp_94_2_fu_571_p2(0) = '1') else 
        ap_const_lv6_0;
    tmp_data_3_V_3_fu_759_p3 <= 
        p_mux_3_fu_751_p3 when (tmp_94_3_fu_669_p2(0) = '1') else 
        ap_const_lv6_0;
    tmp_fu_219_p2 <= "1" when (i_reg_208 = ap_const_lv13_1235) else "0";
end behav;