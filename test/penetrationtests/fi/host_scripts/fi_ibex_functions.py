from target.communication.fi_ibex_commands import OTFIIbex
from target.chip import *
from target.dut import DUT
import time

def char_address_translation(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_address_translation()
        response = target.read_response()
    return response

def char_address_translation_config(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_address_translation_config()
        response = target.read_response()
    return response

def char_addi_single_beq(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_addi_single_beq()
        response = target.read_response()
    return response

def char_addi_single_beq_cm(opentitantool):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    ibexfi.ibex_char_addi_single_beq_cm()
    # This crashes the chip in a regular circumstance
    return target.check_crash_or_read_reponse()

def char_addi_single_beq_cm2(opentitantool):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    ibexfi.ibex_char_addi_single_beq_cm2()
    # This crashes the chip in a regular circumstance
    return target.check_crash_or_read_reponse()

def char_addi_single_beq_neg(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_addi_single_beq_neg()
        response = target.read_response()
    return response

def char_addi_single_bne(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_addi_single_bne()
        response = target.read_response()
    return response

def char_addi_single_bne_neg(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_addi_single_bne_neg()
        response = target.read_response()
    return response

def char_combi(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_combi()
        response = target.read_response()
    return response

def char_conditional_branch_beq(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_conditional_branch_beq()
        response = target.read_response()
    return response

def char_conditional_branch_bge(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_conditional_branch_bge()
        response = target.read_response()
    return response

def char_conditional_branch_bgeu(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_conditional_branch_bgeu()
        response = target.read_response()
    return response

def char_conditional_branch_blt(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_conditional_branch_blt()
        response = target.read_response()
    return response

def char_conditional_branch_bltu(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_conditional_branch_bltu()
        response = target.read_response()
    return response

def char_conditional_branch_bne(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_conditional_branch_bne()
        response = target.read_response()
    return response

def char_csr_read(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_csr_read()
        response = target.read_response()
    return response

def char_csr_write(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_csr_write()
        response = target.read_response()
    return response

def char_csr_combi(opentitantool, trigger, ref_values, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_csr_combi(trigger, ref_values)
        response = target.read_response()
    return response

def char_flash_read(opentitantool, flash_region, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_flash_read(flash_region)
        response = target.read_response()
    return response

def char_flash_write(opentitantool, flash_region, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_flash_write(flash_region)
        response = target.read_response()
    return response

def char_hardened_check_eq_complement_branch(opentitantool):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    ibexfi.ibex_char_hardened_check_eq_complement_branch()
    # This crashes the chip in a regular circumstance
    return target.check_crash_or_read_reponse()

def char_hardened_check_eq_unimp(opentitantool):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    ibexfi.ibex_char_hardened_check_eq_unimp()
    # This crashes the chip in a regular circumstance
    return target.check_crash_or_read_reponse()

def char_hardened_check_eq_2_unimps(opentitantool):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    ibexfi.ibex_char_hardened_check_eq_2_unimps()
    # This crashes the chip in a regular circumstance
    return target.check_crash_or_read_reponse()

def char_hardened_check_eq_3_unimps(opentitantool):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    ibexfi.ibex_char_hardened_check_eq_3_unimps()
    # This crashes the chip in a regular circumstance
    return target.check_crash_or_read_reponse()

def char_hardened_check_eq_4_unimps(opentitantool):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    ibexfi.ibex_char_hardened_check_eq_4_unimps()
    # This crashes the chip in a regular circumstance
    return target.check_crash_or_read_reponse()

def char_hardened_check_eq_5_unimps(opentitantool):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    ibexfi.ibex_char_hardened_check_eq_5_unimps()
    # This crashes the chip in a regular circumstance
    return target.check_crash_or_read_reponse()

def char_mem_op_loop(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_mem_op_loop()
        response = target.read_response()
    return response

def char_register_file(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_register_file()
        response = target.read_response()
    return response

def char_register_file_read(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_register_file_read()
        response = target.read_response()
    return response

def char_reg_op_loop(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_reg_op_loop()
        response = target.read_response()
    return response

def char_single_beq(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_single_beq()
        response = target.read_response()
    return response

def char_single_bne(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_single_bne()
        response = target.read_response()
    return response

def char_sram_read(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_sram_read()
        response = target.read_response()
    return response

def char_sram_read_ret(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_sram_read_ret()
        response = target.read_response()
    return response

def char_sram_static(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_sram_static()
        response = target.read_response()
    return response

def char_sram_write(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_sram_write()
        response = target.read_response()
    return response

def char_sram_write_read(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_sram_write_read()
        response = target.read_response()
    return response

def char_sram_write_read_alt(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_sram_write_read_alt()
        response = target.read_response()
    return response

def char_sram_write_static_unrolled(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_sram_write_static_unrolled()
        response = target.read_response()
    return response

def char_unconditional_branch(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_unconditional_branch()
        response = target.read_response()
    return response

def char_unconditional_branch_nop(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_unconditional_branch_nop()
        response = target.read_response()
    return response

def char_unrolled_mem_op_loop(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_unrolled_mem_op_loop()
        response = target.read_response()
    return response

def char_unrolled_reg_op_loop(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_unrolled_reg_op_loop()
        response = target.read_response()
    return response

def char_unrolled_reg_op_loop_chain(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_unrolled_reg_op_loop_chain()
        response = target.read_response()
    return response

def char_otp_data_read(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_otp_data_read()
        response = target.read_response()
    return response

def char_otp_read_lock(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexfi = OTFIIbex(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
    for _ in range(iterations):
        ibexfi.ibex_char_otp_read_lock()
        response = target.read_response()
    return response