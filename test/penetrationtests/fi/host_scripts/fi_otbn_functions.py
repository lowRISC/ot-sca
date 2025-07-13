from communication.fi_otbn_commands import OTFIOtbn
from communication.chip import *
from communication.dut import DUT
import time

def char_beq(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_beq()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_bn_rshi(opentitantool, iterations, data):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_bn_rshi(data)
        response = target.read_response()
    # Return the result that is read out
    return response

def char_bn_sel(opentitantool, iterations, data):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_bn_sel(data)
        response = target.read_response()
    # Return the result that is read out
    return response

def char_bn_wsrr(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_bn_wsrr()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_bne(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_bne()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_dmem_access(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_dmem_access()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_dmem_write(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_dmem_write()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_dmem_op_loop(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_hardware_dmem_op_loop()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_reg_op_loop(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_hardware_reg_op_loop()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_jal(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_jal()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_lw(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_lw()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_mem(opentitantool, iterations, byte_offset, num_words, imem, dmem):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_mem(byte_offset, num_words, imem, dmem)
        response = target.read_response()
    # Return the result that is read out
    return response

def char_rf(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_rf()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_unrolled_dmem_op_loop(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_unrolled_dmem_op_loop()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_unrolled_reg_op_loop(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_char_unrolled_reg_op_loop()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_load_integrity(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_load_integrity()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_pc(opentitantool, iterations, pc):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otbnfi = OTFIOtbn(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
    for _ in range(iterations):
        otbnfi.otbn_pc(pc)
        response = target.read_response()
    # Return the result that is read out
    return response