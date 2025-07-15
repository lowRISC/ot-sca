from target.communication.sca_ibex_commands import OTIbex
from target.communication.sca_prng_commands import OTPRNG
from target.chip import *
from target.dut import DUT
import time

def char_combi_operations_batch(opentitantool, iterations, num_segments, trigger, fixed_data1, fixed_data2):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_combi_operations_batch(num_iterations = num_segments, trigger = trigger, fixed_data1 = fixed_data1, fixed_data2 = fixed_data2)
        response = target.read_response()
    return response

def char_combi_operations_batch_fvsr(opentitantool, iterations, num_segments, trigger, fixed_data1, fixed_data2):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_combi_operations_batch_fvsr(num_iterations = num_segments, trigger = trigger, fixed_data1 = fixed_data1, fixed_data2 = fixed_data2)
        response = target.read_response()
    return response

def char_register_file_read(opentitantool, iterations, data):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_register_file_read(data = data)
        response = target.read_response()
    return response

def char_register_file_read_batch_fvsr(opentitantool, iterations, data, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_register_file_read_batch_fvsr(data = data, num_segments = num_segments)
        response = target.read_response()
    return response

def char_register_file_read_batch_random(opentitantool, iterations, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_register_file_read_batch_random(num_segments = num_segments)
        response = target.read_response()
    return response

def char_register_file_write(opentitantool, iterations, data):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_register_file_write(data = data)
        response = target.read_response()
    return response

def char_register_file_write_batch_fvsr(opentitantool, iterations, data, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_register_file_write_batch_fvsr(data = data, num_segments = num_segments)
        response = target.read_response()
    return response

def char_register_file_write_batch_random(opentitantool, iterations, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_register_file_write_batch_random(num_segments = num_segments)
        response = target.read_response()
    return response

def char_tl_read(opentitantool, iterations, data):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_tl_read(data = data)
        response = target.read_response()
    return response

def char_tl_read_batch_fvsr(opentitantool, iterations, data, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_tl_read_batch_fvsr(data = data, num_segments = num_segments)
        response = target.read_response()
    return response

def char_tl_read_batch_fvsr_fix_address(opentitantool, iterations, data, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_tl_read_batch_fvsr_fix_address(data = data, num_segments = num_segments)
        response = target.read_response()
    return response

def char_tl_read_batch_random(opentitantool, iterations, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_tl_read_batch_random(num_segments = num_segments)
        response = target.read_response()
    return response

def char_tl_read_batch_random_fix_address(opentitantool, iterations, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_tl_read_batch_random_fix_address(num_segments = num_segments)
        response = target.read_response()
    return response

def char_tl_write(opentitantool, iterations, data):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_tl_write(data = data)
        response = target.read_response()
    return response

def char_tl_write_batch_fvsr(opentitantool, iterations, data, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_tl_write_batch_fvsr(data = data, num_segments = num_segments)
        response = target.read_response()
    return response

def char_tl_write_batch_fvsr_fix_address(opentitantool, iterations, data, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_tl_write_batch_fvsr_fix_address(data = data, num_segments = num_segments)
        response = target.read_response()
    return response

def char_tl_write_batch_random(opentitantool, iterations, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_tl_write_batch_random(num_segments = num_segments)
        response = target.read_response()
    return response

def char_tl_write_batch_random_fix_address(opentitantool, iterations, num_segments):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    # Seed the prng to make synchronized randomness
    # This is the same as using python rand with seed 1
    otprng = OTPRNG(target, protocol="ujson")
    otprng.seed_prng([1,0,0,0])

    ibexsca = OTIbex(target)
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    for _ in range(iterations):
        ibexsca.ibex_sca_tl_write_batch_random_fix_address(num_segments = num_segments)
        response = target.read_response()
    return response