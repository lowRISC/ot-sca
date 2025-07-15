
from test.penetrationtests.sca.host_scripts.sca_ibex_functions import *
from target.communication.sca_ibex_commands import OTIbex
from python.runfiles import Runfiles
from target.chip import *
from test.penetrationtests.util.utils import *
import os
import json
import random

ignored_keys_set = set([])

def reset_test(opentitantool, target):
    reset_target(opentitantool)
    while True:
        read_line = str(target.readline().decode().strip())
        if len(read_line) > 0:
            if "firmware_sca.c" in read_line:
                return True
        else:
            return False


def init_test(opentitantool, target):
    ibexsca = OTIbex(target)
    device_id, owner_page, boot_log, boot_measurements, version = ibexsca.init()
    device_id_json = json.loads(device_id)
    owner_page_json = json.loads(owner_page)
    boot_log_json = json.loads(boot_log)
    boot_measurements_json = json.loads(boot_measurements)

    expected_device_id_keys = {
        "device_id",
        "rom_digest",
        "icache_en",
        "dummy_instr_en",
        "clock_jitter_locked",
        "clock_jitter_en",
        "sram_main_readback_locked",
        "sram_main_readback_en",
        "sram_ret_readback_locked",
        "sram_ret_readback_en"
    }
    actual_device_id_keys = set(device_id_json.keys())

    if not actual_device_id_keys == expected_device_id_keys:
        print("device_id keys do not match the expected set.")
        print(f"Expected: {expected_device_id_keys}")
        print(f"Actual: {actual_device_id_keys}")
        return False

    expected_owner_page_keys = {
        "config_version",
        "sram_exec_mode",
        "ownership_key_alg",
        "update_mode",
        "min_security_version_bl0",
        "lock_constraint"
    }
    actual_owner_page_keys = set(owner_page_json.keys())
    if not expected_owner_page_keys == actual_owner_page_keys:
        print("owner_page keys do not match the expected set.")
        print(f"Expected: {expected_owner_page_keys}")
        print(f"Actual: {actual_owner_page_keys}")
        return False

    expected_boot_log_keys = {
        "digest",
        "identifier",
        "scm_revision_low",
        "scm_revision_high",
        "rom_ext_slot",
        "rom_ext_major",
        "rom_ext_minor",
        "rom_ext_size",
        "bl0_slot",
        "ownership_state",
        "ownership_transfers",
        "rom_ext_min_sec_ver",
        "bl0_min_sec_ver",
        "primary_bl0_slot",
        "retention_ram_initialized"
    }
    actual_boot_log_keys = set(boot_log_json.keys())
    if not expected_boot_log_keys == actual_boot_log_keys:
        print("boot_log keys do not match the expected set.")
        print(f"Expected: {expected_boot_log_keys}")
        print(f"Actual: {actual_boot_log_keys}")
        return False

    expected_boot_measurements_keys = {
        "bl0",
        "rom_ext"
    }
    actual_boot_measurements_keys = set(boot_measurements_json.keys())
    if not expected_boot_measurements_keys == actual_boot_measurements_keys:
        print("boot_measurements keys do not match the expected set.")
        print(f"Expected: {expected_boot_measurements_keys}")
        print(f"Actual: {actual_boot_measurements_keys}")
        return False

    if "PENTEST" not in version:
      print("Did not receive a PENTEST version.")
      print(f"Actual: {version}")
      return False

    return True

def char_combi_operations_batch_test(opentitantool_path, iterations, num_segments):
    trigger = 0
    fixed_data1 = 11
    fixed_data2 = 11
    actual_result = char_combi_operations_batch(opentitantool_path, iterations, num_segments, trigger, fixed_data1, fixed_data2)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch gave an unexpected result")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":[0,0,0,0,0,0,0,0,0,0,0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch failed")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    trigger = 32
    fixed_data1 = 11
    fixed_data2 = 11
    actual_result = char_combi_operations_batch(opentitantool_path, iterations, num_segments, trigger, fixed_data1, fixed_data2)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch gave an unexpected result")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False
    xor = fixed_data1^fixed_data2
    add = (fixed_data1+fixed_data2)&0xFFFFFFFF
    sub = (fixed_data1-fixed_data2)&0xFFFFFFFF
    shift_operand = (fixed_data2&0xFFFFFFFF) % 32
    shift = ( (fixed_data1 << shift_operand)&0xFFFFFFFF | (fixed_data1 >> (32 - shift_operand)&0xFFFFFFFF))
    mult = (fixed_data1 * fixed_data2)&0xFFFFFFFF
    if fixed_data2 == 0:
        div = 0xFFFFFFFF
    elif to_signed32(fixed_data1) == -2147483648 and to_signed32(fixed_data2) == -1:
        div = 0x80000000
    else:
        div = int(to_signed32(fixed_data1) / to_signed32(fixed_data2)) & 0xFFFFFFFF

    expected_result_json = {
        "result": [
            0, 0, 0, 0, 0, div, 0, 0, 0, 0, 0, 0
        ]
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch failed")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    trigger = 256
    fixed_data1 = 11
    fixed_data2 = 11
    actual_result = char_combi_operations_batch(opentitantool_path, iterations, num_segments, trigger, fixed_data1, fixed_data2)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch gave an unexpected result")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False

    expected_result_json = {
        "result": [
            0, 0, 0, 0, 0, 0, 0, 0, fixed_data2, 0, 0, 0
        ]
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch failed")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    trigger = 4095
    fixed_data1 = 11
    fixed_data2 = 11
    actual_result = char_combi_operations_batch(opentitantool_path, iterations, num_segments, trigger, fixed_data1, fixed_data2)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch gave an unexpected result")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False

    xor = fixed_data1^fixed_data2
    add = (fixed_data1+fixed_data2)&0xFFFFFFFF
    sub = (fixed_data1-fixed_data2)&0xFFFFFFFF
    shift_operand = (fixed_data2&0xFFFFFFFF) % 32
    shift = ( (fixed_data1 << shift_operand)&0xFFFFFFFF | (fixed_data1 >> (32 - shift_operand)&0xFFFFFFFF))
    mult = (fixed_data1 * fixed_data2)&0xFFFFFFFF
    if fixed_data2 == 0:
        div = 0xFFFFFFFF
    elif to_signed32(fixed_data1) == -2147483648 and to_signed32(fixed_data2) == -1:
        div = 0x80000000
    else:
        div = int(to_signed32(fixed_data1) / to_signed32(fixed_data2)) & 0xFFFFFFFF
    
    expected_result_json = {
        "result": [
            xor, add, sub, shift, mult, div, fixed_data1, fixed_data1, fixed_data2, fixed_data2, fixed_data2, fixed_data2
        ]
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch failed")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    return True

def char_combi_operations_batch_fvsr_test(opentitantool_path, iterations, num_segments):
    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    trigger = 0
    fixed_data1 = 11
    fixed_data2 = 11
    actual_result = char_combi_operations_batch_fvsr(opentitantool_path, iterations, num_segments, trigger, fixed_data1, fixed_data2)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch_fvsr gave an unexpected result")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False

    # Calculate the expected result
    for _ in range(iterations):
        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data1 = fixed_data1
            else:
                batch_data1 = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1

        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data2 = fixed_data2
            else:
                batch_data2 = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1

    effective_shift = batch_data2 % 32
    expected_result_json = json.loads('{"result":[0,0,0,0,0,0,0,0,0,0,0,0]}')    
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch_fvsr failed")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    trigger = 32
    fixed_data1 = 11
    fixed_data2 = 11
    actual_result = char_combi_operations_batch_fvsr(opentitantool_path, iterations, num_segments, trigger, fixed_data1, fixed_data2)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch_fvsr gave an unexpected result")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False

    # Calculate the expected result
    for _ in range(iterations):
        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data1 = fixed_data1
            else:
                batch_data1 = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1

        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data2 = fixed_data2
            else:
                batch_data2 = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1

    batch_data1 = to_signed32(batch_data1)
    batch_data2 = to_signed32(batch_data2)
    if batch_data2 == 0:
        div = 0xFFFFFFFF
    elif batch_data1 == -2147483648 and batch_data2 == -1:
        div = 0x80000000
    else:
        div = int(batch_data1 / batch_data2) & 0xFFFFFFFF

    expected_result_json = {
        "result": [
            0, 0, 0, 0, 0, div, 0, 0, 0, 0, 0, 0
        ]
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch_fvsr failed")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    trigger = 256
    fixed_data1 = 11
    fixed_data2 = 11
    actual_result = char_combi_operations_batch_fvsr(opentitantool_path, iterations, num_segments, trigger, fixed_data1, fixed_data2)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch_fvsr gave an unexpected result")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False

    # Calculate the expected result
    for _ in range(iterations):
        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data1 = fixed_data1
            else:
                batch_data1 = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1

        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data2 = fixed_data2
            else:
                batch_data2 = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1

    expected_result_json = {
        "result": [
            0, 0, 0, 0, 0, 0, 0, 0, batch_data2, 0, 0, 0
        ]
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch_fvsr failed")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    trigger = 4095
    fixed_data1 = 11
    fixed_data2 = 11
    actual_result = char_combi_operations_batch_fvsr(opentitantool_path, iterations, num_segments, trigger, fixed_data1, fixed_data2)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch_fvsr gave an unexpected result")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False

    # Calculate the expected result
    for _ in range(iterations):
        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data1 = fixed_data1
            else:
                batch_data1 = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1

        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data2 = fixed_data2
            else:
                batch_data2 = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1

    xor = batch_data1^batch_data2
    add = (batch_data1+batch_data2)&0xFFFFFFFF
    sub = (batch_data1-batch_data2)&0xFFFFFFFF
    shift_operand = (batch_data2&0xFFFFFFFF) % 32
    shift = ( (batch_data1 << shift_operand)&0xFFFFFFFF | (batch_data1 >> (32 - shift_operand)&0xFFFFFFFF))
    mult = (batch_data1 * batch_data2)&0xFFFFFFFF
    if batch_data2 == 0:
        div = 0xFFFFFFFF
    elif to_signed32(batch_data1) == -2147483648 and to_signed32(batch_data2) == -1:
        div = 0x80000000
    else:
        div = int(to_signed32(batch_data1) / to_signed32(batch_data2)) & 0xFFFFFFFF

    expected_result_json = {
        "result": [
            xor, add, sub, shift, mult, div, batch_data1, batch_data1, batch_data2, batch_data2, batch_data2, batch_data2
        ]
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch_fvsr failed")
        print(f"Input trigger {trigger}, fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    return True

def char_register_file_read_test(opentitantool_path, iterations):
    data = [7,6,5,4,3,2,1,0]
    actual_result = char_register_file_read(opentitantool_path, iterations, data)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_register_file_read gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":0}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_register_file_read failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_register_file_read_batch_fvsr_test(opentitantool_path, iterations, num_segments):
    fixed_data = 2048
    actual_result = char_register_file_read_batch_fvsr(opentitantool_path, iterations, fixed_data, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_register_file_read_batch_fvsr gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data = fixed_data
            else:
                batch_data = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_register_file_read_batch_fvsr failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_register_file_read_batch_random_test(opentitantool_path, iterations, num_segments):
    actual_result = char_register_file_read_batch_random(opentitantool_path, iterations, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_register_file_read_batch_random gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        for __ in range(num_segments):
            batch_data = random.getrandbits(32)
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_register_file_read_batch_random failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_register_file_write_test(opentitantool_path, iterations):
    data = [7,6,5,4,3,2,1,0]
    actual_result = char_register_file_write(opentitantool_path, iterations, data)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_register_file_write gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":0}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_register_file_write failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_register_file_write_batch_fvsr_test(opentitantool_path, iterations, num_segments):
    fixed_data = 2048
    actual_result = char_register_file_write_batch_fvsr(opentitantool_path, iterations, fixed_data, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_register_file_write_batch_fvsr gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data = fixed_data
            else:
                batch_data = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_register_file_write_batch_fvsr failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_register_file_write_batch_random_test(opentitantool_path, iterations, num_segments):
    actual_result = char_register_file_write_batch_random(opentitantool_path, iterations, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_register_file_write_batch_random gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        # We generate random data for 6 registers per call
        for __ in range(num_segments*6):
            batch_data = random.getrandbits(32)
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_register_file_write_batch_random failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_tl_read_test(opentitantool_path, iterations):
    data = [255,255,255,0,0,0,0,0]
    actual_result = char_tl_read(opentitantool_path, iterations, data)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_tl_read gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":0}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_tl_read failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_tl_read_batch_fvsr_test(opentitantool_path, iterations, num_segments):
    fixed_data = 2048
    actual_result = char_tl_read_batch_fvsr(opentitantool_path, iterations, fixed_data, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_tl_read_batch_fvsr gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data = fixed_data
            else:
                batch_data = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_tl_read_batch_fvsr failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_tl_read_batch_fvsr_fix_address_test(opentitantool_path, iterations, num_segments):
    fixed_data = 2048
    actual_result = char_tl_read_batch_fvsr_fix_address(opentitantool_path, iterations, fixed_data, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_tl_read_batch_fvsr_fix_address gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data = fixed_data
            else:
                batch_data = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_tl_read_batch_fvsr_fix_address failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_tl_read_batch_random_test(opentitantool_path, iterations, num_segments):
    actual_result = char_tl_read_batch_random(opentitantool_path, iterations, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_tl_read_batch_random gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        for __ in range(num_segments):
            batch_data = random.getrandbits(32)
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_tl_read_batch_random failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_tl_read_batch_random_fix_address_test(opentitantool_path, iterations, num_segments):
    actual_result = char_tl_read_batch_random_fix_address(opentitantool_path, iterations, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_tl_read_batch_random_fix_address gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        for __ in range(num_segments):
            batch_data = random.getrandbits(32)
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_tl_read_batch_random_fix_address failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_tl_write_test(opentitantool_path, iterations):
    data = [255,255,255,0,0,0,0,0]
    actual_result = char_tl_write(opentitantool_path, iterations, data)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_tl_write gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":0}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_tl_write failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_tl_write_batch_fvsr_test(opentitantool_path, iterations, num_segments):
    fixed_data = 2048
    actual_result = char_tl_write_batch_fvsr(opentitantool_path, iterations, fixed_data, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_tl_write_batch_fvsr gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data = fixed_data
            else:
                batch_data = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_tl_write_batch_fvsr failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_tl_write_batch_fvsr_fix_address_test(opentitantool_path, iterations, num_segments):
    fixed_data = 2048
    actual_result = char_tl_write_batch_fvsr_fix_address(opentitantool_path, iterations, fixed_data, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_tl_write_batch_fvsr_fix_address gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        sample_fixed = True
        for __ in range(num_segments):
            if sample_fixed:
                batch_data = fixed_data
            else:
                batch_data = random.getrandbits(32)
            sample_fixed = random.getrandbits(32) & 0x1
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_tl_write_batch_fvsr_fix_address failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_tl_write_batch_random_test(opentitantool_path, iterations, num_segments):
    actual_result = char_tl_write_batch_random(opentitantool_path, iterations, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_tl_write_batch_random gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        for __ in range(num_segments):
            batch_data = random.getrandbits(32)
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_tl_write_batch_random failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_tl_write_batch_random_fix_address_test(opentitantool_path, iterations, num_segments):
    actual_result = char_tl_write_batch_random_fix_address(opentitantool_path, iterations, num_segments)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_tl_write_batch_random_fix_address gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    for _ in range(iterations):
        for __ in range(num_segments):
            batch_data = random.getrandbits(32)
    expected_result_json = {
        "result": batch_data
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_tl_write_batch_random_fix_address failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True


def main():
    r = Runfiles.Create()
    opentitantool_path = r.Rlocation("lowrisc_opentitan/sw/host/opentitantool/opentitantool")

    firmware_target_name = os.environ.get("SELECTED_FIRMWARE_TARGET", "pen_test_sca_silicon_owner_gb_rom_ext")
    firmware_path = r.Rlocation(f"lowrisc_opentitan/sw/device/tests/penetrationtests/firmware/{firmware_target_name}.img")

    target = DUT()

    flash_target(opentitantool_path, firmware_path)
    target.dump_all()

    if not reset_test(opentitantool_path, target):
        print("Reset test failure")
        return False

    if not init_test(opentitantool_path, target):
        print("Init test failure")
        return False

    iterations = 10
    num_segments_list = [1, 2, 5, 10, 12]

    for num_segments in num_segments_list:
        char_combi_operations_batch_test(opentitantool_path, iterations, num_segments)
        char_combi_operations_batch_fvsr_test(opentitantool_path, iterations, num_segments)
    char_register_file_read_test(opentitantool_path, iterations)
    for num_segments in num_segments_list:
        char_register_file_read_batch_fvsr_test(opentitantool_path, iterations, num_segments)
        char_register_file_read_batch_random_test(opentitantool_path, iterations, num_segments)
    char_register_file_write_test(opentitantool_path, iterations)
    for num_segments in num_segments_list:
        char_register_file_write_batch_fvsr_test(opentitantool_path, iterations, num_segments)
        char_register_file_write_batch_random_test(opentitantool_path, iterations, num_segments)
    char_tl_read_test(opentitantool_path, iterations)
    for num_segments in num_segments_list:
        char_tl_read_batch_fvsr_test(opentitantool_path, iterations, num_segments)
        char_tl_read_batch_fvsr_fix_address_test(opentitantool_path, iterations, num_segments)
        char_tl_read_batch_random_test(opentitantool_path, iterations, num_segments)
        char_tl_read_batch_random_fix_address_test(opentitantool_path, iterations, num_segments)
    char_tl_write_test(opentitantool_path, iterations)
    for num_segments in num_segments_list:
        char_tl_write_batch_fvsr_test(opentitantool_path, iterations, num_segments)
        char_tl_write_batch_fvsr_fix_address_test(opentitantool_path, iterations, num_segments)
        char_tl_write_batch_random_test(opentitantool_path, iterations, num_segments)
        char_tl_write_batch_random_fix_address_test(opentitantool_path, iterations, num_segments)

    print("Testing finished")
    return True

if __name__ == "__main__":
  main()