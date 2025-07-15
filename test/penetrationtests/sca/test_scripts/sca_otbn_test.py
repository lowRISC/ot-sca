from test.penetrationtests.sca.host_scripts.sca_otbn_functions import *
from target.communication.sca_otbn_commands import OTOTBN
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
    otbnsca = OTOTBN(target, "ujson")
    device_id, owner_page, boot_log, boot_measurements, version = otbnsca.init()
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
    fixed_data1 = 0
    fixed_data2 = 0
    print_flag = True
    actual_result = char_combi_operations_batch(opentitantool_path, iterations, num_segments, fixed_data1, fixed_data2, print_flag, trigger)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch gave an unexpected result")
        print(f"Input fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False

    # Calculate the expected result
    fixed_data_array1 = [fixed_data1 for _ in range(8)]
    fixed_data_array2 = [fixed_data2 for _ in range(8)]
    add = int_to_array((array_to_int(fixed_data_array1)+array_to_int(fixed_data_array2))%(1<<256))
    sub = int_to_array((array_to_int(fixed_data_array1)-array_to_int(fixed_data_array2))%(1<<256))
    xor = int_to_array((array_to_int(fixed_data_array1)^array_to_int(fixed_data_array2))%(1<<256))
    shift = int_to_array((array_to_int(fixed_data_array1)<<1)%(1<<256))
    fixed_data_array1 = [fixed_data1, fixed_data1, 0, 0, 0, 0, 0, 0]
    fixed_data_array2 = [fixed_data2, fixed_data2, 0, 0, 0, 0, 0, 0]
    mult = int_to_array((array_to_int(fixed_data_array1)*array_to_int(fixed_data_array2))%(1<<256))
    FG = 0
    if fixed_data1 == fixed_data2:
        FG += 8
    if fixed_data1 < fixed_data2:
        FG += 1
    if sub[0] & 0x1:
        FG += 4
    if (array_to_int(sub) >> 255) & 0x1:
        FG += 2

    expected_result_json = {
        "result1": [fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1],
        "result2": [fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1],
        "result3": add,
        "result4": sub,
        "result5": xor,
        "result6": shift,
        "result7": mult,
        "result8": FG,
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch_fvsr failed")
        print(f"Input fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    fixed_data1 = 1
    fixed_data2 = 1
    print_flag = True
    actual_result = char_combi_operations_batch(opentitantool_path, iterations, num_segments, fixed_data1, fixed_data2, print_flag, trigger)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch gave an unexpected result")
        print(f"Input fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False

    # Calculate the expected result
    fixed_data_array1 = [fixed_data1 for _ in range(8)]
    fixed_data_array2 = [fixed_data2 for _ in range(8)]
    add = int_to_array((array_to_int(fixed_data_array1)+array_to_int(fixed_data_array2))%(1<<256))
    sub = int_to_array((array_to_int(fixed_data_array1)-array_to_int(fixed_data_array2))%(1<<256))
    xor = int_to_array((array_to_int(fixed_data_array1)^array_to_int(fixed_data_array2))%(1<<256))
    shift = int_to_array((array_to_int(fixed_data_array1)<<1)%(1<<256))
    fixed_data_array1 = [fixed_data1, fixed_data1, 0, 0, 0, 0, 0, 0]
    fixed_data_array2 = [fixed_data2, fixed_data2, 0, 0, 0, 0, 0, 0]
    mult = int_to_array((array_to_int(fixed_data_array1)*array_to_int(fixed_data_array2))%(1<<256))
    FG = 0
    if fixed_data1 == fixed_data2:
        FG += 8
    if fixed_data1 < fixed_data2:
        FG += 1
    if sub[0] & 0x1:
        FG += 4
    if (array_to_int(sub) >> 255) & 0x1:
        FG += 2

    expected_result_json = {
        "result1": [fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1],
        "result2": [fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1],
        "result3": add,
        "result4": sub,
        "result5": xor,
        "result6": shift,
        "result7": mult,
        "result8": FG,
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch_fvsr failed")
        print(f"Input fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    fixed_data1 = random.getrandbits(32)
    fixed_data2 = random.getrandbits(32)
    print_flag = True
    actual_result = char_combi_operations_batch(opentitantool_path, iterations, num_segments, fixed_data1, fixed_data2, print_flag, trigger)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi_operations_batch gave an unexpected result")
        print(f"Input fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
        print(f"Actual: {actual_result}")
        return False

    # Calculate the expected result
    fixed_data_array1 = [fixed_data1 for _ in range(8)]
    fixed_data_array2 = [fixed_data2 for _ in range(8)]
    add = int_to_array((array_to_int(fixed_data_array1)+array_to_int(fixed_data_array2))%(1<<256))
    sub = int_to_array((array_to_int(fixed_data_array1)-array_to_int(fixed_data_array2))%(1<<256))
    xor = int_to_array((array_to_int(fixed_data_array1)^array_to_int(fixed_data_array2))%(1<<256))
    shift = int_to_array((array_to_int(fixed_data_array1)<<1)%(1<<256))
    fixed_data_array1 = [fixed_data1, fixed_data1, 0, 0, 0, 0, 0, 0]
    fixed_data_array2 = [fixed_data2, fixed_data2, 0, 0, 0, 0, 0, 0]
    mult = int_to_array((array_to_int(fixed_data_array1)*array_to_int(fixed_data_array2))%(1<<256))
    FG = 0
    if fixed_data1 == fixed_data2:
        FG += 8
    if fixed_data1 < fixed_data2:
        FG += 1
    if sub[0] & 0x1:
        FG += 4
    if (array_to_int(sub) >> 255) & 0x1:
        FG += 2

    expected_result_json = {
        "result1": [fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1],
        "result2": [fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1, fixed_data1],
        "result3": add,
        "result4": sub,
        "result5": xor,
        "result6": shift,
        "result7": mult,
        "result8": FG,
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi_operations_batch_fvsr failed")
        print(f"Input fixed_data1 {fixed_data1} and fixed_data2 {fixed_data2}")
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

    print("Testing finished")
    return True

if __name__ == "__main__":
  main()