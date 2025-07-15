from test.penetrationtests.sca.host_scripts.sca_kmac_functions import *
from target.communication.sca_kmac_commands import OTKMAC
from python.runfiles import Runfiles
from target.chip import *
from test.penetrationtests.util.utils import *
import os
import json
import random
from Crypto.Hash import KMAC128

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
    kmacsca = OTKMAC(target, "ujson")
    device_id, owner_page, boot_log, boot_measurements, version = kmacsca.init(0)
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

def char_kmac_single_test(opentitantool_path, iterations):
    key = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    text = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    masking = True
    actual_result = char_kmac_single(opentitantool_path, iterations, masking, key, text)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_kmac_single gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    mac = KMAC128.new(key=bytes(key), mac_len=32)
    mac.update(bytes(text))
    tag = bytearray(mac.digest())
    expected_result = [x for x in tag]

    expected_result_json = {
        "batch_digest": expected_result,
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_kmac_single failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False    
    return True

def char_kmac_batch_daisy_chain_test(opentitantool_path, iterations, num_segments):
    key = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    text = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    masking = True
    actual_result = char_kmac_batch_daisy_chain(opentitantool_path, iterations, num_segments, masking, key, text)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_kmac_batch_daisy_chain gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    for i in range(num_segments):
        mac = KMAC128.new(key=bytes(key), mac_len=32)
        mac.update(bytes(text))
        digest = [x for x in bytearray(mac.digest())]
        text = digest[:16]

    expected_result_json = {
        "digest": text,
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_kmac_batch_daisy_chain failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False    
    return True

def char_kmac_batch_test(opentitantool_path, iterations, num_segments):
    key = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    masking = True
    actual_result = char_kmac_batch(opentitantool_path, iterations, num_segments, masking, key)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_kmac_batch gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Seed the synchronized randomness with the same seed as in the chip which is 1
    random.seed(1)

    # Generate the batch data
    sample_fixed = 0
    for _ in range(iterations):
        xor_tag = [0 for _ in range(32)]
        for __ in range(num_segments):
            if sample_fixed == 1:
                batch_key = key
            else:
                batch_key = [random.randint(0, 255) for _ in range(16)]
            batch_data = [random.randint(0, 255) for _ in range(16)]
            sample_fixed = batch_data[0] & 0x1

            mac = KMAC128.new(key=bytes(batch_key), mac_len=32)
            mac.update(bytes(batch_data))
            tag = [x for x in bytearray(mac.digest())]
            xor_tag = [xor_tag[i] ^ tag[i] for i in range(32)]
            digest = tag[:16]

    expected_result_json = {
        "batch_digest": xor_tag,
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_kmac_batch failed")
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
    num_segments_list = [1, 2, 5, 10, 12, 50]

    char_kmac_single_test(opentitantool_path, iterations)
    for num_segments in num_segments_list:
        char_kmac_batch_daisy_chain_test(opentitantool_path, iterations, num_segments)
        char_kmac_batch_test(opentitantool_path, iterations, num_segments)

    print("Testing finished")
    return True

if __name__ == "__main__":
  main()