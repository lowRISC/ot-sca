from test.penetrationtests.sca.host_scripts.sca_aes_functions import *
from target.communication.sca_aes_commands import OTAES
from python.runfiles import Runfiles
from target.chip import *
import util.data_generator as dg
from test.penetrationtests.util.utils import *
import os
import json
import random
from Crypto.Cipher import AES

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
    aessca = OTAES(target)
    device_id, owner_page, boot_log, boot_measurements, version = aessca.init(0)
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

def char_aes_single_encrypt_test(opentitantool_path, iterations):
    masking = False
    key = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    text = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    actual_result = char_aes_single_encrypt(opentitantool_path, iterations, masking, key, text)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_aes_single_encrypt gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    cipher_gen = AES.new(bytes(key), AES.MODE_ECB)
    expected_result = [x for x in cipher_gen.encrypt(bytes(text))]

    expected_result_json = {
        "ciphertext": expected_result,
        "ciphertext_length": 16
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_aes_single_encrypt failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    
    return True

def char_aes_batch_alternative_encrypt_test(opentitantool_path, iterations, num_segments):
    masking = True
    key = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    text = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    actual_result = char_aes_batch_alternative_encrypt(opentitantool_path, iterations, num_segments, masking, key, text)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_aes_batch_alternative_encrypt gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    cipher_gen = AES.new(bytes(key), AES.MODE_ECB)
    for _ in range(iterations):
        for __ in range(num_segments):
            text = [x for x in cipher_gen.encrypt(bytes(text))]

    expected_result_json = {
        "ciphertext": text,
        "ciphertext_length": len(text)
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_aes_batch_alternative_encrypt failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_aes_batch_data_fvsr_encrypt_test(opentitantool_path, iterations, num_segments):
    masking = True
    actual_result = char_aes_batch_data_fvsr_encrypt(opentitantool_path, iterations, num_segments, masking)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_aes_batch_data_fvsr_encrypt gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Set the syncrhonized randomness
    batch_prng_seed = 1
    random.seed(batch_prng_seed)

    # Start with a fixed measurement
    sample_fixed = 1
    prng_state = 0x99999999
    dg.set_start('FVSR_DATA')
    for _ in range(iterations):
        for __ in range(num_segments):
            if sample_fixed:
                plaintext, ciphertext, key = dg.get_fixed('FVSR_DATA')
            else:
                plaintext, ciphertext, key = dg.get_random('FVSR_DATA')
            sample_fixed = prng_state & 0x1
            prng_state = prng_state >> 1
            if sample_fixed:
                prng_state ^= 0x80000057

    expected_result_json = {
        "ciphertext": ciphertext,
        "ciphertext_length": len(ciphertext)
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_aes_batch_data_fvsr_encrypt failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_aes_batch_key_fvsr_encrypt_test(opentitantool_path, iterations, num_segments):
    masking = True
    actual_result = char_aes_batch_key_fvsr_encrypt(opentitantool_path, iterations, num_segments, masking)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_aes_batch_key_fvsr_encrypt gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    # Set the syncrhonized randomness
    batch_prng_seed = 1
    random.seed(batch_prng_seed)

    # Start with a fixed measurement
    sample_fixed = 1
    prng_state = 0x99999999
    dg.set_start('FVSR_KEY')
    for _ in range(iterations):
        for __ in range(num_segments):
            if sample_fixed:
                plaintext, ciphertext, key = dg.get_fixed('FVSR_KEY')
            else:
                plaintext, ciphertext, key = dg.get_random('FVSR_KEY')
            sample_fixed = plaintext[0] & 0x1

    expected_result_json = {
        "ciphertext": ciphertext,
        "ciphertext_length": len(ciphertext)
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_aes_batch_key_fvsr_encrypt failed")
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
    char_aes_single_encrypt_test(opentitantool_path, iterations)
    for num_segments in num_segments_list:
        char_aes_batch_alternative_encrypt_test(opentitantool_path, iterations, num_segments)
        char_aes_batch_data_fvsr_encrypt_test(opentitantool_path, iterations, num_segments)
        char_aes_batch_key_fvsr_encrypt_test(opentitantool_path, iterations, num_segments)

    print("Testing finished")
    return True

if __name__ == "__main__":
  main()