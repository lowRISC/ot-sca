from test.penetrationtests.fi.host_scripts.fi_crypto_functions import *
from target.communication.fi_crypto_commands import OTFICrypto
from python.runfiles import Runfiles
from target.chip import *
from test.penetrationtests.util.utils import *
import os
import json
from Crypto.Hash import SHA256

ignored_keys_set = set(["share0", "share1"])

def reset_test(opentitantool, target):
    reset_target(opentitantool)
    while True:
        read_line = str(target.readline().decode().strip())
        if len(read_line) > 0:
            if "firmware_fi.c" in read_line:
                return True
        else:
            return False


def init_test(opentitantool, target):
    cryptofi = OTFICrypto(target)
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = cryptofi.init()
    device_id_json = json.loads(device_id)
    sensors_json = json.loads(sensors)
    alerts_json = json.loads(alerts)
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

    expected_sensors_keys = {
        "sensor_ctrl_en",
        "sensor_ctrl_fatal"
    }
    actual_sensors_keys = set(sensors_json.keys())

    if not expected_sensors_keys == actual_sensors_keys:
        print("sensor keys do not match the expected set.")
        print(f"Expected: {expected_sensors_keys}")
        print(f"Actual: {actual_sensors_keys}")
        return False

    expected_alerts_keys = {
        "alert_classes",
        "enabled_alerts",
        "enabled_classes",
        "accumulation_thresholds",
        "duration_cycles",
        "escalation_signals_en",
        "escalation_signals_map"
    }
    actual_alerts_keys = set(alerts_json.keys())
    if not expected_alerts_keys == actual_alerts_keys:
        print("alert keys do not match the expected set.")
        print(f"Expected: {expected_alerts_keys}")
        print(f"Actual: {actual_alerts_keys}")
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

def char_aes_test(opentitantool_path, iterations):
    trigger = 0
    actual_result = char_aes(opentitantool_path, iterations, trigger)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_aes gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"ciphertext":[141,145,88,155,234,129,16,92,221,12,69,21,69,208,99,12],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_aes failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_kmac_test(opentitantool_path, iterations):
    trigger = 0
    actual_result = char_kmac(opentitantool_path, iterations, trigger)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_kmac gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"digest":[184,34,91,108,231,47,251,27],"digest_2nd":[142,188,186,201,216,47,203,192],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_kmac failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_kmac_state_test(opentitantool_path, iterations):
    actual_result = char_kmac_state(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_kmac_state gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"digest":[184,34,91,108,231,47,251,27],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_kmac_state failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_sha256_test(opentitantool_path, iterations):
    trigger = 0
    msg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    actual_result = char_sha256(opentitantool_path, iterations, msg, trigger)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_sha256 gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    sha256 = SHA256.new()
    sha256.update(bytes(msg))
    expected_result = bytes_to_32bit_words(bytearray(sha256.digest()))
    expected_result.reverse()

    expected_result_json = {
        "tag": expected_result,
        "err_status": 0,
        "alerts":[0,0,0],
        "ast_alerts":[0,0]
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_sha256 failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    msg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    actual_result = char_sha256(opentitantool_path, iterations, msg, trigger)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_sha256 gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    sha256 = SHA256.new()
    sha256.update(bytes(msg))
    expected_result = bytes_to_32bit_words(bytearray(sha256.digest()))
    expected_result.reverse()

    expected_result_json = {
        "tag": expected_result,
        "err_status": 0,
        "alerts":[0,0,0],
        "ast_alerts":[0,0]
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_sha256 failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False

    msg = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    actual_result = char_sha256(opentitantool_path, iterations, msg, trigger)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_sha256 gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    sha256 = SHA256.new()
    sha256.update(bytes(msg))
    expected_result = bytes_to_32bit_words(bytearray(sha256.digest()))
    expected_result.reverse()

    expected_result_json = {
        "tag": expected_result,
        "err_status": 0,
        "alerts":[0,0,0],
        "ast_alerts":[0,0]
    }
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_sha256 failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False                
    return True

def char_shadow_reg_access_test(opentitantool_path, iterations):
    actual_result = char_shadow_reg_access(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_shadow_reg_access gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":[68162304,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_shadow_reg_access failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_shadow_reg_read_test(opentitantool_path, iterations):
    actual_result = char_shadow_reg_read(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_shadow_reg_read gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":[0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_shadow_reg_read failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def main():
    r = Runfiles.Create()
    opentitantool_path = r.Rlocation("lowrisc_opentitan/sw/host/opentitantool/opentitantool")

    firmware_target_name = os.environ.get("SELECTED_FIRMWARE_TARGET", "pen_test_fi_silicon_owner_gb_rom_ext")
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

    iterations = 20

    char_aes_test(opentitantool_path, iterations)
    char_kmac_test(opentitantool_path, iterations)
    char_kmac_state_test(opentitantool_path, iterations)
    char_sha256_test(opentitantool_path, iterations)
    char_shadow_reg_access_test(opentitantool_path, iterations)
    char_shadow_reg_read_test(opentitantool_path, iterations)

    print("Testing finished")
    return True

if __name__ == "__main__":
  main()