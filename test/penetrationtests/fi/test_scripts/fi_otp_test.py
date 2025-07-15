from test.penetrationtests.fi.host_scripts.fi_otp_functions import *
from target.communication.fi_otp_commands import OTFIOtp
from python.runfiles import Runfiles
from target.chip import *
from test.penetrationtests.util.utils import *
import os
import json

ignored_keys_set = set(["partition_ref", "partition_fi"])

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
    otpfi = OTFIOtp(target)
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otpfi.init()
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
    
def char_vendor_test_test(opentitantool_path, iterations):
    actual_result = char_vendor_test(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_vendor_test gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],"otp_error_causes":[0,0,0,0,0,0,0,0,0,0],"otp_status_codes":0,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_vendor_test failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_owner_sw_cfg_test(opentitantool_path, iterations):
    actual_result = char_owner_sw_cfg(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_owner_sw_cfg gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],"otp_error_causes":[0,0,0,0,0,0,0,0,0,0],"otp_status_codes":0,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_owner_sw_cfg failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_hw_cfg_test(opentitantool_path, iterations):
    actual_result = char_hw_cfg(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_hw_cfg gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],"otp_error_causes":[0,0,0,0,0,0,0,0,0,0],"otp_status_codes":0,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_hw_cfg failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_life_cycle_test(opentitantool_path, iterations):
    actual_result = char_life_cycle(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_life_cycle gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],"otp_error_causes":[0,0,0,0,0,0,0,0,0,0],"otp_status_codes":0,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_life_cycle failed")
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

    iterations = 10
    char_vendor_test_test(opentitantool_path, iterations)
    char_owner_sw_cfg_test(opentitantool_path, iterations)
    char_hw_cfg_test(opentitantool_path, iterations)
    char_life_cycle_test(opentitantool_path, iterations)

    print("Testing finished")
    return True

if __name__ == "__main__":
  main()
