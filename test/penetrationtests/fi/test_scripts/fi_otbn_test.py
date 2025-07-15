from test.penetrationtests.fi.host_scripts.fi_otbn_functions import *
from target.communication.fi_otbn_commands import OTFIOtbn
from python.runfiles import Runfiles
from target.chip import *
from test.penetrationtests.util.utils import *
import os
import json

ignored_keys_set = set(["data"])

def reset_test(opentitantool, target):
    reset_target(opentitantool)
    while True:
        read_line = str(target.readline().decode().strip())
        if len(read_line) > 0:
            if "firmware_fi_otbn.c" in read_line:
                return True
        else:
            return False


def init_test(opentitantool, target):
    otbnfi = OTFIOtbn(target)
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otbnfi.init()
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

def char_beq_test(opentitantool_path, iterations):
    actual_result = char_beq(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_beq_test gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":0,"insn_cnt":509,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_beq failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_bn_rshi_test(opentitantool_path, iterations):
    data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    actual_result = char_bn_rshi(opentitantool_path, iterations, data)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_bn_rshi gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"big_num":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"insn_cnt":109,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_bn_rshi failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_bn_sel_test(opentitantool_path, iterations):
    data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    actual_result = char_bn_sel(opentitantool_path, iterations, data)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_bn_sel gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"big_num":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"insn_cnt":1014,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_bn_sel failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_bn_wsrr_test(opentitantool_path, iterations):
    actual_result = char_bn_wsrr(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_bn_wsrr gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"res":0,"insn_cnt":1089,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_bn_wsrr failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_bne_test(opentitantool_path, iterations):
    actual_result = char_bne(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_bne gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":0,"insn_cnt":509,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_bne failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_dmem_access_test(opentitantool_path, iterations):
    actual_result = char_dmem_access(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_dmem_access gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"res":0,"insn_cnt":271,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_dmem_access failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_dmem_write_test(opentitantool_path, iterations):
    actual_result = char_dmem_write(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_dmem_write gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"insn_cnt":1,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_dmem_write failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_dmem_op_loop_test(opentitantool_path, iterations):
    actual_result = char_dmem_op_loop(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_dmem_op_loop gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"loop_counter":10000,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_dmem_op_loop failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_reg_op_loop_test(opentitantool_path, iterations):
    actual_result = char_reg_op_loop(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_reg_op_loop gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"loop_counter":10000,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_reg_op_loop failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_jal_test(opentitantool_path, iterations):
    actual_result = char_jal(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_jal gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":0,"insn_cnt":505,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_jal failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_lw_test(opentitantool_path, iterations):
    actual_result = char_lw(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_lw gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"insn_cnt":1084,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_lw failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_mem_test(opentitantool_path, iterations):
    byte_offset = 0
    num_words = 4
    imem = False
    dmem = True
    actual_result = char_mem(opentitantool_path, iterations, byte_offset, num_words, imem, dmem)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_mem gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"res":0,"imem_data":[0,0,0,0,0,0,0,0],"imem_addr":[0,0,0,0,0,0,0,0],"dmem_data":[0,0,0,0,0,0,0,0],"dmem_addr":[0,0,0,0,0,0,0,0],"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_mem failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_rf_test(opentitantool_path, iterations):
    actual_result = char_rf(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_rf gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"res":0,"faulty_gpr":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"faulty_wdr":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_rf failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_unrolled_dmem_op_loop_test(opentitantool_path, iterations):
    actual_result = char_unrolled_dmem_op_loop(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_unrolled_dmem_op_loop gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"loop_counter":100,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_unrolled_dmem_op_loop failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_unrolled_reg_op_loop_test(opentitantool_path, iterations):
    actual_result = char_unrolled_reg_op_loop(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_unrolled_reg_op_loop gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"loop_counter":100,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_unrolled_reg_op_loop failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_load_integrity_test(opentitantool_path, iterations):
    actual_result = char_load_integrity(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_load_integrity gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":0,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_load_integrity failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_pc_test(opentitantool_path, iterations):
    pc = 2224
    actual_result = char_pc(opentitantool_path, iterations, pc)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_pc gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"pc_dmem":2224,"pc_otbn":2224,"insn_cnt":472,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_pc failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def main():
    r = Runfiles.Create()
    opentitantool_path = r.Rlocation("lowrisc_opentitan/sw/host/opentitantool/opentitantool")

    firmware_target_name = os.environ.get("SELECTED_FIRMWARE_TARGET", "pen_test_fi_otbn_silicon_owner_gb_rom_ext")
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

    iterations = 5

    char_beq_test(opentitantool_path, iterations)
    char_bn_rshi_test(opentitantool_path, iterations)
    char_bn_sel_test(opentitantool_path, iterations)
    char_bn_wsrr_test(opentitantool_path, iterations)
    char_bne_test(opentitantool_path, iterations)
    char_dmem_access_test(opentitantool_path, iterations)
    char_dmem_write_test(opentitantool_path, iterations)
    char_dmem_op_loop_test(opentitantool_path, iterations)
    char_reg_op_loop_test(opentitantool_path, iterations)
    char_jal_test(opentitantool_path, iterations)
    char_lw_test(opentitantool_path, iterations)
    char_mem_test(opentitantool_path, iterations)
    char_rf_test(opentitantool_path, iterations)
    char_unrolled_dmem_op_loop_test(opentitantool_path, iterations)
    char_unrolled_reg_op_loop_test(opentitantool_path, iterations)
    char_load_integrity_test(opentitantool_path, iterations)
    char_pc_test(opentitantool_path, iterations)

    print("Testing finished")
    return True

if __name__ == "__main__":
  main()