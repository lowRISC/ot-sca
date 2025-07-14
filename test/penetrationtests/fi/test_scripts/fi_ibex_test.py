from test.penetrationtests.fi.host_scripts.fi_ibex_functions import *
from communication.fi_ibex_commands import OTFIIbex
from python.runfiles import Runfiles
from communication.chip import *
from test.penetrationtests.util.utils import *
import os
import json

ignored_keys_set = set(["registers", "registers_test_1", "registers_test_2", "registers_test_3"])

def reset_test(opentitantool, target):
    reset_target(opentitantool)
    while True:
        read_line = str(target.readline().decode().strip())
        if len(read_line) > 0:
            if "firmware_fi_ibex.c" in read_line:
                return True
        else:
            return False


def init_test(opentitantool, target):
    ibexfi = OTFIIbex(target)
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = ibexfi.init()
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

def char_addi_single_beq_test(opentitantool_path, iterations):
    actual_result = char_addi_single_beq(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_addi_single_beq_test gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_addi_single_beq failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_addi_single_beq_neg_test(opentitantool_path, iterations):
    actual_result = char_addi_single_beq_neg(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_addi_single_beq_neg gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_addi_single_beq_neg failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_addi_single_bne_test(opentitantool_path, iterations):
    actual_result = char_addi_single_bne(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_addi_single_bne gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_addi_single_bne failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_addi_single_bne_neg_test(opentitantool_path, iterations):
    actual_result = char_addi_single_bne_neg(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_addi_single_bne_neg gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_addi_single_bne_neg failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_combi_test(opentitantool_path, iterations):
    actual_result = char_combi(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_combi gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty_test_1":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data_test_1":[0,0,0,0,0,0,0,0,0,0,0,0,0],"result_test_2":13,"result_test_3":15,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_combi failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_conditional_branch_beq_test(opentitantool_path, iterations):
    actual_result = char_conditional_branch_beq(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_conditional_branch_beq gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result1":175,"result2":239,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_conditional_branch_beq failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_conditional_branch_bge_test(opentitantool_path, iterations):
    actual_result = char_conditional_branch_bge(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_conditional_branch_bge gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result1":175,"result2":239,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_conditional_branch_bge failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_conditional_branch_bgeu_test(opentitantool_path, iterations):
    actual_result = char_conditional_branch_bgeu(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_conditional_branch_bgeu gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result1":175,"result2":239,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_conditional_branch_bgeu failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_conditional_branch_blt_test(opentitantool_path, iterations):
    actual_result = char_conditional_branch_blt(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_conditional_branch_blt gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result1":239,"result2":175,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_conditional_branch_blt failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_conditional_branch_bltu_test(opentitantool_path, iterations):
    actual_result = char_conditional_branch_bltu(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_conditional_branch_bltu gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result1":239,"result2":175,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_conditional_branch_bltu failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_conditional_branch_bne_test(opentitantool_path, iterations):
    actual_result = char_conditional_branch_bne(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_conditional_branch_bne gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result1":175,"result2":175,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_conditional_branch_bne failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_csr_read_test(opentitantool_path, iterations):
    actual_result = char_csr_read(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_csr_read gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_csr_read failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_csr_write_test(opentitantool_path, iterations):
    actual_result = char_csr_write(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_csr_write gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":0,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_csr_write failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_csr_combi_test(opentitantool_path, iterations):
    ref_values = [1,5,33,101,3,5,39321,6,5138,50115,39321,38502,39321,39321,1,2,3]
    actual_result = char_csr_combi(opentitantool_path, trigger = 0, ref_values = ref_values, iterations = iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_csr_combi gave an unexpected result")
        print(f"ref_values: {ref_values}")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"output":[1,5,33,101,3,5,39321,6,5138,50115,39321,38502,39321,39321,1,2,3], "data_faulty": [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_csr_combi failed")
        print(f"ref_values: {ref_values}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    
    ref_values = [2,3,16641,4,5,6,38553,7,5137,256,26214,26214,26214,5,2,3,4]
    actual_result = char_csr_combi(opentitantool_path, trigger = 0, ref_values = ref_values, iterations = iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_csr_combi gave an unexpected result")
        print(f"ref_values: {ref_values}")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"output":[2,3,16641,4,5,6,38553,7,5137,256,26214,26214,26214,5,2,3,4], "data_faulty": [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_csr_combi failed")
        print(f"ref_values: {ref_values}")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_flash_read_test(opentitantool_path, iterations):
    for flash_region in range(10):
        actual_result = char_flash_read(opentitantool_path, flash_region = flash_region, iterations = iterations)
        try:
          actual_result_json = json.loads(actual_result)
          expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
          if compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
              print(f"char_flash_read succes with flash_region = {flash_region}")
              return True
        except:
            continue
    print("char_flash_read failed")
    print(f"Expected: {expected_result_json}")
    print(f"Actual: {actual_result}")
    return False

def char_flash_write_test(opentitantool_path, iterations):
    for flash_region in range(10):
        actual_result = char_flash_write(opentitantool_path, flash_region = flash_region, iterations = iterations)
        try:
          actual_result_json = json.loads(actual_result)
          expected_result_json = json.loads('{"result":0,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
          if compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
              print(f"char_flash_write succes with flash_region = {flash_region}")
              return True
        except:
            continue
    print("char_flash_write failed")
    print(f"Expected: {expected_result_json}")
    print(f"Actual: {actual_result}")
    return False

def char_mem_op_loop_test(opentitantool_path, iterations):
    actual_result = char_mem_op_loop(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_mem_op_loop gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":[10000,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_mem_op_loop failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_register_file_test(opentitantool_path, iterations):
    actual_result = char_register_file(opentitantool_path, iterations)
    actual_result_json = json.loads(actual_result)
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_register_file failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_register_file_read_test(opentitantool_path, iterations):
    actual_result = char_register_file_read(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_register_file_read gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_register_file_read failed")
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
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_reg_op_loop failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_single_beq_test(opentitantool_path, iterations):
    actual_result = char_single_beq(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_single_beq gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_single_beq failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_single_bne_test(opentitantool_path, iterations):
    actual_result = char_single_bne(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_single_bne gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_single_bne failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_sram_read_test(opentitantool_path, iterations):
    actual_result = char_sram_read(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_sram_read gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_sram_read failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_sram_read_ret_test(opentitantool_path, iterations):
    actual_result = char_sram_read_ret(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_sram_read_ret gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"sram_err_status":56, "data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_sram_read_ret failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_sram_static_test(opentitantool_path, iterations):
    actual_result = char_sram_static(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_sram_static gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"addresses":[0,0,0,0,0,0,0,0,0,0,0,0,0],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_sram_static failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_sram_write_test(opentitantool_path, iterations):
    actual_result = char_sram_write(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_sram_write gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"memory":[464367618,2343432205,2779096485,2880154539,2881141438,2880293630,3131746989,3134333474,3148725999,3200171710,3203386062,3221229823,3405697037,3405709037,3435973836,3452816845,219540062,3735883980,3735928559,3735931646,3735929054,3735943697,3735941133,3741239533,3735936685,3490524077,3958107115,4208909997,4261281277,4276215469,4277009102,4277075694,464367618,2343432205,2779096485,2880154539,2881141438,2880293630,3131746989,3134333474,3148725999,3200171710,3203386062,3221229823,3405697037,3405709037,3435973836,3452816845,219540062,3735883980,3735928559,3735931646,3735929054,3735943697,3735941133,3741239533,3735936685,3490524077,3958107115,4208909997,4261281277,4276215469,4277009102,4277075694],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_sram_write failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_sram_write_read_test(opentitantool_path, iterations):
    actual_result = char_sram_write_read(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_sram_write_read gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_sram_write_read failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_sram_write_read_alt_test(opentitantool_path, iterations):
    actual_result = char_sram_write_read_alt(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_sram_write_read_alt gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"memory":[2880293630,3131746989,3134333474,2880293630,3131746989,3134333474,2880293630,3131746989,3134333474,2880293630,3131746989,3134333474,2880293630,3131746989,3134333474,2880293630,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_sram_write_read_alt failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_sram_write_static_unrolled_test(opentitantool_path, iterations):
    actual_result = char_sram_write_static_unrolled(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_sram_write_static_unrolled gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"memory":[464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618,464367618],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_sram_write_static_unrolled failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_unconditional_branch_test(opentitantool_path, iterations):
    actual_result = char_unconditional_branch(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_unconditional_branch gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":30,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_unconditional_branch failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_unconditional_branch_nop_test(opentitantool_path, iterations):
    actual_result = char_unconditional_branch_nop(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_unconditional_branch_nop gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":0,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_unconditional_branch_nop failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_unrolled_mem_op_loop_test(opentitantool_path, iterations):
    actual_result = char_unrolled_mem_op_loop(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_unrolled_mem_op_loop gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":10000,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_unrolled_mem_op_loop failed")
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
    expected_result_json = json.loads('{"result":10000,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_unrolled_reg_op_loop failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_unrolled_reg_op_loop_chain_test(opentitantool_path, iterations):
    actual_result = char_unrolled_reg_op_loop_chain(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_unrolled_reg_op_loop_chain gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result":[60,55,56,57,58,59,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_unrolled_reg_op_loop_chain failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_otp_data_read_test(opentitantool_path, iterations):
    actual_result = char_otp_data_read(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_otp_data_read gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result": 0,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_otp_data_read failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_otp_read_lock_test(opentitantool_path, iterations):
    actual_result = char_otp_read_lock(opentitantool_path, iterations)
    try:
        actual_result_json = json.loads(actual_result)
    except:
        print("char_otp_read_lock gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False
    expected_result_json = json.loads('{"result": 5,"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}')
    if not compare_json_data(actual_result_json, expected_result_json, ignored_keys_set):
        print("char_otp_read_lock failed")
        print(f"Expected: {expected_result_json}")
        print(f"Actual: {actual_result_json}")
        print("")
        return False
    return True

def char_addi_single_beq_cm_test(opentitantool_path):
    actual_result, status = char_addi_single_beq_cm(opentitantool_path)
    if not actual_result or status != False or "FAULT" not in actual_result:
        print("char_addi_single_beq_cm gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    return True

def char_addi_single_beq_cm2_test(opentitantool_path):
    actual_result, status = char_addi_single_beq_cm2(opentitantool_path)
    if not actual_result or status != False or "FAULT" not in actual_result:
        print("char_addi_single_beq_cm gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    return True

def char_hardened_check_eq_complement_branch_test(opentitantool_path):
    actual_result, status = char_hardened_check_eq_complement_branch(opentitantool_path)
    if not actual_result or status != False or "FAULT" not in actual_result:
        print("char_hardened_check_eq_complement_branch gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    return True

def char_hardened_check_eq_unimp_test(opentitantool_path):
    actual_result, status = char_hardened_check_eq_unimp(opentitantool_path)
    if not actual_result or status != False or "FAULT" not in actual_result:
        print("char_hardened_check_eq_unimp gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    return True

def char_hardened_check_eq_2_unimps_test(opentitantool_path):
    actual_result, status = char_hardened_check_eq_2_unimps(opentitantool_path)
    if not actual_result or status != False or "FAULT" not in actual_result:
        print("char_hardened_check_eq_2_unimps gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    return True

def char_hardened_check_eq_3_unimps_test(opentitantool_path):
    actual_result, status = char_hardened_check_eq_3_unimps(opentitantool_path)
    if not actual_result or status != False or "FAULT" not in actual_result:
        print("char_hardened_check_eq_3_unimps gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    return True

def char_hardened_check_eq_4_unimps_test(opentitantool_path):
    actual_result, status = char_hardened_check_eq_4_unimps(opentitantool_path)
    if not actual_result or status != False or "FAULT" not in actual_result:
        print("char_hardened_check_eq_4_unimps gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    return True

def char_hardened_check_eq_5_unimps_test(opentitantool_path):
    actual_result, status = char_hardened_check_eq_5_unimps(opentitantool_path)
    if not actual_result or status != False or "FAULT" not in actual_result:
        print("char_hardened_check_eq_5_unimps gave an unexpected result")
        print(f"Actual: {actual_result}")
        return False

    return True

def main():
    r = Runfiles.Create()
    opentitantool_path = r.Rlocation("lowrisc_opentitan/sw/host/opentitantool/opentitantool")

    firmware_target_name = os.environ.get("SELECTED_FIRMWARE_TARGET", "pen_test_fi_ibex_silicon_owner_gb_rom_ext")
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

    char_addi_single_beq_test(opentitantool_path, iterations)
    char_addi_single_bne_test(opentitantool_path, iterations)
    char_addi_single_bne_neg_test(opentitantool_path, iterations)
    char_combi_test(opentitantool_path, iterations)
    char_conditional_branch_beq_test(opentitantool_path, iterations)
    char_conditional_branch_bge_test(opentitantool_path, iterations)
    char_conditional_branch_bgeu_test(opentitantool_path, iterations)
    char_conditional_branch_blt_test(opentitantool_path, iterations)
    char_conditional_branch_bltu_test(opentitantool_path, iterations)
    char_conditional_branch_bne_test(opentitantool_path, iterations)
    char_csr_read_test(opentitantool_path, iterations)
    char_csr_write_test(opentitantool_path, iterations)
    char_csr_combi_test(opentitantool_path, iterations)
    char_flash_read_test(opentitantool_path, iterations)
    char_flash_write_test(opentitantool_path, iterations)
    char_mem_op_loop_test(opentitantool_path, iterations)
    char_register_file_test(opentitantool_path, iterations)
    char_register_file_read_test(opentitantool_path, iterations)
    char_reg_op_loop_test(opentitantool_path, iterations)
    char_single_beq_test(opentitantool_path, iterations)
    char_single_bne_test(opentitantool_path, iterations)
    char_sram_read_test(opentitantool_path, iterations)
    char_sram_read_ret_test(opentitantool_path, iterations)
    char_sram_static_test(opentitantool_path, iterations)
    char_sram_write_test(opentitantool_path, iterations)
    char_sram_write_read_test(opentitantool_path, iterations)
    char_sram_write_read_alt_test(opentitantool_path, iterations)
    char_sram_write_static_unrolled_test(opentitantool_path, iterations)
    char_unconditional_branch_test(opentitantool_path, iterations)
    char_unconditional_branch_nop_test(opentitantool_path, iterations)
    char_unrolled_mem_op_loop_test(opentitantool_path, iterations)
    char_unrolled_reg_op_loop_test(opentitantool_path, iterations)
    char_unrolled_reg_op_loop_chain_test(opentitantool_path, iterations)
    char_otp_data_read_test(opentitantool_path, iterations)
    char_otp_read_lock_test(opentitantool_path, iterations)
    char_addi_single_beq_cm_test(opentitantool_path)
    char_addi_single_beq_cm2_test(opentitantool_path)
    char_hardened_check_eq_unimp_test(opentitantool_path)
    char_hardened_check_eq_2_unimps_test(opentitantool_path)
    char_hardened_check_eq_3_unimps_test(opentitantool_path)
    char_hardened_check_eq_4_unimps_test(opentitantool_path)
    char_hardened_check_eq_5_unimps_test(opentitantool_path)

    print("Testing finished")
    return True

if __name__ == "__main__":
  main()
