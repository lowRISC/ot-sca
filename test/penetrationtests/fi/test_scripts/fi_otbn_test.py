# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from test.penetrationtests.fi.host_scripts import fi_otbn_functions
from target.communication.fi_otbn_commands import OTFIOtbn
from python.runfiles import Runfiles
from target.chip import Chip
from target.dut import DUT
from test.penetrationtests.util import utils
import os
import json
import unittest

opentitantool_path = ""
iterations = 3
ignored_keys_set = set(["data"])


class OtbnFiTest(unittest.TestCase):

    def test_init(self):
        target = DUT()
        otbnfi = OTFIOtbn(target)
        device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
            otbnfi.init()
        )
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
            "sram_ret_readback_en",
            "data_ind_timing_en",
        }
        actual_device_id_keys = set(device_id_json.keys())

        self.assertEqual(
            expected_device_id_keys,
            actual_device_id_keys,
            "device_id keys do not match",
        )

        expected_sensors_keys = {"sensor_ctrl_en", "sensor_ctrl_fatal"}
        actual_sensors_keys = set(sensors_json.keys())

        self.assertEqual(
            expected_sensors_keys, actual_sensors_keys, "sensor keys do not match"
        )

        expected_alerts_keys = {
            "alert_classes",
            "enabled_alerts",
            "enabled_classes",
            "accumulation_thresholds",
            "duration_cycles",
            "escalation_signals_en",
            "escalation_signals_map",
        }
        actual_alerts_keys = set(alerts_json.keys())

        self.assertEqual(
            expected_alerts_keys, actual_alerts_keys, "alert keys do not match"
        )

        expected_owner_page_keys = {
            "config_version",
            "sram_exec_mode",
            "ownership_key_alg",
            "update_mode",
            "min_security_version_bl0",
            "lock_constraint",
        }
        actual_owner_page_keys = set(owner_page_json.keys())

        self.assertEqual(
            expected_owner_page_keys,
            actual_owner_page_keys,
            "owner_page keys do not match",
        )

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
            "retention_ram_initialized",
        }
        actual_boot_log_keys = set(boot_log_json.keys())

        self.assertEqual(
            expected_boot_log_keys, actual_boot_log_keys, "boot_log keys do not match"
        )

        expected_boot_measurements_keys = {"bl0", "rom_ext"}
        actual_boot_measurements_keys = set(boot_measurements_json.keys())

        self.assertEqual(
            expected_boot_measurements_keys,
            actual_boot_measurements_keys,
            "boot_measurements keys do not match",
        )

        self.assertIn("PENTEST", version)

    def test_char_beq(self):
        actual_result = fi_otbn_functions.char_beq(opentitantool_path, iterations)
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"result":0,"insn_cnt":509,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_bn_rshi(self):
        data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        actual_result = fi_otbn_functions.char_bn_rshi(
            opentitantool_path, iterations, data
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"big_num":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"insn_cnt":109,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_bn_sel(self):
        data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        actual_result = fi_otbn_functions.char_bn_sel(
            opentitantool_path, iterations, data
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"big_num":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"insn_cnt":1014,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_bn_wsrr(self):
        actual_result = fi_otbn_functions.char_bn_wsrr(opentitantool_path, iterations)
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"res":0,"insn_cnt":1089,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_bne(self):
        actual_result = fi_otbn_functions.char_bne(opentitantool_path, iterations)
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"result":0,"insn_cnt":509,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_dmem_access(self):
        actual_result = fi_otbn_functions.char_dmem_access(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"res":0,"insn_cnt":271,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_dmem_write(self):
        actual_result = fi_otbn_functions.char_dmem_write(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"result":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"insn_cnt":1,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_dmem_op_loop(self):
        actual_result = fi_otbn_functions.char_dmem_op_loop(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"loop_counter":10000,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_reg_op_loop(self):
        actual_result = fi_otbn_functions.char_reg_op_loop(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"loop_counter":10000,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_jal(self):
        actual_result = fi_otbn_functions.char_jal(opentitantool_path, iterations)
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"result":0,"insn_cnt":505,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_lw(self):
        actual_result = fi_otbn_functions.char_lw(opentitantool_path, iterations)
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"result":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"insn_cnt":1084,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_mem(self):
        byte_offset = 0
        num_words = 4
        imem = False
        dmem = True
        actual_result = fi_otbn_functions.char_mem(
            opentitantool_path, iterations, byte_offset, num_words, imem, dmem
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"res":0,"imem_data":[0,0,0,0,0,0,0,0],"imem_addr":[0,0,0,0,0,0,0,0],"dmem_data":[0,0,0,0,0,0,0,0],"dmem_addr":[0,0,0,0,0,0,0,0],"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_rf(self):
        actual_result = fi_otbn_functions.char_rf(opentitantool_path, iterations)
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"res":0,"faulty_gpr":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"faulty_wdr":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_unrolled_dmem_op_loop(self):
        actual_result = fi_otbn_functions.char_unrolled_dmem_op_loop(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"loop_counter":100,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_unrolled_reg_op_loop(self):
        actual_result = fi_otbn_functions.char_unrolled_reg_op_loop(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"loop_counter":100,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_load_integrity(self):
        actual_result = fi_otbn_functions.char_load_integrity(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"result":0,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_pc(self):
        pc = 2224
        actual_result = fi_otbn_functions.char_pc(opentitantool_path, iterations, pc)
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"pc_dmem":2224,"pc_otbn":2224,"insn_cnt":472,"err_otbn":0,"err_ibx":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )


if __name__ == "__main__":
    r = Runfiles.Create()
    opentitantool_path = r.Rlocation(
        "lowrisc_opentitan/sw/host/opentitantool/opentitantool"
    )

    firmware_target_name = os.environ.get(
        "SELECTED_FIRMWARE_TARGET", "pen_test_fi_otbn_silicon_owner_gb_rom_ext"
    )
    firmware_path = r.Rlocation(
        f"lowrisc_opentitan/sw/device/tests/penetrationtests/firmware/{firmware_target_name}.img"
    )

    target = DUT()
    chip = Chip(opentitantool_path)
    chip.flash_target(firmware_path)
    target.dump_all()

    unittest.main()
