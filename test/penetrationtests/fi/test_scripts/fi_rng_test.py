# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from test.penetrationtests.fi.host_scripts import fi_rng_functions
from target.communication.fi_rng_commands import OTFIRng
from python.runfiles import Runfiles
from target.chip import Chip
from target.dut import DUT
from test.penetrationtests.util import utils
import os
import json
import unittest

ignored_keys_set = set(["rand"])
opentitantool_path = ""
iterations = 3


class RngFiTest(unittest.TestCase):

    def test_init(self):
        target = DUT()
        rngfi = OTFIRng(target)
        device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
            rngfi.init("test")
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

    def test_char_csrng_bias(self):
        trigger = 0
        actual_result = fi_rng_functions.char_csrng_bias(
            opentitantool_path, iterations, trigger
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"res":0,"rand":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_edn_resp_ack(self):
        actual_result = fi_rng_functions.char_edn_resp_ack(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"collisions":0,"rand":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_edn_bias(self):
        actual_result = fi_rng_functions.char_edn_bias(opentitantool_path, iterations)
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"rand":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_entropy_bias(self):
        actual_result = fi_rng_functions.char_entropy_bias(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_fw_overwrite(self):
        disable_health_check = False
        actual_result = fi_rng_functions.char_fw_overwrite(
            opentitantool_path, iterations, disable_health_check
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

        disable_health_check = True
        actual_result = fi_rng_functions.char_fw_overwrite(
            opentitantool_path, iterations, disable_health_check
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )
        return True


if __name__ == "__main__":
    r = Runfiles.Create()
    opentitantool_path = r.Rlocation(
        "lowrisc_opentitan/sw/host/opentitantool/opentitantool"
    )

    firmware_target_name = os.environ.get(
        "SELECTED_FIRMWARE_TARGET", "pen_test_fi_silicon_owner_gb_rom_ext"
    )
    firmware_path = r.Rlocation(
        f"lowrisc_opentitan/sw/device/tests/penetrationtests/firmware/{firmware_target_name}.img"
    )

    target = DUT()

    chip = Chip(opentitantool_path)
    chip.flash_target(firmware_path)
    target.dump_all()

    unittest.main()
