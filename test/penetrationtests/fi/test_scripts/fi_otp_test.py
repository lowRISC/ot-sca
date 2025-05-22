# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from test.penetrationtests.fi.host_scripts import fi_otp_functions
from target.communication.fi_otp_commands import OTFIOtp
from python.runfiles import Runfiles
from target.chip import Chip
from target.dut import DUT
from test.penetrationtests.util import utils
import os
import json
import unittest

ignored_keys_set = set(["partition_ref", "partition_fi", "otp_status_codes"])
opentitantool_path = ""
iterations = 3


class OtpFiTest(unittest.TestCase):

    def test_init(self):
        target = DUT()
        otpfi = OTFIOtp(target)
        device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
            otpfi.init()
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

    def test_char_vendor_test(self):
        actual_result = fi_otp_functions.char_vendor_test(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],"otp_error_causes":[0,0,0,0,0,0,0,0,0,0],"otp_status_codes":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_owner_sw_cfg(self):
        actual_result = fi_otp_functions.char_owner_sw_cfg(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],"otp_error_causes":[0,0,0,0,0,0,0,0,0,0],"otp_status_codes":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_hw_cfg(self):
        actual_result = fi_otp_functions.char_hw_cfg(opentitantool_path, iterations)
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],"otp_error_causes":[0,0,0,0,0,0,0,0,0,0],"otp_status_codes":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_life_cycle(self):
        actual_result = fi_otp_functions.char_life_cycle(opentitantool_path, iterations)
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"data_faulty":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],"otp_error_causes":[0,0,0,0,0,0,0,0,0,0],"otp_status_codes":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
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
