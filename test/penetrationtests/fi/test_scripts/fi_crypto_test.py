# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from test.penetrationtests.fi.host_scripts import fi_crypto_functions
from target.communication.fi_crypto_commands import OTFICrypto
from python.runfiles import Runfiles
from target.chip import Chip
from target.dut import DUT
from test.penetrationtests.util import utils
import os
import json
from Crypto.Hash import SHA256, SHA384, SHA512, HMAC
import unittest

opentitantool_path = ""
iterations = 5
ignored_keys_set = set(["share0", "share1"])


class CryptoFiTest(unittest.TestCase):

    def test_init(self):
        target = DUT()
        cryptofi = OTFICrypto(target)
        device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
            cryptofi.init()
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

    def test_char_aes(self):
        trigger = 0
        actual_result = fi_crypto_functions.char_aes(
            opentitantool_path, iterations, trigger
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"ciphertext":[141,145,88,155,234,129,16,92,221,12,69,21,69,208,99,12],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_kmac(self):
        trigger = 0
        actual_result = fi_crypto_functions.char_kmac(
            opentitantool_path, iterations, trigger
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"digest":[184,34,91,108,231,47,251,27],"digest_2nd":[142,188,186,201,216,47,203,192],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_kmac_state(self):
        actual_result = fi_crypto_functions.char_kmac_state(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"digest":[184,34,91,108,231,47,251,27],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_hmac(self):
        trigger = 0
        message_endianness_big = False
        digest_endianness_big = False
        key_endianness_big = False
        enable_hmac = False
        hash_mode = 0
        msg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        key = [0, 0, 0, 0, 0, 0, 0, 0]
        actual_result = fi_crypto_functions.char_hmac(
            opentitantool_path,
            iterations,
            msg,
            key,
            trigger,
            enable_hmac,
            message_endianness_big,
            digest_endianness_big,
            key_endianness_big,
            hash_mode,
        )
        actual_result_json = json.loads(actual_result)

        sha256 = SHA256.new()
        sha256.update(bytes(msg))
        expected_result = utils.bytes_to_words(bytearray(sha256.digest()))
        expected_result.reverse()
        # Pad the rest with zero
        expected_result += [0] * 8

        expected_result_json = {
            "tag": expected_result,
            "err_status": 0,
            "alerts": [0, 0, 0],
            "ast_alerts": [0, 0],
        }
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

        msg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        trigger = 0
        message_endianness_big = True
        digest_endianness_big = False
        key_endianness_big = False
        enable_hmac = False
        hash_mode = 0
        actual_result = fi_crypto_functions.char_hmac(
            opentitantool_path,
            iterations,
            msg,
            key,
            trigger,
            enable_hmac,
            message_endianness_big,
            digest_endianness_big,
            key_endianness_big,
            hash_mode,
        )
        actual_result_json = json.loads(actual_result)

        # Switch the endianness of the message
        msg = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
        sha256 = SHA256.new()
        sha256.update(bytes(msg))
        expected_result = utils.bytes_to_words(bytearray(sha256.digest()))
        expected_result.reverse()
        # Pad the rest with zero
        expected_result += [0] * 8

        expected_result_json = {
            "tag": expected_result,
            "err_status": 0,
            "alerts": [0, 0, 0],
            "ast_alerts": [0, 0],
        }
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

        msg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        trigger = 0
        message_endianness_big = False
        digest_endianness_big = False
        key_endianness_big = False
        enable_hmac = False
        hash_mode = 1  # SHA384
        actual_result = fi_crypto_functions.char_hmac(
            opentitantool_path,
            iterations,
            msg,
            key,
            trigger,
            enable_hmac,
            message_endianness_big,
            digest_endianness_big,
            key_endianness_big,
            hash_mode,
        )
        actual_result_json = json.loads(actual_result)

        sha384 = SHA384.new()
        sha384.update(bytes(msg))
        expected_result = utils.bytes_to_words(bytearray(sha384.digest()))
        expected_result.reverse()
        # Pad the rest with zero
        expected_result += [0] * 4

        expected_result_json = {
            "tag": expected_result,
            "err_status": 0,
            "alerts": [0, 0, 0],
            "ast_alerts": [0, 0],
        }
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

        msg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        trigger = 0
        message_endianness_big = False
        digest_endianness_big = False
        key_endianness_big = False
        enable_hmac = False
        hash_mode = 2  # SHA512
        actual_result = fi_crypto_functions.char_hmac(
            opentitantool_path,
            iterations,
            msg,
            key,
            trigger,
            enable_hmac,
            message_endianness_big,
            digest_endianness_big,
            key_endianness_big,
            hash_mode,
        )
        actual_result_json = json.loads(actual_result)

        sha512 = SHA512.new()
        sha512.update(bytes(msg))
        expected_result = utils.bytes_to_words(bytearray(sha512.digest()))
        expected_result.reverse()

        expected_result_json = {
            "tag": expected_result,
            "err_status": 0,
            "alerts": [0, 0, 0],
            "ast_alerts": [0, 0],
        }
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

        # HMAC tests

        msg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        key = [0, 1, 2, 3, 4, 5, 6, 7]
        trigger = 0
        message_endianness_big = False
        digest_endianness_big = False
        key_endianness_big = False
        enable_hmac = True
        hash_mode = 0  # SHA256
        actual_result = fi_crypto_functions.char_hmac(
            opentitantool_path,
            iterations,
            msg,
            key,
            trigger,
            enable_hmac,
            message_endianness_big,
            digest_endianness_big,
            key_endianness_big,
            hash_mode,
        )
        actual_result_json = json.loads(actual_result)

        hmac = HMAC.new(key=bytes(utils.words_to_bytes(key)), digestmod=SHA256)
        hmac.update(bytes(msg))
        expected_result = utils.bytes_to_words(bytearray(hmac.digest()))
        expected_result.reverse()
        expected_result += [0] * 8

        expected_result_json = {
            "tag": expected_result,
            "err_status": 0,
            "alerts": [0, 0, 0],
            "ast_alerts": [0, 0],
        }
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

        msg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        trigger = 0
        message_endianness_big = False
        digest_endianness_big = False
        key_endianness_big = False
        enable_hmac = True
        hash_mode = 1  # SHA384
        actual_result = fi_crypto_functions.char_hmac(
            opentitantool_path,
            iterations,
            msg,
            key,
            trigger,
            enable_hmac,
            message_endianness_big,
            digest_endianness_big,
            key_endianness_big,
            hash_mode,
        )
        actual_result_json = json.loads(actual_result)

        hmac = HMAC.new(key=bytes(utils.words_to_bytes(key)), digestmod=SHA384)
        hmac.update(bytes(msg))
        expected_result = utils.bytes_to_words(bytearray(hmac.digest()))
        expected_result.reverse()
        expected_result += [0] * 4

        expected_result_json = {
            "tag": expected_result,
            "err_status": 0,
            "alerts": [0, 0, 0],
            "ast_alerts": [0, 0],
        }
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

        msg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        trigger = 0
        message_endianness_big = False
        digest_endianness_big = False
        key_endianness_big = False
        enable_hmac = True
        hash_mode = 2  # SHA512
        actual_result = fi_crypto_functions.char_hmac(
            opentitantool_path,
            iterations,
            msg,
            key,
            trigger,
            enable_hmac,
            message_endianness_big,
            digest_endianness_big,
            key_endianness_big,
            hash_mode,
        )
        actual_result_json = json.loads(actual_result)

        hmac = HMAC.new(key=bytes(utils.words_to_bytes(key)), digestmod=SHA512)
        hmac.update(bytes(msg))
        expected_result = utils.bytes_to_words(bytearray(hmac.digest()))
        expected_result.reverse()

        expected_result_json = {
            "tag": expected_result,
            "err_status": 0,
            "alerts": [0, 0, 0],
            "ast_alerts": [0, 0],
        }
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_shadow_reg_access(self):
        actual_result = fi_crypto_functions.char_shadow_reg_access(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"result":[68162304,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
        )
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_shadow_reg_read(self):
        actual_result = fi_crypto_functions.char_shadow_reg_read(
            opentitantool_path, iterations
        )
        actual_result_json = json.loads(actual_result)
        expected_result_json = json.loads(
            '{"result":[0,0,0],"err_status":0,"alerts":[0,0,0],"ast_alerts":[0,0]}'
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
