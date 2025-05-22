# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from test.penetrationtests.sca.host_scripts import sca_kmac_functions
from target.communication.sca_kmac_commands import OTKMAC
from python.runfiles import Runfiles
from target.chip import Chip
from target.dut import DUT
from test.penetrationtests.util import utils
import os
import json
import random
from Crypto.Hash import KMAC128
import unittest

ignored_keys_set = set([])
opentitantool_path = ""
iterations = 3
num_segments_list = [1, 5, 10, 12]
# For testing, we only use the software trigger
fpga = 0


class KmacScaTest(unittest.TestCase):

    def test_init(self):
        target = DUT()
        kmacsca = OTKMAC(target)
        device_id, owner_page, boot_log, boot_measurements, version = kmacsca.init(fpga)
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
            "sram_ret_readback_en",
            "data_ind_timing_en",
        }
        actual_device_id_keys = set(device_id_json.keys())

        self.assertEqual(
            expected_device_id_keys,
            actual_device_id_keys,
            "device_id keys do not match",
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

    def test_char_kmac_single(self):
        key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        text = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        masking = True
        actual_result = sca_kmac_functions.char_kmac_single(
            opentitantool_path, iterations, fpga, masking, key, text
        )
        actual_result_json = json.loads(actual_result)

        mac = KMAC128.new(key=bytes(key), mac_len=32)
        mac.update(bytes(text))
        tag = bytearray(mac.digest())
        expected_result = [x for x in tag]

        expected_result_json = {
            "batch_digest": expected_result,
        }
        utils.compare_json_data(actual_result_json, expected_result_json, ignored_keys_set)

    def test_char_kmac_batch_daisy_chain(self):
        for num_segments in num_segments_list:
            key = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            text = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            masking = True
            actual_result = sca_kmac_functions.char_kmac_batch_daisy_chain(
                opentitantool_path, iterations, num_segments, fpga, masking, key, text
            )
            actual_result_json = json.loads(actual_result)

            for i in range(num_segments):
                mac = KMAC128.new(key=bytes(key), mac_len=32)
                mac.update(bytes(text))
                digest = [x for x in bytearray(mac.digest())]
                text = digest[:16]

            expected_result_json = {
                "digest": text,
            }
            utils.compare_json_data(
                actual_result_json, expected_result_json, ignored_keys_set
            )

    def test_char_kmac_batch(self):
        for num_segments in num_segments_list:
            key = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            masking = True
            actual_result = sca_kmac_functions.char_kmac_batch(
                opentitantool_path, iterations, num_segments, fpga, masking, key
            )
            actual_result_json = json.loads(actual_result)

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

            expected_result_json = {
                "batch_digest": xor_tag,
            }
            utils.compare_json_data(
                actual_result_json, expected_result_json, ignored_keys_set
            )


if __name__ == "__main__":
    r = Runfiles.Create()
    opentitantool_path = r.Rlocation(
        "lowrisc_opentitan/sw/host/opentitantool/opentitantool"
    )

    firmware_target_name = os.environ.get(
        "SELECTED_FIRMWARE_TARGET", "pen_test_sca_silicon_owner_gb_rom_ext"
    )
    firmware_path = r.Rlocation(
        f"lowrisc_opentitan/sw/device/tests/penetrationtests/firmware/{firmware_target_name}.img"
    )

    target = DUT()

    chip = Chip(opentitantool_path)
    chip.flash_target(firmware_path)
    target.dump_all()

    unittest.main()
