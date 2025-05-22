# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from test.penetrationtests.sca.host_scripts import sca_aes_functions
from target.communication.sca_aes_commands import OTAES
from python.runfiles import Runfiles
from target.chip import Chip
from target.dut import DUT
from test.penetrationtests.util import utils
import os
import json
import random
from Crypto.Cipher import AES
import unittest

ignored_keys_set = set([])
opentitantool_path = ""
iterations = 3
num_segments_list = [1, 5, 10, 12]
# For testing, we only use the software trigger
fpga = 0


class AesScaTest(unittest.TestCase):

    def test_init(self):
        target = DUT()
        aessca = OTAES(target)
        device_id, owner_page, boot_log, boot_measurements, version = aessca.init(fpga)
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

    def test_char_aes_single_encrypt(self):
        # Note that setting this to false gives errors for production chips
        masking = True
        key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        text = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        actual_result = sca_aes_functions.char_aes_single_encrypt(
            opentitantool_path, iterations, fpga, masking, key, text
        )
        actual_result_json = json.loads(actual_result)

        cipher_gen = AES.new(bytes(key), AES.MODE_ECB)
        expected_result = [x for x in cipher_gen.encrypt(bytes(text))]

        expected_result_json = {"ciphertext": expected_result, "ciphertext_length": 16}
        utils.compare_json_data(
            actual_result_json, expected_result_json, ignored_keys_set
        )

    def test_char_aes_batch_daisy_chain(self):
        for num_segments in num_segments_list:
            # Note that setting this to false gives errors for production chips
            masking = True
            key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            text = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            actual_result = sca_aes_functions.char_aes_batch_daisy_chain(
                opentitantool_path, iterations, num_segments, fpga, masking, key, text
            )
            actual_result_json = json.loads(actual_result)

            cipher_gen = AES.new(bytes(key), AES.MODE_ECB)
            for _ in range(iterations):
                chained_text = text
                for __ in range(num_segments):
                    chained_text = [x for x in cipher_gen.encrypt(bytes(chained_text))]

            expected_result_json = {
                "ciphertext": chained_text,
                "ciphertext_length": len(chained_text),
            }
            utils.compare_json_data(
                actual_result_json, expected_result_json, ignored_keys_set
            )

    def test_char_aes_batch_fvsr_data(self):
        for num_segments in num_segments_list:
            # Note that setting this to false gives errors for production chips
            masking = True
            key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            text = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            actual_result = sca_aes_functions.char_aes_batch_fvsr_data(
                opentitantool_path, iterations, num_segments, fpga, masking, key, text
            )
            actual_result_json = json.loads(actual_result)

            # Set the syncrhonized randomness
            batch_prng_seed = 1
            random.seed(batch_prng_seed)

            # Generate the batch data
            for _ in range(iterations):
                sample_fixed = 1
                for __ in range(num_segments):
                    if sample_fixed == 1:
                        batch_data = text
                    else:
                        batch_data = [random.randint(0, 255) for _ in range(len(text))]
                    sample_fixed = random.randint(0, 255) & 0x1

            cipher_gen = AES.new(bytes(key), AES.MODE_ECB)
            expected_result = [x for x in cipher_gen.encrypt(bytes(batch_data))]

            expected_result_json = {
                "ciphertext": expected_result,
                "ciphertext_length": len(expected_result),
            }
            utils.compare_json_data(
                actual_result_json, expected_result_json, ignored_keys_set
            )

    def test_char_aes_batch_fvsr_key(self):
        for num_segments in num_segments_list:
            masking = True
            key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            actual_result = sca_aes_functions.char_aes_batch_fvsr_key(
                opentitantool_path, iterations, num_segments, fpga, masking, key
            )
            actual_result_json = json.loads(actual_result)

            # Set the syncrhonized randomness
            batch_prng_seed = 1
            random.seed(batch_prng_seed)

            # Generate the batch data
            for _ in range(iterations):
                sample_fixed = 1
                for __ in range(num_segments):
                    if sample_fixed == 1:
                        batch_key = key
                    else:
                        batch_key = [random.randint(0, 255) for _ in range(len(key))]
                    batch_data = [random.randint(0, 255) for _ in range(16)]
                    sample_fixed = random.randint(0, 255) & 0x1

            cipher_gen = AES.new(bytes(batch_key), AES.MODE_ECB)
            expected_result = [x for x in cipher_gen.encrypt(bytes(batch_data))]

            expected_result_json = {
                "ciphertext": expected_result,
                "ciphertext_length": len(expected_result),
            }
            utils.compare_json_data(
                actual_result_json, expected_result_json, ignored_keys_set
            )

    def test_char_aes_batch_random(self):
        for num_segments in num_segments_list:
            masking = True
            key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            actual_result = sca_aes_functions.char_aes_batch_random(
                opentitantool_path, iterations, num_segments, fpga, masking, key
            )
            actual_result_json = json.loads(actual_result)

            # Set the syncrhonized randomness
            batch_prng_seed = 1
            random.seed(batch_prng_seed)

            # Generate the batch data
            for _ in range(iterations):
                for __ in range(num_segments):
                    batch_data = [random.randint(0, 255) for _ in range(16)]

            cipher_gen = AES.new(bytes(key), AES.MODE_ECB)
            expected_result = [x for x in cipher_gen.encrypt(bytes(batch_data))]

            expected_result_json = {
                "ciphertext": expected_result,
                "ciphertext_length": len(expected_result),
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
