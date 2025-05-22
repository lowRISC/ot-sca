# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from test.penetrationtests.sca.host_scripts import sca_otbn_functions
from target.communication.sca_otbn_commands import OTOTBN
from python.runfiles import Runfiles
from target.chip import Chip
from target.dut import DUT
from test.penetrationtests.util import utils
import os
import json
import random
import unittest

ignored_keys_set = set([])
opentitantool_path = ""
iterations = 3
num_segments_list = [1, 5, 10, 12]


class AesScaTest(unittest.TestCase):

    def test_init(self):
        target = DUT()
        otbnsca = OTOTBN(target)
        device_id, owner_page, boot_log, boot_measurements, version = otbnsca.init()
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

    def test_char_combi_operations_batch(self):
        for num_segments in num_segments_list:
            trigger = 3
            fixed_data1 = 0
            fixed_data2 = 0
            print_flag = True
            actual_result = sca_otbn_functions.char_combi_operations_batch(
                opentitantool_path,
                iterations,
                num_segments,
                fixed_data1,
                fixed_data2,
                print_flag,
                trigger,
            )
            actual_result_json = json.loads(actual_result)

            # Calculate the expected result
            fixed_data_array1 = [fixed_data1 for _ in range(8)]
            fixed_data_array2 = [fixed_data2 for _ in range(8)]
            add = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) + utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            sub = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) - utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            xor = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) ^ utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            shift = utils.int_to_array((utils.array_to_int(fixed_data_array1) << 1) % (1 << 256))
            fixed_data_array1 = [fixed_data1, fixed_data1, 0, 0, 0, 0, 0, 0]
            fixed_data_array2 = [fixed_data2, fixed_data2, 0, 0, 0, 0, 0, 0]
            mult = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) * utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            FG = 0
            if fixed_data1 == fixed_data2:
                FG += 8
            if fixed_data1 < fixed_data2:
                FG += 1
            if sub[0] & 0x1:
                FG += 4
            if (utils.array_to_int(sub) >> 255) & 0x1:
                FG += 2

            expected_result_json = {
                "result1": [
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                ],
                "result2": [
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                ],
                "result3": add,
                "result4": sub,
                "result5": xor,
                "result6": shift,
                "result7": mult,
                "result8": FG,
            }
            utils.compare_json_data(
                actual_result_json, expected_result_json, ignored_keys_set
            )

            fixed_data1 = 1
            fixed_data2 = 1
            print_flag = True
            actual_result = sca_otbn_functions.char_combi_operations_batch(
                opentitantool_path,
                iterations,
                num_segments,
                fixed_data1,
                fixed_data2,
                print_flag,
                trigger,
            )
            actual_result_json = json.loads(actual_result)

            # Calculate the expected result
            fixed_data_array1 = [fixed_data1 for _ in range(8)]
            fixed_data_array2 = [fixed_data2 for _ in range(8)]
            add = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) + utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            sub = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) - utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            xor = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) ^ utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            shift = utils.int_to_array((utils.array_to_int(fixed_data_array1) << 1) % (1 << 256))
            fixed_data_array1 = [fixed_data1, fixed_data1, 0, 0, 0, 0, 0, 0]
            fixed_data_array2 = [fixed_data2, fixed_data2, 0, 0, 0, 0, 0, 0]
            mult = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) * utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            FG = 0
            if fixed_data1 == fixed_data2:
                FG += 8
            if fixed_data1 < fixed_data2:
                FG += 1
            if sub[0] & 0x1:
                FG += 4
            if (utils.array_to_int(sub) >> 255) & 0x1:
                FG += 2

            expected_result_json = {
                "result1": [
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                ],
                "result2": [
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                ],
                "result3": add,
                "result4": sub,
                "result5": xor,
                "result6": shift,
                "result7": mult,
                "result8": FG,
            }
            utils.compare_json_data(
                actual_result_json, expected_result_json, ignored_keys_set
            )

            fixed_data1 = random.getrandbits(32)
            fixed_data2 = random.getrandbits(32)
            print_flag = True
            actual_result = sca_otbn_functions.char_combi_operations_batch(
                opentitantool_path,
                iterations,
                num_segments,
                fixed_data1,
                fixed_data2,
                print_flag,
                trigger,
            )
            actual_result_json = json.loads(actual_result)

            # Calculate the expected result
            fixed_data_array1 = [fixed_data1 for _ in range(8)]
            fixed_data_array2 = [fixed_data2 for _ in range(8)]
            add = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) + utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            sub = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) - utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            xor = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) ^ utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            shift = utils.int_to_array((utils.array_to_int(fixed_data_array1) << 1) % (1 << 256))
            fixed_data_array1 = [fixed_data1, fixed_data1, 0, 0, 0, 0, 0, 0]
            fixed_data_array2 = [fixed_data2, fixed_data2, 0, 0, 0, 0, 0, 0]
            mult = utils.int_to_array(
                (utils.array_to_int(fixed_data_array1) * utils.array_to_int(fixed_data_array2))
                % (1 << 256)
            )
            FG = 0
            if fixed_data1 == fixed_data2:
                FG += 8
            if fixed_data1 < fixed_data2:
                FG += 1
            if sub[0] & 0x1:
                FG += 4
            if (utils.array_to_int(sub) >> 255) & 0x1:
                FG += 2

            expected_result_json = {
                "result1": [
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                ],
                "result2": [
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                    fixed_data1,
                ],
                "result3": add,
                "result4": sub,
                "result5": xor,
                "result6": shift,
                "result7": mult,
                "result8": FG,
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
