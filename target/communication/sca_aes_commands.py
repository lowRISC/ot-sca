# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the AES SCA application on OpenTitan.

Communication with OpenTitan happens over the uJson
command interface.
"""
import json
import time
from typing import Optional


class OTAES:
    def __init__(self, target) -> None:
        self.target = target

    def _ujson_aes_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("AesSca").encode("ascii"))

    def init(self, fpga_mode_bit: int) -> list:
        """ Initializes AES on the target.
        Args:
            fpga_mode_bit: Indicates whether FPGA specific AES test is started.
            
        Returns:
            Device id
            The owner info page
            The boot log
            The boot measurements
            The testOS version
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # Init the AES core.
        self.target.write(json.dumps("Init").encode("ascii"))
        # FPGA mode.
        time.sleep(0.01)
        fpga_mode = {"fpga_mode": fpga_mode_bit}
        self.target.write(json.dumps(fpga_mode).encode("ascii"))
        parameters = {"enable_icache": True, "enable_dummy_instr": True, "dummy_instr_count": 3, "enable_jittery_clock": True, "enable_sram_readback": True}
        self.target.write(json.dumps(parameters).encode("ascii"))
        parameters = {"sensor_ctrl_enable": True, "sensor_ctrl_en_fatal": [False, False, False, False, False, False, False, False, False, False, False]}
        self.target.write(json.dumps(parameters).encode("ascii"))
        device_id = self.target.read_response()
        owner_page = self.target.read_response()
        boot_log = self.target.read_response()
        boot_measurements = self.target.read_response()
        version = self.target.read_response()
        return device_id, owner_page, boot_log, boot_measurements, version

    def key_set(self, key, key_length = 16):
        """ Write key to AES.
        Args:
            key: Bytearray containing the key.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # KeySet command.
        self.target.write(json.dumps("KeySet").encode("ascii"))
        # Key payload.
        time.sleep(0.01)
        key_data = {"key": key, "key_length": key_length}
        self.target.write(json.dumps(key_data).encode("ascii"))

    def fvsr_key_set(self, key, key_length: Optional[int] = 16):
        """ Write key to AES.
        Args:
            key: Bytearray containing the key.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # FvsrKeySet command.
        self.target.write(json.dumps("FvsrKeySet").encode("ascii"))
        # Key payload.
        time.sleep(0.01)
        key_data = {"key": key, "key_length": key_length}
        self.target.write(json.dumps(key_data).encode("ascii"))

    def seed_lfsr(self, seed):
        """ Seed the LFSR.
        Args:
            seed: The 4-byte seed.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # SeedLfsr command.
        self.target.write(json.dumps("SeedLfsr").encode("ascii"))
        # Seed payload.
        time.sleep(0.01)
        seed_data = {"seed": [x for x in seed]}
        self.target.write(json.dumps(seed_data).encode("ascii"))

    def start_fvsr_batch_generate(self, command):
        """Set SW PRNG to starting values for FvsR data
        generation.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # FvsrKeyStartBatchGenerate command.
        self.target.write(json.dumps("FvsrKeyStartBatchGenerate").encode("ascii"))
        # Command.
        time.sleep(0.01)
        cmd = {"cmd": command}
        self.target.write(json.dumps(cmd).encode("ascii"))

    def write_fvsr_batch_generate(self, num_segments):
        """ Generate random keys for FVSR.
        Args:
            num_segments: Number of encryptions to perform.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # FvsrKeyBatchGenerate command.
        self.target.write(json.dumps("FvsrKeyBatchGenerate").encode("ascii"))
        # Number of encryptions.
        time.sleep(0.01)
        num_encryption_data = {"num_enc": num_segments}
        self.target.write(json.dumps(num_encryption_data).encode("ascii"))

    def batch_alternative_encrypt(self, num_segments):
        """ Start encryption for batch (alternative).
        Args:
            num_segments: Number of encryptions to perform.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # BatchAlternativeEncrypt command.
        self.target.write(json.dumps("BatchAlternativeEncrypt").encode("ascii"))
        # Number of encryptions.
        time.sleep(0.01)
        num_encryption_data = {"num_enc": num_segments}
        self.target.write(json.dumps(num_encryption_data).encode("ascii"))

    def batch_encrypt(self, num_segments):
        """ Start encryption for batch.
        Args:
            num_segments: Number of encryptions to perform.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # BatchEncrypt command.
        self.target.write(json.dumps("BatchEncrypt").encode("ascii"))
        # Number of encryptions.
        time.sleep(0.01)
        num_encryption_data = {"num_enc": num_segments}
        self.target.write(json.dumps(num_encryption_data).encode("ascii"))

    def fvsr_key_batch_encrypt(self, num_segments):
        """ Start batch encryption for FVSR key.
        Args:
            num_segments: Number of encryptions to perform.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # FvsrKeyBatchEncrypt command.
        self.target.write(json.dumps("FvsrKeyBatchEncrypt").encode("ascii"))
        # Number of encryptions.
        time.sleep(0.01)
        num_encryption_data = {"num_enc": num_segments}
        self.target.write(json.dumps(num_encryption_data).encode("ascii"))

    def fvsr_data_batch_encrypt(self, num_segments):
        """ Start batch encryption for FVSR data.
        Args:
            num_segments: Number of encryptions to perform.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # FvsrKeyBatchEncrypt command.
        self.target.write(json.dumps("FvsrDataBatchEncrypt").encode("ascii"))
        # Number of encryptions.
        time.sleep(0.01)
        num_encryption_data = {"num_enc": num_segments}
        self.target.write(json.dumps(num_encryption_data).encode("ascii"))

    def batch_plaintext_set(self, text, text_length = 16):
        """ Write plaintext to OpenTitan AES.

        This command is designed to set the initial plaintext for
        batch_alternative_encrypt.

        Args:
            text: The plaintext bytearray.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # BatchPlaintextSet command.
        self.target.write(json.dumps("BatchPlaintextSet").encode("ascii"))
        # Text payload.
        time.sleep(0.01)
        text_int = [x for x in text]
        text_data = {"text": text_int, "text_length": text_length}
        self.target.write(json.dumps(text_data).encode("ascii"))

    def single_encrypt(self, text, text_length = 16):
        """ Write plaintext to OpenTitan AES & start encryption.
        Args:
            text: The plaintext bytearray.
        """
        # AesSca command.
        self._ujson_aes_sca_cmd()
        # SingleEncrypt command.
        self.target.write(json.dumps("SingleEncrypt").encode("ascii"))
        # Text payload.
        time.sleep(0.01)
        text_data = {"text": text, "text_length": text_length}
        self.target.write(json.dumps(text_data).encode("ascii"))
    