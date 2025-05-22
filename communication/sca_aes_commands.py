# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the AES SCA application on OpenTitan.

Communication with OpenTitan either happens over simpleserial or the uJson
command interface.
"""
import json
import time
from typing import Optional


class OTAES:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        self.simple_serial = True
        if protocol == "ujson":
            self.simple_serial = False

    def _ujson_aes_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("AesSca").encode("ascii"))

    def init(self, fpga_mode_bit: int):
        """ Initializes AES on the target.
        Args:
            fpga_mode_bit: Indicates whether FPGA specific AES test is started.
        """
        if not self.simple_serial:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # Init the AES core.
            self.target.write(json.dumps("Init").encode("ascii"))
            # FPGA mode.
            time.sleep(0.01)
            fpga_mode = {"fpga_mode": fpga_mode_bit}
            self.target.write(json.dumps(fpga_mode).encode("ascii"))
            data = {"icache_disable": True, "dummy_instr_disable": True, "enable_jittery_clock": False, "enable_sram_readback": False}
            self.target.write(json.dumps(data).encode("ascii"))

    def key_set(self, key, key_length = 16):
        """ Write key to AES.
        Args:
            key: Bytearray containing the key.
        """
        if self.simple_serial:
            self.target.write(cmd="k", data=bytearray(key))
        else:
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
        if self.simple_serial:
            self.target.write(cmd="f", data=bytearray(key))
        else:
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
        if self.simple_serial:
            self.target.write(cmd="l", data=seed)
        else:
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
        if self.simple_serial:
            self.target.write(cmd="d", data=command.to_bytes(4, "little"))
            self.target.wait_ack()
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # FvsrKeyStartBatchGenerate command.
            self.target.write(json.dumps("FvsrKeyStartBatchGenerate").encode("ascii"))
            # Command.
            time.sleep(0.01)
            cmd = {"data": [command]}
            self.target.write(json.dumps(cmd).encode("ascii"))

    def write_fvsr_batch_generate(self, num_segments):
        """ Generate random plaintexts for FVSR.
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.write(cmd="g", data=num_segments)
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # FvsrKeyBatchGenerate command.
            self.target.write(json.dumps("FvsrKeyBatchGenerate").encode("ascii"))
            # Number of encryptions.
            time.sleep(0.01)
            num_encryption_data = {"data": [x for x in num_segments]}
            self.target.write(json.dumps(num_encryption_data).encode("ascii"))

    def batch_alternative_encrypt(self, num_segments):
        """ Start encryption for batch (alternative).
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.write(cmd="a", data=num_segments)
            self.target.wait_ack()
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # BatchEncrypt command.
            self.target.write(json.dumps("BatchAlternativeEncrypt").encode("ascii"))
            # Number of encryptions.
            time.sleep(0.01)
            num_encryption_data = {"data": [x for x in num_segments]}
            self.target.write(json.dumps(num_encryption_data).encode("ascii"))

    def batch_encrypt(self, num_segments):
        """ Start encryption for batch.
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.write(cmd="a", data=num_segments)
            self.target.wait_ack()
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # BatchEncrypt command.
            self.target.write(json.dumps("BatchEncrypt").encode("ascii"))
            # Number of encryptions.
            time.sleep(0.01)
            num_encryption_data = {"data": [x for x in num_segments]}
            self.target.write(json.dumps(num_encryption_data).encode("ascii"))

    def fvsr_key_batch_encrypt(self, num_segments):
        """ Start batch encryption for FVSR key.
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.write(cmd="e", data=num_segments)
            self.target.wait_ack()
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # FvsrKeyBatchEncrypt command.
            self.target.write(json.dumps("FvsrKeyBatchEncrypt").encode("ascii"))
            # Number of encryptions.
            time.sleep(0.01)
            num_encryption_data = {"data": [x for x in num_segments]}
            self.target.write(json.dumps(num_encryption_data).encode("ascii"))

    def fvsr_data_batch_encrypt(self, num_segments):
        """ Start batch encryption for FVSR data.
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.write(cmd = "h", data = num_segments)
            self.target.wait_ack()
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # FvsrKeyBatchEncrypt command.
            self.target.write(json.dumps("FvsrDataBatchEncrypt").encode("ascii"))
            # Number of encryptions.
            time.sleep(0.01)
            num_encryption_data = {"data": [x for x in num_segments]}
            self.target.write(json.dumps(num_encryption_data).encode("ascii"))

    def batch_plaintext_set(self, text, text_length = 16):
        """ Write plaintext to OpenTitan AES.

        This command is designed to set the initial plaintext for
        batch_alternative_encrypt.

        Args:
            text: The plaintext bytearray.
        """
        if self.simple_serial:
            self.target.write(cmd="i", data=bytearray(text))
        else:
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
        if self.simple_serial:
            self.target.write(cmd="p", data=bytearray(text))
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # SingleEncrypt command.
            self.target.write(json.dumps("SingleEncrypt").encode("ascii"))
            # Text payload.
            time.sleep(0.01)
            text_data = {"text": text, "text_length": text_length}
            self.target.write(json.dumps(text_data).encode("ascii"))

    def read_ciphertext(self, len_bytes):
        """ Read ciphertext from OpenTitan AES.
        Args:
            len_bytes: Number of bytes to read.

        Returns:
            The received ciphertext.
        """
        if self.simple_serial:
            response_byte = self.target.read("r", len_bytes, ack=False)
            # Convert response into int array.
            return [x for x in response_byte]
        else:
            count = 0
            while True:
                read_line = str(self.target.readline())
                if "RESP_OK" in read_line:
                    json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                    try:
                        ciphertext = json.loads(json_string)["ciphertext"]
                        return ciphertext[0:len_bytes]
                    except Exception:
                        pass  # noqa: E302
                else:
                    count += 1
                    time.sleep(0.1)
                    if count > 10:
                        break

    def read_response(self, max_tries = 1) -> str:
        """ Read response from AES SCA framework.
        Args:
            max_tries: Maximum number of attempts to read from UART.

        Returns:
            The JSON response of OpenTitan.
        """
        it = 0
        while it != max_tries:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                return read_line.split("RESP_OK:")[1].split(" CRC:")[0]
            it += 1
        return ""
    