# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the KMAC SCA application on OpenTitan.

Communication with OpenTitan either happens over simpleserial or the uJson
command interface.
"""
import json
import time
from typing import Optional


class OTKMAC:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        self.simple_serial = True
        if protocol == "ujson":
            self.simple_serial = False

    def _ujson_kmac_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("KmacSca").encode("ascii"))

    def init(self, fpga_mode_bit: int, jittery_clock):
        """ Initializes KMAC on the target.
        Args:
            fpga_mode_bit: Indicates whether FPGA specific KMAC test is started.
        """
        if not self.simple_serial:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # Init command.
            self.target.write(json.dumps("Init").encode("ascii"))
            # FPGA mode.
            time.sleep(0.01)
            fpga_mode = {"fpga_mode": fpga_mode_bit}
            self.target.write(json.dumps(fpga_mode).encode("ascii"))
            parameters = {"icache_disable": True, "dummy_instr_disable": True, "enable_jittery_clock": False, "enable_sram_readback": False}
            self.target.write(json.dumps(parameters).encode("ascii"))

    def write_key(self, key: list[int]):
        """ Write the key to KMAC.
        Args:
            key: Bytearray containing the key.
        """
        if self.simple_serial:
            self.target.write(cmd="k", data=bytearray(key))
        else:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # SetKey command.
            self.target.write(json.dumps("SetKey").encode("ascii"))
            # Key payload.
            time.sleep(0.01)
            key_data = {"key": key, "key_length": 16}
            self.target.write(json.dumps(key_data).encode("ascii"))

    def fvsr_key_set(self, key: list[int], key_length: Optional[int] = 16):
        """ Write the fixed key to KMAC.
        Args:
            key: Bytearray containing the key.
        """
        if self.simple_serial:
            self.target.write(cmd="f", data=bytearray(key))
        else:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # SetKey command.
            self.target.write(json.dumps("FixedKeySet").encode("ascii"))
            # FixedKeySet payload.
            time.sleep(0.01)
            key_data = {"key": key, "key_length": key_length}
            self.target.write(json.dumps(key_data).encode("ascii"))

    def write_lfsr_seed(self, seed):
        """ Seed the LFSR.
        Args:
            seed: The 4-byte seed.
        """
        if self.simple_serial:
            self.target.write(cmd="l", data=seed)
        else:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # SeedLfsr command.
            self.target.write(json.dumps("SeedLfsr").encode("ascii"))
            # Seed payload.
            time.sleep(0.01)
            seed_int = [x for x in seed]
            seed_data = {"seed": seed_int}
            self.target.write(json.dumps(seed_data).encode("ascii"))

    def absorb_batch(self, num_segments):
        """ Start absorb for batch.
        Args:
            num_segments: Number of encryptions to perform.
        """
        if self.simple_serial:
            self.target.write(cmd="b", data=num_segments)
        else:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # Batch command.
            self.target.write(json.dumps("Batch").encode("ascii"))
            # Num_segments payload.
            time.sleep(0.01)
            num_segments_data = {"data": [x for x in num_segments]}
            self.target.write(json.dumps(num_segments_data).encode("ascii"))

    def absorb(self, text, text_length: Optional[int] = 16):
        """ Write plaintext to OpenTitan KMAC & start absorb.
        Args:
            text: The plaintext bytearray.
        """
        if self.simple_serial:
            self.target.write(cmd="p", data=text)
        else:
            # KmacSca command.
            self._ujson_kmac_sca_cmd()
            # SingleAbsorb command.
            self.target.write(json.dumps("SingleAbsorb").encode("ascii"))
            # Msg payload.
            time.sleep(0.01)
            text_int = [x for x in text]
            text_data = {"msg": text_int, "msg_length": text_length}
            self.target.write(json.dumps(text_data).encode("ascii"))

    def read_ciphertext(self, len_bytes):
        """ Read ciphertext from OpenTitan KMAC.
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
            while True:
                read_line = str(self.target.readline())
                if "RESP_OK" in read_line:
                    json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                    try:
                        batch_digest = json.loads(json_string)["batch_digest"]
                        return batch_digest[0:len_bytes]
                    except Exception:
                        pass  # noqa: E302

    def read_response(self, max_tries: Optional[int] = 1) -> str:
        """ Read response from Ibex SCA framework.
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
