# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the SHA3 SCA application on OpenTitan.

Communication with OpenTitan either happens over simpleserial or the uJson
command interface.
"""
import json
import time
from typing import Optional


class OTSHA3:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        self.simple_serial = True
        if protocol == "ujson":
            self.simple_serial = False

    def _ujson_sha3_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("Sha3Sca").encode("ascii"))

    def init(self, fpga_mode_bit: int):
        """ Initializes SHA3 on the target.
        Args:
            fpga_mode_bit: Indicates whether FPGA specific KMAC test is started.
        """
        if not self.simple_serial:
            # Sha3Sca command.
            self._ujson_sha3_sca_cmd()
            # Init the SHA3 core.
            self.target.write(json.dumps("Init").encode("ascii"))
            # FPGA mode.
            time.sleep(0.01)
            fpga_mode = {"fpga_mode": fpga_mode_bit}
            self.target.write(json.dumps(fpga_mode).encode("ascii"))
            parameters = {"icache_disable": True, "dummy_instr_disable": True, "enable_jittery_clock": False, "enable_sram_readback": False}
            self.target.write(json.dumps(parameters).encode("ascii"))

    def _ujson_sha3_sca_ack(self, num_attempts: Optional[int] = 100):
        # Wait for ack.
        read_counter = 0
        while read_counter < num_attempts:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                try:
                    if "status" in json_string:
                        status = json.loads(json_string)["status"]
                        if status != 0:
                            raise Exception("Acknowledge error: Device and host not in sync")
                        return status
                except Exception:
                    raise Exception("Acknowledge error: Device and host not in sync")
            else:
                read_counter += 1
        raise Exception("Acknowledge error: Device and host not in sync")

    def set_mask_off(self):
        if self.simple_serial:
            self.target.write(cmd="m", data=bytearray([0x01]))
            ack_ret = self.target.wait_ack(5000)
            if ack_ret is None:
                raise Exception("Device and host not in sync")
        else:
            # Sha3Sca command.
            self._ujson_sha3_sca_cmd()
            # DisableMasking command.
            self.target.write(json.dumps("DisableMasking").encode("ascii"))
            # Num_segments payload.
            time.sleep(0.01)
            mask = {"masks_off": 1}
            self.target.write(json.dumps(mask).encode("ascii"))
            # Wait for ack.
            self._ujson_sha3_sca_ack()

    def set_mask_on(self):
        if self.simple_serial:
            self.target.write(cmd="m", data=bytearray([0x00]))
            ack_ret = self.target.wait_ack(5000)
            if ack_ret is None:
                raise Exception("Device and host not in sync")
        else:
            # Sha3Sca command.
            self._ujson_sha3_sca_cmd()
            # DisableMasking command.
            self.target.write(json.dumps("DisableMasking").encode("ascii"))
            # Num_segments payload.
            time.sleep(0.01)
            mask = {"masks_off": 0}
            self.target.write(json.dumps(mask).encode("ascii"))
            # Wait for ack.
            self._ujson_sha3_sca_ack()

    def absorb(self, text, text_length: Optional[int] = 16):
        """ Write plaintext to OpenTitan SHA3 & start absorb.
        Args:
            text: The plaintext bytearray.
        """
        if self.simple_serial:
            self.target.write(cmd="p", data=text)
        else:
            # Sha3Sca command.
            self._ujson_sha3_sca_cmd()
            # SingleAbsorb command.
            self.target.write(json.dumps("SingleAbsorb").encode("ascii"))
            # SingleAbsorb payload.
            time.sleep(0.01)
            text_int = [x for x in text]
            text_data = {"msg": text_int, "msg_length": text_length}
            self.target.write(json.dumps(text_data).encode("ascii"))

    def absorb_batch(self, num_segments):
        """ Start absorb for batch.
        Args:
            num_segments: Number of hashings to perform.
        """
        if self.simple_serial:
            self.target.write(cmd="b", data=num_segments)
            ack_ret = self.target.wait_ack(5000)
            if ack_ret is None:
                raise Exception("Batch mode acknowledge error: Device and host not in sync")
        else:
            # Sha3Sca command.
            self._ujson_sha3_sca_cmd()
            # Batch command.
            self.target.write(json.dumps("Batch").encode("ascii"))
            # Num_segments payload.
            time.sleep(0.01)
            num_segments_data = {"data": [x for x in num_segments]}
            self.target.write(json.dumps(num_segments_data).encode("ascii"))
            # Wait for ack.
            self._ujson_sha3_sca_ack()

    def write_lfsr_seed(self, seed):
        """ Seed the LFSR.
        Args:
            seed: The 4-byte seed.
        """
        if self.simple_serial:
            self.target.write(cmd="l", data=seed)
        else:
            # Sha3Sca command.
            self._ujson_sha3_sca_cmd()
            # SeedLfsr command.
            self.target.write(json.dumps("SeedLfsr").encode("ascii"))
            # Seed payload.
            time.sleep(0.01)
            seed_int = [x for x in seed]
            seed_data = {"seed": seed_int}
            self.target.write(json.dumps(seed_data).encode("ascii"))

    def fvsr_fixed_msg_set(self, msg, msg_length: Optional[int] = 16):
        """ Write the fixed message to SHA3.
        Args:
            msg: Bytearray containing the message.
        """
        if self.simple_serial:
            self.target.write(cmd="f", data=bytearray(msg))
        else:
            # Sha3Sca command.
            self._ujson_sha3_sca_cmd()
            # FixedMessageSet command.
            self.target.write(json.dumps("FixedMessageSet").encode("ascii"))
            # Msg payload.
            time.sleep(0.01)
            msg_int = [x for x in msg]
            msg_data = {"msg": msg_int, "msg_length": msg_length}
            self.target.write(json.dumps(msg_data).encode("ascii"))

    def read_ciphertext(self, len_bytes, num_attempts: Optional[int] = 100):
        """ Read ciphertext from OpenTitan SHA3.
        Args:
            len_bytes: Number of bytes to read.
            num_attempts: Number of attempts to read from device.

        Returns:
            The received ciphertext and a status flag indicating whether data
            was received or not.
        """
        if self.simple_serial:
            response_byte = self.target.read("r", len_bytes, ack=False)
            # Convert response into int array.
            return [x for x in response_byte], True
        else:
            read_counter = 0
            while read_counter < num_attempts:
                read_line = str(self.target.readline())
                if "RESP_OK" in read_line:
                    json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                    try:
                        batch_digest = json.loads(json_string)["batch_digest"]
                        return batch_digest[0:len_bytes], True
                    except Exception:
                        pass  # noqa: E302
                read_counter += 1
            # Reading from device failed.
            return None, False

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
