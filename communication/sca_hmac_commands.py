# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the HMAC SCA application on OpenTitan.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time


class OTHMAC:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        if protocol == "simpleserial":
            raise Exception("Only uJSON protocol is supported for this test.")

    def _ujson_hmac_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("HmacSca").encode("ascii"))
        time.sleep(0.01)

    def init(self, jittery_clock):
        """ Initializes HMAC on the target.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # Init command.
        self.target.write(json.dumps("Init").encode("ascii"))
        time.sleep(0.01)
        data = {"icache_disable": True, "dummy_instr_disable": True, "enable_jittery_clock": jittery_clock, "enable_sram_readback": False}
        self.target.write(json.dumps(data).encode("ascii"))

    def test(self, key: list[int], num_segments: int):
         # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # Single command.
        self.target.write(json.dumps("Test").encode("ascii"))
        # Key payload.
        time.sleep(0.01)
        key_data = {"key": key}
        self.target.write(json.dumps(key_data).encode("ascii"))
        # Number of iterations payload.
        time.sleep(0.05)
        num_it_data = {"num_iterations": num_segments}
        self.target.write(json.dumps(num_it_data).encode("ascii"))

    def single(self, msg: list[int], key: list[int]):
        """ Start a single HMAC operation using the given message and key.
        Args:
            msg: The list containing the message.
            key: The key containing the message.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # Single command.
        self.target.write(json.dumps("Single").encode("ascii"))
        # Key payload.
        time.sleep(0.01)
        key_data = {"key": key}
        self.target.write(json.dumps(key_data).encode("ascii"))
        # Message payload.
        time.sleep(0.01)
        msg_data = {"message": msg}
        self.target.write(json.dumps(msg_data).encode("ascii"))

    def fvsr_batch(self, key: list[int], num_segments: int):
        """ Start num_segments HMAC operation in FvsR batch mode.
        Args:
            key: The key containing the message.
            num_segments: The number of iterations.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # BatchFvsr command.
        self.target.write(json.dumps("BatchFvsr").encode("ascii"))
        # Key payload.
        time.sleep(0.01)
        key_data = {"key": key}
        self.target.write(json.dumps(key_data).encode("ascii"))
        # Number of iterations payload.
        time.sleep(0.05)
        num_it_data = {"num_iterations": num_segments}
        self.target.write(json.dumps(num_it_data).encode("ascii"))

    def random_batch(self, num_segments: int):
        """ Start num_segments HMAC operations in random batch mode.
        Args:
            num_segments: The number of iterations.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # BatchRandom command.
        self.target.write(json.dumps("BatchRandom").encode("ascii"))
        # Number of iterations payload.
        time.sleep(0.01)
        num_it_data = {"num_iterations": num_segments}
        self.target.write(json.dumps(num_it_data).encode("ascii"))
        
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
    
    def crypto_sha2(self, msg, calculation_trigger) -> None:
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # Sha2 command.
        time.sleep(0.01)
        self.target.write(json.dumps("Sha2").encode("ascii"))
        time.sleep(0.01)
        if calculation_trigger:
            mode = {"message": msg, "update_trigger": False,
                    "final_trigger": True}
        else:
            mode = {"message": msg, "update_trigger": True,
                    "final_trigger": False}
        self.target.write(json.dumps(mode).encode("ascii"))

    def read_tag(self):
        """ Read tag from OpenTitan HMAC.

        Returns:
            The received tag.
        """
        while True:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                try:
                    tag = json.loads(json_string)["tag"]
                    return tag
                except Exception:
                    pass  # noqa: E302
                
    def read_digest(self):
        """ Read tag from OpenTitan HMAC.

        Returns:
            The received tag.
        """
        while True:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                try:
                    tag = json.loads(json_string)["digest"]
                    return tag
                except Exception:
                    pass  # noqa: E302
