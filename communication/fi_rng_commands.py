# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan RNG FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


class OTFIRng:
    def __init__(self, target) -> None:
        self.target = target

    def _ujson_rng_cmd(self) -> None:
        time.sleep(0.01)
        self.target.write(json.dumps("RngFi").encode("ascii"))
        time.sleep(0.01)

    def init(self, test: str) -> None:
        """ Initialize the RNG FI code on the chip.

        Args:
            test: The selected test.
        """
        # RngFi command.
        self._ujson_rng_cmd()
        # Init command.
        time.sleep(0.01)
        if "csrng" in test:
            self.target.write(json.dumps("CsrngInit").encode("ascii"))
        else:
            self.target.write(json.dumps("EdnInit").encode("ascii"))
        parameters = {"icache_disable": True, "dummy_instr_disable": True, "enable_jittery_clock": False, "enable_sram_readback": False}
        self.target.write(json.dumps(parameters).encode("ascii"))

    def rng_csrng_bias(self) -> None:
        """ Starts the rng_csrng_bias test.
        """
        # RngFi command.
        time.sleep(0.05)
        self._ujson_rng_cmd()
        # CsrngBias command.
        time.sleep(0.05)
        self.target.write(json.dumps("CsrngBias").encode("ascii"))

    def rng_edn_resp_ack(self) -> None:
        """ Starts the rng_edn_resp_ack test.
        """
        # RngFi command.
        time.sleep(0.05)
        self._ujson_rng_cmd()
        # EdnRespAck command.
        time.sleep(0.05)
        self.target.write(json.dumps("EdnRespAck").encode("ascii"))

    def rng_edn_bias(self) -> None:
        """ Starts the rng_edn_bias test.
        """
        # RngFi command.
        time.sleep(0.05)
        self._ujson_rng_cmd()
        # EdnBias command.
        time.sleep(0.05)
        self.target.write(json.dumps("EdnBias").encode("ascii"))

    def rng_fw_overwrite(self, init: Optional[bool] = False,
                         disable_health_check: Optional[bool] = False) -> None:
        """ Starts the rng_fw_overwrite test.

        Args:
            init: Using disable_health_check is only possible at the very first
              rng_fw_overwrite test. Afterwards this option cannot be switched.
            disable_health_check: Turn the health check on or off.
        """
        # RngFi command.
        time.sleep(0.05)
        self._ujson_rng_cmd()
        # FWOverride command.
        time.sleep(0.05)
        self.target.write(json.dumps("FWOverride").encode("ascii"))
        if init:
            data = {"disable_health_check": disable_health_check}
            self.target.write(json.dumps(data).encode("ascii"))

    def start_test(self, cfg: dict) -> None:
        """ Start the selected test.

        Call the function selected in the config file. Uses the getattr()
        construct to call the function.

        Args:
            cfg: Config dict containing the selected test.
        """
        test_function = getattr(self, cfg["test"]["which_test"])
        test_function()

    def read_response(self, max_tries: Optional[int] = 1) -> str:
        """ Read response from RNG FI framework.
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
