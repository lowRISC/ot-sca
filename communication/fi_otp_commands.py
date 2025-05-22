# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan Otp FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


class OTFIOtp:
    def __init__(self, target) -> None:
        self.target = target

    def _ujson_otp_fi_cmd(self) -> None:
        time.sleep(0.01)
        self.target.write(json.dumps("OtpFi").encode("ascii"))

    def otp_fi_vendor_test(self) -> None:
        """ Reads otp VENDOR_TEST partition.
        """
        # IbexFi command.
        self._ujson_otp_fi_cmd()
        # VendorTest command.
        time.sleep(0.01)
        self.target.write(json.dumps("VendorTest").encode("ascii"))

    def otp_fi_owner_sw_cfg(self) -> None:
        """ Reads otp OWNER_SW_CFG partition.
        """
        # IbexFi command.
        self._ujson_otp_fi_cmd()
        # OwnerSwCfg command.
        time.sleep(0.01)
        self.target.write(json.dumps("OwnerSwCfg").encode("ascii"))

    def otp_fi_hw_cfg(self) -> None:
        """ Reads otp HW_CFG partition.
        """
        # IbexFi command.
        self._ujson_otp_fi_cmd()
        # HwCfg command.
        time.sleep(0.01)
        self.target.write(json.dumps("HwCfg").encode("ascii"))

    def otp_fi_life_cycle(self) -> None:
        """ Reads otp LIFE_CYCLE partition.
        """
        # IbexFi command.
        self._ujson_otp_fi_cmd()
        # LifeCycle command.
        time.sleep(0.01)
        self.target.write(json.dumps("LifeCycle").encode("ascii"))

    def init(self) -> None:
        """ Initialize the Otp FI code on the chip.
        Args:
            cfg: Config dict containing the selected test.
        """
        # OtpFi command.
        self._ujson_otp_fi_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))
        parameters = {"icache_disable": True, "dummy_instr_disable": True, "enable_jittery_clock": False, "enable_sram_readback": False}
        self.target.write(json.dumps(parameters).encode("ascii"))

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
        """ Read response from Otp FI framework.
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
