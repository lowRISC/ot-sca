# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan EDN FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


class OTFIEDN:
    def __init__(self, target) -> None:
        self.target = target

    def _ujson_edn_fi_cmd(self) -> None:
        time.sleep(0.01)
        self.target.write(json.dumps("EdnFi").encode("ascii"))

    def init(self) -> None:
        """ Initialize the EDN FI code on the chip.
        """
        # EdnFi command.
        self._ujson_edn_fi_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))

    def edn_fi_bus_ack(self) -> None:
        """ Starts the edn.fi.bus_ack test.
        """
        # EdnFi command.
        self._ujson_edn_fi_cmd()
        # BusAck command.
        time.sleep(0.01)
        self.target.write(json.dumps("BusAck").encode("ascii"))

    def edn_fi_bus_data(self) -> None:
        """ Starts the edn.fi.bus_data test.
        """
        # EdnFi command.
        self._ujson_edn_fi_cmd()
        # BusData command.
        time.sleep(0.01)
        self.target.write(json.dumps("BusData").encode("ascii"))

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
        """ Read response from EDN FI framework.
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
