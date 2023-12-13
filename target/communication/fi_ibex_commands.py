# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional

import serial


class OTUART:
    def __init__(self, protocol: str, port: Optional[str] = "None",
                 baudrate: Optional[int] = 115200,
                 timeout: Optional[int] = 1) -> None:
        """ Inits the UART port.

        Args:
            protocol: uJSON for OT communication.
            port: The port OT is connected (e.g., /dev/ttyACM0)
            baudrate: Default is 115200
            timeout: Read timeout for read_response function.
        """
        self.uart = None
        if protocol == "ujson":
            if not port:
                raise RuntimeError("Error: No uJson port provided!")
            else:
                self.uart = serial.Serial(port)
                self.uart.baudrate = baudrate
                self.uart.timeout = timeout


class OTFIIbex:
    def __init__(self, uart: OTUART) -> None:
        self.uart = uart

    def _ujson_ibex_fi_cmd(self) -> None:
        time.sleep(0.01)
        self.uart.write(json.dumps("IbexFi").encode("ascii"))

    def ibex_char_unrolled_reg_op_loop(self) -> None:
        """ Starts the ibex.char.unrolled_reg_op_loop test.
        """
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharUnrolledRegOpLoop command.
        time.sleep(0.01)
        self.uart.write(json.dumps("CharUnrolledRegOpLoop").encode("ascii"))

    def ibex_char_unrolled_mem_op_loop(self) -> None:
        """ Starts the ibex.char.unrolled_mem_op_loop test.
        """
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharUnrolledMemOpLoop command.
        time.sleep(0.01)
        self.uart.write(json.dumps("CharUnrolledMemOpLoop").encode("ascii"))

    def ibex_char_reg_op_loop(self) -> None:
        """ Starts the ibex.char.reg_op_loop test.
        """
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharRegOpLoop command.
        time.sleep(0.01)
        self.uart.write(json.dumps("CharRegOpLoop").encode("ascii"))

    def ibex_char_mem_op_loop(self) -> None:
        """ Starts the ibex.char.mem_op_loop test.
        """
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # CharMemOpLoop command.
        time.sleep(0.01)
        self.uart.write(json.dumps("CharMemOpLoop").encode("ascii"))

    def init_trigger(self) -> None:
        """ Initialize the FI trigger on the chip.

        Args:
            cfg: Config dict containing the selected test.
        """
        # IbexFi command.
        self._ujson_ibex_fi_cmd()
        # InitTrigger command.
        time.sleep(0.01)
        self.uart.write(json.dumps("InitTrigger").encode("ascii"))

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
        """ Read response from Ibex FI framework.
        Args:
            max_tries: Maximum number of attempts to read from UART.

        Returns:
            The JSON response of OpenTitan.
        """
        it = 0
        while it != max_tries:
            read_line = str(self.uart.readline())
            if "RESP_OK" in read_line:
                return read_line.split("RESP_OK:")[1].split(" CRC:")[0]
            it += 1
        return ""
