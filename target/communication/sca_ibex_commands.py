# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the Ibex SCA application on OpenTitan.

Communication with OpenTitan happens over the uJson command interface.
"""
import json
import time
from typing import Optional


class OTIbex:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        if protocol == "simpleserial":
            raise RuntimeError("Error: Simpleserial not supported!")

    def _ujson_ibex_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("IbexSca").encode("ascii"))

    def _ujson_ibex_sca_ack(self, num_attempts: Optional[int] = 100):
        # Wait for ack.
        read_counter = 0
        while read_counter < num_attempts:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                try:
                    if "result" in json_string:
                        status = json.loads(json_string)["result"]
                        if status != 0:
                            raise Exception("Acknowledge error: Device and host not in sync")
                        return status
                except Exception:
                    raise Exception("Acknowledge error: Device and host not in sync")
            else:
                read_counter += 1
        raise Exception("Acknowledge error: Device and host not in sync")

    def init(self):
        """ Initializes the Ibex SCA tests on the target.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # Init the Ibex SCA tests.
        self.target.write(json.dumps("Init").encode("ascii"))

    def register_file_read(self, num_iterations: int, data: list[int]):
        """ Start ibex.sca.register_file_read test.
        Args:
            num_iterations: The number of iterations the RF is read.
            data: The data that is first written into the RF and then read back.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # RFRead command.
        self.target.write(json.dumps("RFRead").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_iterations, "data": data}
        self.target.write(json.dumps(data).encode("ascii"))
        # Wait for ack.
        time.sleep(0.01)
        self._ujson_ibex_sca_ack()

    def register_file_write(self, num_iterations: int, data: list[int]):
        """ Start ibex.sca.register_file_write test.
        Args:
            num_iterations: The number of iterations the RF is written.
            data: The data that is written into the RF.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # RFWrite command.
        self.target.write(json.dumps("RFWrite").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_iterations, "data": data}
        self.target.write(json.dumps(data).encode("ascii"))
        # Wait for ack.
        time.sleep(0.01)
        self._ujson_ibex_sca_ack()

    def tl_write(self, num_iterations: int, data: list[int]):
        """ Start ibex.sca.tl_write test.
        Args:
            num_iterations: The number of iterations the RF is written.
            data: The data that is written into the SRAM over Tl-UL.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLWrite command.
        self.target.write(json.dumps("TLWrite").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_iterations, "data": data}
        self.target.write(json.dumps(data).encode("ascii"))
        # Wait for ack.
        time.sleep(0.01)
        self._ujson_ibex_sca_ack()

    def tl_read(self, num_iterations: int, data: list[int]):
        """ Start ibex.sca.tl_read test.
        Args:
            num_iterations: The number of iterations the RF is written.
            data: The data that is written into the SRAM over Tl-UL.
        """
        # IbexSca command.
        self._ujson_ibex_sca_cmd()
        # TLRead command.
        self.target.write(json.dumps("TLRead").encode("ascii"))
        # Data payload.
        time.sleep(0.01)
        data = {"num_iterations": num_iterations, "data": data}
        self.target.write(json.dumps(data).encode("ascii"))
        # Wait for ack.
        time.sleep(0.01)
        self._ujson_ibex_sca_ack()
