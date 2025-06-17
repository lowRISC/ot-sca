# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan Rom FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


class OTFIRom:
    def __init__(self, target) -> None:
        self.target = target

    def _ujson_rom_fi_cmd(self) -> None:
        time.sleep(0.01)
        self.target.write(json.dumps("RomFi").encode("ascii"))

    def rom_read(self) -> None:
        """ Reads Rom digest.
        """
        # RomFi command.
        self._ujson_rom_fi_cmd()
        # Read command.
        time.sleep(0.01)
        self.target.write(json.dumps("Read").encode("ascii"))

    def init(self) -> list:
        """ Initialize the ROM FI code on the chip.
        Args:
            cfg: Config dict containing the selected test.
            
        Returns:
            Device id
            The owner info page
            The boot log
            The boot measurements
            The testOS version
        """
        # RomFi command.
        self._ujson_rom_fi_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))
        parameters = {"enable_icache": True, "enable_dummy_instr": True, "dummy_instr_count": 3, "enable_jittery_clock": False, "enable_sram_readback": False}
        self.target.write(json.dumps(parameters).encode("ascii"))
        parameters = {"sensor_ctrl_enable": True, "sensor_ctrl_en_fatal": [False, False, False, False, False, False, False, False, False, False, False]}
        self.target.write(json.dumps(parameters).encode("ascii"))
        parameters = {"alert_classes":[2,2,2,2,0,0,2,2,2,2,0,0,0,0,0,1,0,0,0,2,2,2,0,0,0,1,0,2,2,2,2,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1], "accumulation_threshold": 2, "signals": [4294967295, 0, 2, 3], "duration_cycles": [0, 2400000,48,48], "ping_timeout": 1200}
        self.target.write(json.dumps(parameters).encode("ascii"))
        device_id = self.read_response()
        sensors = self.read_response()
        alerts = self.read_response()
        owner_page = self.read_response()
        boot_log = self.read_response()
        boot_measurements = self.read_response()
        version = self.read_response()
        return device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version

    def read_response(self, max_tries: Optional[int] = 1) -> str:
        """ Read response from Rom FI framework.
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
