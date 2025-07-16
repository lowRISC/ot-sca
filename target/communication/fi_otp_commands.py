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

    def init(self) -> list:
        """ Initialize the Otp FI code on the chip.
        Args:
            cfg: Config dict containing the selected test.
            
        Returns:
            Device id
            The owner info page
            The boot log
            The boot measurements
            The testOS version
        """
        # OtpFi command.
        self._ujson_otp_fi_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))
        parameters = {"enable_icache": True, "enable_dummy_instr": True, "dummy_instr_count": 3, "enable_jittery_clock": True, "enable_sram_readback": True}
        self.target.write(json.dumps(parameters).encode("ascii"))
        parameters = {"sensor_ctrl_enable": True, "sensor_ctrl_en_fatal": [False, False, False, False, False, False, False, False, False, False, False]}
        self.target.write(json.dumps(parameters).encode("ascii"))
        parameters = {"alert_classes":[2,2,2,2,0,0,2,2,2,2,0,0,0,0,0,1,0,0,0,2,2,2,0,0,0,1,0,2,2,2,2,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1], "enable_alerts": [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True], "enable_classes": [True,True,False,False], "accumulation_thresholds": [2,2,2,2], "signals": [4294967295, 0, 2, 3], "duration_cycles": [0, 7200,48,48], "ping_timeout": 1200}
        self.target.write(json.dumps(parameters).encode("ascii"))
        device_id = self.target.read_response()
        sensors = self.target.read_response()
        alerts = self.target.read_response()
        owner_page = self.target.read_response()
        boot_log = self.target.read_response()
        boot_measurements = self.target.read_response()
        version = self.target.read_response()
        return device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version

    def start_test(self, cfg: dict) -> None:
        """ Start the selected test.

        Call the function selected in the config file. Uses the getattr()
        construct to call the function.

        Args:
            cfg: Config dict containing the selected test.
        """
        test_function = getattr(self, cfg["test"]["which_test"])
        test_function()
