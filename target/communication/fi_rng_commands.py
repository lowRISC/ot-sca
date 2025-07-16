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

    def init(self, test: str) -> list:
        """ Initialize the RNG FI code on the chip.

        Args:
            test: The selected test.
        
        Returns:
            Device id
            The owner info page
            The boot log
            The boot measurements
            The testOS version
        """
        # RngFi command.
        self._ujson_rng_cmd()
        # Init command.
        time.sleep(0.01)
        if "csrng" in test:
            self.target.write(json.dumps("CsrngInit").encode("ascii"))
        else:
            self.target.write(json.dumps("EdnInit").encode("ascii"))
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

    def rng_csrng_bias(self, trigger: int) -> None:
        """ Starts the rng_csrng_bias test.
        """
        # RngFi command.
        time.sleep(0.05)
        self._ujson_rng_cmd()
        # CsrngBias command.
        time.sleep(0.05)
        self.target.write(json.dumps("CsrngBias").encode("ascii"))
        if trigger == 0:
            mode = {"start_trigger": True, "valid_trigger": False,
                "read_trigger": False, "all_trigger": False}
        elif trigger == 1:
            mode = {"start_trigger": False, "valid_trigger": True,
                "read_trigger": False, "all_trigger": False}
        elif trigger == 2:
            mode = {"start_trigger": False, "valid_trigger": False,
                "read_trigger": True, "all_trigger": False}
        elif trigger == 3:
            mode = {"start_trigger": False, "valid_trigger": False,
                "read_trigger": False, "all_trigger": True}
        self.target.write(json.dumps(mode).encode("ascii"))

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

    def rng_entropy_bias(self) -> None:
        """ Starts the entropy_src_bias test.
        """
        # RngFi command.
        time.sleep(0.05)
        self._ujson_rng_cmd()
        # EntropySrcBias command.
        time.sleep(0.05)
        self.target.write(json.dumps("EntropySrcBias").encode("ascii"))

    def rng_fw_overwrite(self, disable_health_check: Optional[bool] = False) -> None:
        """ Starts the rng_fw_overwrite test.

        Args:
            disable_health_check: Turn the health check on or off.
        """
        # RngFi command.
        time.sleep(0.05)
        self._ujson_rng_cmd()
        # FWOverride command.
        time.sleep(0.05)
        self.target.write(json.dumps("FWOverride").encode("ascii"))
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
