# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the HMAC SCA application on OpenTitan.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


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

    def init(self, enable_icache: bool, enable_dummy_instr: bool,
             enable_jittery_clock: bool, enable_sram_readback: bool) -> list:
        """ Initializes HMAC on the target.
        Args:
            enable_icache: If true, enable the iCache.
            enable_dummy_instr: If true, enable the dummy instructions.
            enable_jittery_clock: If true, enable the jittery clock.
            enable_sram_readback: If true, enable the SRAM readback feature.
        Returns:
            The device ID and countermeasure config of the device.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # Init command.
        self.target.write(json.dumps("Init").encode("ascii"))
        # Configure device and countermeasures.
        time.sleep(0.01)
        data = {"enable_icache": enable_icache, "enable_dummy_instr": enable_dummy_instr,
                "enable_jittery_clock": enable_jittery_clock,
                "enable_sram_readback": enable_sram_readback}
        self.target.write(json.dumps(data).encode("ascii"))
        # Read back device ID and countermeasure configuration from device.
        device_config = self.read_response(max_tries=30)
        # Read flash owner page.
        device_config += self.read_response(max_tries=30)
        # Read boot log.
        device_config += self.read_response(max_tries=30)
        # Read boot measurements.
        device_config += self.read_response(max_tries=30)
        # Read pentest framework version.
        device_config += self.read_response(max_tries=30)
        return device_config

    def single(self, msg: list[int], key: list[int], start_trigger: bool,
               msg_trigger: bool, process_trigger: bool, finish_trigger: bool):
        """ Start a single HMAC operation using the given message and key.
        Args:
            msg: The list containing the message.
            key: The key containing the message.
            start_trigger: Set trigger during the start phase.
            msg_trigger: Set trigger during the message pushing phase.
            process_trigger: Set trigger during the processing phase.
            finish_trigger: Set trigger during the finish phase.
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
        # Trigger configuration.
        time.sleep(0.05)
        trigger_data = {"start_trigger": start_trigger, "msg_trigger": msg_trigger,
                        "process_trigger": process_trigger, "finish_trigger": finish_trigger}
        self.target.write(json.dumps(trigger_data).encode("ascii"))

    def fvsr_batch(self, key: list[int], num_segments: int, start_trigger: bool,
                   msg_trigger: bool, process_trigger: bool, finish_trigger: bool):
        """ Start num_segments HMAC operation in FvsR batch mode.
        Args:
            key: The key containing the message.
            num_segments: The number of iterations.
            start_trigger: Set trigger during the start phase.
            msg_trigger: Set trigger during the message pushing phase.
            process_trigger: Set trigger during the processing phase.
            finish_trigger: Set trigger during the finish phase.
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
        # Trigger configuration.
        time.sleep(0.05)
        trigger_data = {"start_trigger": start_trigger, "msg_trigger": msg_trigger,
                        "process_trigger": process_trigger, "finish_trigger": finish_trigger}
        self.target.write(json.dumps(trigger_data).encode("ascii"))

    def random_batch(self, num_segments: int, start_trigger: bool,
                     msg_trigger: bool, process_trigger: bool, finish_trigger: bool):
        """ Start num_segments HMAC operations in random batch mode.
        Args:
            num_segments: The number of iterations.
            start_trigger: Set trigger during the start phase.
            msg_trigger: Set trigger during the message pushing phase.
            process_trigger: Set trigger during the processing phase.
            finish_trigger: Set trigger during the finish phase.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # BatchRandom command.
        self.target.write(json.dumps("BatchRandom").encode("ascii"))
        # Number of iterations payload.
        time.sleep(0.01)
        num_it_data = {"num_iterations": num_segments}
        self.target.write(json.dumps(num_it_data).encode("ascii"))
        # Trigger configuration.
        time.sleep(0.05)
        trigger_data = {"start_trigger": start_trigger, "msg_trigger": msg_trigger,
                        "process_trigger": process_trigger, "finish_trigger": finish_trigger}
        self.target.write(json.dumps(trigger_data).encode("ascii"))

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

    def read_response(self, max_tries: Optional[int] = 1) -> str:
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
