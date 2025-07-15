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

    def init(self) -> list:
        """ Initializes HMAC on the target.

        Returns:
            Device id
            The owner info page
            The boot log
            The boot measurements
            The testOS version
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # Init command.
        self.target.write(json.dumps("Init").encode("ascii"))
        time.sleep(0.01)
        parameters = {"enable_icache": True, "enable_dummy_instr": True, "dummy_instr_count": 3, "enable_jittery_clock": True, "enable_sram_readback": True}
        self.target.write(json.dumps(parameters).encode("ascii"))
        parameters = {"sensor_ctrl_enable": True, "sensor_ctrl_en_fatal": [False, False, False, False, False, False, False, False, False, False, False]}
        self.target.write(json.dumps(parameters).encode("ascii"))
        device_id = self.target.read_response()
        owner_page = self.target.read_response()
        boot_log = self.target.read_response()
        boot_measurements = self.target.read_response()
        version = self.target.read_response()
        return device_id, owner_page, boot_log, boot_measurements, version

    def single(self, msg: list[int], key: list[int], trigger: int):
        """ Start a single HMAC operation using the given message and key.
        Args:
            msg: The list containing the message.
            key: The key containing the message.
            trigger: Which trigger to raise.
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
        # Trigger payload.
        time.sleep(0.01)
        if trigger == 0:
            mode = {"start_trigger": True, "msg_trigger": False,
                    "process_trigger": False, "finish_trigger": False}
        elif trigger == 1:
            mode = {"start_trigger": False, "msg_trigger": True,
                    "process_trigger": False, "finish_trigger": False}
        elif trigger == 2:
            mode = {"start_trigger": False, "msg_trigger": False,
                    "process_trigger": True, "finish_trigger": False}
        elif trigger == 3:
            mode = {"start_trigger": False, "msg_trigger": False,
                    "process_trigger": False, "finish_trigger": True}
        self.target.write(json.dumps(mode).encode("ascii"))

    def fvsr_batch(self, key: list[int], num_segments: int, trigger: int):
        """ Start num_segments HMAC operation in FvsR batch mode.
        Args:
            key: The key containing the message.
            num_segments: The number of iterations.
            trigger: Which trigger to raise.
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
        # Trigger payload.
        time.sleep(0.01)
        if trigger == 0:
            mode = {"start_trigger": True, "msg_trigger": False,
                    "process_trigger": False, "finish_trigger": False}
        elif trigger == 1:
            mode = {"start_trigger": False, "msg_trigger": True,
                    "process_trigger": False, "finish_trigger": False}
        elif trigger == 2:
            mode = {"start_trigger": False, "msg_trigger": False,
                    "process_trigger": True, "finish_trigger": False}
        elif trigger == 3:
            mode = {"start_trigger": False, "msg_trigger": False,
                    "process_trigger": False, "finish_trigger": True}
        self.target.write(json.dumps(mode).encode("ascii"))

    def random_batch(self, num_segments: int, trigger: int):
        """ Start num_segments HMAC operations in random batch mode.
        Args:
            num_segments: The number of iterations.
            trigger: Which trigger to raise.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # BatchRandom command.
        self.target.write(json.dumps("BatchRandom").encode("ascii"))
        # Number of iterations payload.
        time.sleep(0.01)
        num_it_data = {"num_iterations": num_segments}
        self.target.write(json.dumps(num_it_data).encode("ascii"))
        # Trigger payload.
        time.sleep(0.01)
        if trigger == 0:
            mode = {"start_trigger": True, "msg_trigger": False,
                    "process_trigger": False, "finish_trigger": False}
        elif trigger == 1:
            mode = {"start_trigger": False, "msg_trigger": True,
                    "process_trigger": False, "finish_trigger": False}
        elif trigger == 2:
            mode = {"start_trigger": False, "msg_trigger": False,
                    "process_trigger": True, "finish_trigger": False}
        elif trigger == 3:
            mode = {"start_trigger": False, "msg_trigger": False,
                    "process_trigger": False, "finish_trigger": True}
        self.target.write(json.dumps(mode).encode("ascii"))

    def daisy_chain(self, text: list[int], key: list[int], num_segments: int, trigger: int):
        """ Start num_segments HMAC operations in daisy chain mode.
        Args:
            text: The input message
            key: The HMAC key
            num_segments: The number of iterations.
            trigger: Which trigger to raise.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # BatchRandom command.
        self.target.write(json.dumps("BatchDaisy").encode("ascii"))
        # Number of iterations payload.
        time.sleep(0.01)
        key_data = {"key": key}
        self.target.write(json.dumps(key_data).encode("ascii"))
        message_data = {"message": text}
        self.target.write(json.dumps(message_data).encode("ascii"))
        time.sleep(0.05)
        num_it_data = {"num_iterations": num_segments}
        self.target.write(json.dumps(num_it_data).encode("ascii"))
        # Trigger payload.
        time.sleep(0.01)
        if trigger == 0:
            mode = {"start_trigger": True, "msg_trigger": False,
                    "process_trigger": False, "finish_trigger": False}
        elif trigger == 1:
            mode = {"start_trigger": False, "msg_trigger": True,
                    "process_trigger": False, "finish_trigger": False}
        elif trigger == 2:
            mode = {"start_trigger": False, "msg_trigger": False,
                    "process_trigger": True, "finish_trigger": False}
        elif trigger == 3:
            mode = {"start_trigger": False, "msg_trigger": False,
                    "process_trigger": False, "finish_trigger": True}
        self.target.write(json.dumps(mode).encode("ascii"))
        
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
