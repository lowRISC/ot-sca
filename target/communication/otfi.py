"""Base class of communication interface for OpenTitan FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


class OTFI:
    IP = ["Ibex", "Otbn", "Crypto"]
    def __init__(self, target, ip) -> None:
        self.target = target
        self.ip = ip

        assert self.ip in OTFI.IP, "ip ({self.ip} not in OTFI.IP ({OTFI.IP})"

    def _ujson_fi_cmd(self) -> None:
        time.sleep(0.01)
        self.target.write(json.dumps(f"{self.ip}Fi").encode("ascii"))
        time.sleep(0.01)

    def init(self, test: Optional[str] = "") -> None:
        """ Initialize the FI code on the chip.
        Returns:
            The device ID of the device.
        """
        # Fi command.
        self._ujson_fi_cmd()
        # Init command.
        time.sleep(0.01)
        self.target.write(json.dumps("Init").encode("ascii"))
        # Read back device ID from device.
        return self.read_response(max_tries=30)

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
        """ Read response from Crypto FI framework.
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
