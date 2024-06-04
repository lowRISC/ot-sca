# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan RNG FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


from target.communication.otfi import OTFI
from target.communication.otfi_test import OTFITest


class OTFIRng(OTFI):
    TESTS = [
        OTFITest("csrng_bias"),
        OTFITest("edn_bus_ack"),
    ]

    def __init__(self, target) -> None:
        super().__init__(target, "Rng")

    def init(self, test: Optional[str] = "") -> None:
        """ Initialize the RNG FI code on the chip.

        Args:
            test: The selected test.
        Returns:
            The device ID of the device.
        """
        # RngFi command.
        self._ujson_fi_cmd()
        # Init command.
        time.sleep(0.01)
        if "csrng" in test:
            self.target.write(json.dumps("CsrngInit").encode("ascii"))
        else:
            self.target.write(json.dumps("EdnInit").encode("ascii"))
        # Read back device ID from device.
        return self.read_response(max_tries=30)
