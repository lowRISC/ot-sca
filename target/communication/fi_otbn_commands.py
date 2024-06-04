# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for OpenTitan OTBN FI framework.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


from target.communication.otfi import OTFI
from target.communication.otfi_test import OTFITest


class OTFIOtbn(OTFI):
    TESTS = [
        OTFITest("char_unrolled_reg_op_loop"),
        OTFITest("char_unrolled_dmem_op_loop"),
        OTFITest("char_hardware_reg_op_loop"),
        OTFITest("char_hardware_dmem_op_loop"),
        OTFITest("key_sideload"),
        OTFITest("load_integrity"),
    ]

    def __init__(self, target) -> None:
        super().__init__(target, "Otbn")

    def init(self, test: Optional[str] = "") -> None:
        self.init_keymgr(test)
        super().init(test)

    def init_keymgr(self, test: str) -> None:
        """ Initialize the key manager on the chip.
        Args:
            test: Name of the test. Used to determine if key manager init is
                  needed.
        """
        if "key_sideload" in test:
            # OtbnFi command.
            self._ujson_fi_cmd()
            # InitTrigger command.
            time.sleep(0.01)
            self.target.write(json.dumps("InitKeyMgr").encode("ascii"))
            # As the init resets the chip, we need to call it again to complete
            # the initialization of the key manager.
            time.sleep(2)
            self._ujson_fi_cmd()
            time.sleep(0.01)
            self.target.write(json.dumps("InitKeyMgr").encode("ascii"))
            time.sleep(2)
