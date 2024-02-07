# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the SHA3 SCA application on OpenTitan.

Communication with OpenTitan either happens over simpleserial or the uJson
command interface.
"""
import json
import time
from typing import Optional


class OTTRIGGER:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        self.simple_serial = True
        if protocol == "ujson":
            self.simple_serial = False

    def select_trigger(self, trigger_source: Optional[int] = 0):
        """ Select the trigger source for SCA.
        Args:
            trigger_source:
                            - 0: Precise, hardware-generated trigger - FPGA only.
                            - 1: Fully software-controlled trigger.
        """
        if self.simple_serial:
            self.target.write(cmd="t", data=bytearray([trigger_source]))
        else:
            # TODO(#256): without the delay, the device uJSON command handler program
            # does not recognize the commands.
            time.sleep(0.01)
            self.target.write(json.dumps("TriggerSca").encode("ascii"))
            # SelectTriggerSource command.
            self.target.write(json.dumps("SelectTriggerSource").encode("ascii"))
            # Source payload.
            time.sleep(0.01)
            src = {"source": trigger_source}
            self.target.write(json.dumps(src).encode("ascii"))
