# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""OpenTitan chip utility functions."""
import time
from subprocess import PIPE, Popen
from typing import Optional


class Chip():
    """Class for the discrete chip. Initializes OpenTitan with the provided
    firmware & provides helper functions.
    """
    def __init__(self, firmware, opentitantool_path,
                 boot_delay: Optional[int] = 1):
        self.firmware = firmware
        self.opentitantool = opentitantool_path
        self.boot_delay = boot_delay
        self._initialize_chip()

    def _initialize_chip(self):
        """Initializes the chip."""
        # Flash the chip using the opentitantool with the provided firmware.
        flash_process = Popen([self.opentitantool, "--rcfile=",
                               "--interface=hyper310",
                               "--exec", "transport init",
                               "--exec", "bootstrap " + self.firmware, "no-op"],
                              stdout=PIPE, stderr=PIPE)
        flash_process.communicate()
        rc = flash_process.returncode
        if rc != 0:
            raise RuntimeError('Error: Failed to flash OpenTitan chip.')
        else:
            # Wait until chip finished booting.
            time.sleep(self.boot_delay)
            print(f'Info: OpenTitan flashed with {self.firmware}.')

    def reset_target(self, boot_delay: Optional[int] = 1):
        """Reset OpenTitan by triggering the reset pin using opentitantool."""
        reset_process = Popen([self.opentitantool, "--rcfile=",
                               "--interface=hyper310",
                               "--exec", "transport init",
                               "--exec", "gpio write RESET false",
                               "--exec", "gpio write RESET true", "no-op"],
                              stdout=PIPE, stderr=PIPE)
        reset_process.communicate()
        rc = reset_process.returncode
        if rc != 0:
            raise RuntimeError('Error: Failed to reset OpenTitan chip.')
        else:
            print("Info: Resetting OpenTitan.")
            # Wait until chip finished booting.
            time.sleep(boot_delay)
