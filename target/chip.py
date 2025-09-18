# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""OpenTitan chip utility functions."""
import time
from subprocess import Popen
from typing import Optional


class Chip:
    """Class for the discrete chip. Initializes OpenTitan with the provided
    firmware & provides helper functions.
    """

    def __init__(self,
                 opentitantool_path,
                 interface: Optional[str] = "hyper310"):
        self.opentitantool = opentitantool_path
        self.interface = interface

    # Command to flash the target given the location of the opentitantool and the firmware
    # For example, opentitantool =
    # "/path/to/opentitan/bazel-bin/sw/host/opentitantool/opentitantool"
    # Firmware is the pre-compiled and signed binary
    def flash_target(self, firmware, boot_delay=4):
        flash_process = Popen([
            self.opentitantool,
            "--rcfile=",
            "--interface=" + self.interface,
            "--exec",
            "transport init",
            "--exec",
            "bootstrap " + firmware,
            "no-op",
        ])
        flash_process.communicate()
        rc = flash_process.returncode
        if rc != 0:
            raise RuntimeError("Error: Failed to flash chip.")
            return 0
        else:
            # Wait until chip finished booting.
            time.sleep(boot_delay)
            print(f"Info: Chip flashed with {firmware}.")
            return 1

    # Command to flash the target given the location of the opentitantool and the firmware
    # Firmware is the pre-compiled and signed binary
    # This uses the rescue protocol in order to flash the binary
    def flash_rescue_target(self, firmware, boot_delay=50):
        flash_process = Popen([
            self.opentitantool,
            "--rcfile=",
            "--interface=" + self.interface,
            "--exec",
            "transport init",
            "--exec",
            "rescue firmware " + firmware,
            "no-op",
        ])
        flash_process.communicate()
        rc = flash_process.returncode
        if rc != 0:
            raise RuntimeError("Error: Failed to flash OpenTitan chip.")
        else:
            # Wait until chip finished booting.
            time.sleep(boot_delay)
            print(f"Info: OpenTitan flashed with {firmware}.")

    # Command to reset the target given the location of the opentitantool
    # For example, opentitantool =
    # "/path/to/opentitan/bazel-bin/sw/host/opentitantool/opentitantool"
    def reset_target(self, reset_delay=0.005):
        """Reset OpenTitan by triggering the reset pin using opentitantool."""
        reset_process = Popen([
            self.opentitantool,
            "--rcfile=",
            "--interface=" + self.interface,
            "--exec",
            "transport init",
            "--exec",
            "gpio write RESET false",
            "--exec",
            "gpio write RESET true",
            "no-op",
        ])
        reset_process.communicate()
        rc = reset_process.returncode
        if rc != 0:
            raise RuntimeError("Error: Failed to reset chip.")
        else:
            time.sleep(reset_delay)
