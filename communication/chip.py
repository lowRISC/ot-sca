# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""OpenTitan chip utility functions."""
import time
from subprocess import PIPE, Popen
from typing import Optional

# Command to flash the target given the location of the opentitantool and the firmware
# For example, opentitantool = "/path/to/opentitan/bazel-bin/sw/host/opentitantool/opentitantool"
# Firmware is the pre-compiled and signed binary
def flash_target(opentitantool, firmware):
    flash_process = Popen([opentitantool,
                        "--rcfile=",
                        "--interface=hyper310",
                        "--exec", "transport init",
                        "--exec", "bootstrap " + firmware, "no-op"])
    flash_process.communicate()
    rc = flash_process.returncode
    if rc != 0:
        raise RuntimeError('Error: Failed to flash chip.')
        return 0
    else:
        # Wait until chip finished booting.
        time.sleep(2)
        print(f'Info: Chip flashed with {firmware}.')
        return 1

# Command to reset the target given the location of the opentitantool
# For example, opentitantool = "/path/to/opentitan/bazel-bin/sw/host/opentitantool/opentitantool"
def reset_target(opentitantool, reset_delay = 0.005):
    """Reset OpenTitan by triggering the reset pin using opentitantool."""
    reset_process = Popen([opentitantool,
                        "--rcfile=",
                        "--interface=hyper310",
                        "--exec", "transport init",
                        "--exec", "gpio write RESET false",
                        "--exec", "gpio write RESET true", "no-op"])
    reset_process.communicate()
    rc = reset_process.returncode
    if rc != 0:
        raise RuntimeError('Error: Failed to reset chip.')
    else:
        time.sleep(reset_delay)

