# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from target.communication.fi_rom_commands import OTFIRom
from target.chip import Chip
from target.dut import DUT


def char_rom_read(opentitantool, iterations):
    target = DUT()
    chip = Chip(opentitantool)
    chip.reset_target()
    # Clear the output from the reset
    target.dump_all()

    romfi = OTFIRom(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        romfi.init()
    )
    for _ in range(iterations):
        romfi.handle_rom_read()
        response = target.read_response()
    return response
