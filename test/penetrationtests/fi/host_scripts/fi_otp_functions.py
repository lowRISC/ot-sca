# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from target.communication.fi_otp_commands import OTFIOtp
from target.chip import Chip
from target.dut import DUT


def char_vendor_test(opentitantool, iterations):
    target = DUT()
    chip = Chip(opentitantool)
    chip.reset_target()
    # Clear the output from the reset
    target.dump_all()

    otpfi = OTFIOtp(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        otpfi.init()
    )
    for _ in range(iterations):
        otpfi.otp_fi_vendor_test()
        response = target.read_response()
    return response


def char_owner_sw_cfg(opentitantool, iterations):
    target = DUT()
    chip = Chip(opentitantool)
    chip.reset_target()
    # Clear the output from the reset
    target.dump_all()

    otpfi = OTFIOtp(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        otpfi.init()
    )
    for _ in range(iterations):
        otpfi.otp_fi_owner_sw_cfg()
        response = target.read_response()
    return response


def char_hw_cfg(opentitantool, iterations):
    target = DUT()
    chip = Chip(opentitantool)
    chip.reset_target()
    # Clear the output from the reset
    target.dump_all()

    otpfi = OTFIOtp(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        otpfi.init()
    )
    for _ in range(iterations):
        otpfi.otp_fi_hw_cfg()
        response = target.read_response()
    return response


def char_life_cycle(opentitantool, iterations):
    target = DUT()
    chip = Chip(opentitantool)
    chip.reset_target()
    # Clear the output from the reset
    target.dump_all()

    otpfi = OTFIOtp(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        otpfi.init()
    )
    for _ in range(iterations):
        otpfi.otp_fi_life_cycle()
        response = target.read_response()
    return response
