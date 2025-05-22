# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from target.communication.fi_rng_commands import OTFIRng
from target.chip import Chip
from target.dut import DUT


def char_csrng_bias(opentitantool, iterations, trigger):
    target = DUT()
    chip = Chip(opentitantool)
    chip.reset_target()
    # Clear the output from the reset
    target.dump_all()

    rngfi = OTFIRng(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        rngfi.init("char_csrng_bias")
    )
    for _ in range(iterations):
        rngfi.rng_csrng_bias(trigger)
        response = target.read_response()
    return response


def char_edn_resp_ack(opentitantool, iterations):
    target = DUT()
    chip = Chip(opentitantool)
    chip.reset_target()
    # Clear the output from the reset
    target.dump_all()

    rngfi = OTFIRng(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        rngfi.init("char_edn_resp_ack")
    )
    for _ in range(iterations):
        rngfi.rng_edn_resp_ack()
        response = target.read_response()
    return response


def char_edn_bias(opentitantool, iterations):
    target = DUT()
    chip = Chip(opentitantool)
    chip.reset_target()
    # Clear the output from the reset
    target.dump_all()

    rngfi = OTFIRng(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        rngfi.init("char_edn_bias")
    )
    for _ in range(iterations):
        rngfi.rng_edn_bias()
        response = target.read_response()
    return response


def char_entropy_bias(opentitantool, iterations):
    target = DUT()
    chip = Chip(opentitantool)
    chip.reset_target()
    # Clear the output from the reset
    target.dump_all()

    rngfi = OTFIRng(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
        rngfi.init("char_entropy_bias")
    )
    for _ in range(iterations):
        rngfi.rng_entropy_bias()
        response = target.read_response()
    return response


def char_fw_overwrite(opentitantool, iterations, disable_health_check):
    target = DUT()
    rngfi = OTFIRng(target)

    for _ in range(iterations):
        chip = Chip(opentitantool)
        chip.reset_target()
        # Clear the output from the reset
        target.dump_all()

        # Initialize our chip and catch its output
        device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = (
            rngfi.init("char_fw_overwrite")
        )
        rngfi.rng_fw_overwrite(disable_health_check)
        response = target.read_response()
    return response
