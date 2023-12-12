#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from fi_gear.dummy.dummy_clk import DummyCLK
from fi_gear.dummy.dummy_emfi import DummyEMFI
from fi_gear.dummy.dummy_lfi import DummyLFI
from fi_gear.dummy.dummy_vcc import DummyVCC
from fi_gear.husky.husky_vcc import HuskyVCC


class FIGear:
    """ Fault Injection Gear class.

    This class represents a FI gear (e.g. LFI, EMFI, ...).
    """
    def __init__(self, cfg: dict) -> None:
        self.gear_type = cfg["fisetup"]["fi_gear"]
        self.fi_type = cfg["fisetup"]["fi_type"]
        self.gear = None

        if self.gear_type == "dummy" and self.fi_type == "voltage_glitch":
            self.gear = DummyVCC(
                glitch_voltage_min = cfg["fisetup"]["glitch_voltage_min"],
                glitch_voltage_max = cfg["fisetup"]["glitch_voltage_max"],
                glitch_voltage_step = cfg["fisetup"]["glitch_voltage_step"],
                glitch_width_min = cfg["fisetup"]["glitch_width_min"],
                glitch_width_max = cfg["fisetup"]["glitch_width_max"],
                glitch_width_step = cfg["fisetup"]["glitch_width_step"],
                trigger_delay_min = cfg["fisetup"]["trigger_delay_min"],
                trigger_delay_max = cfg["fisetup"]["trigger_delay_max"],
                trigger_step = cfg["fisetup"]["trigger_step"],
                num_iterations = cfg["fisetup"]["num_iterations"],
                parameter_generation = cfg["fisetup"]["parameter_generation"])
        elif self.gear_type == "dummy" and self.fi_type == "clock_glitch":
            self.gear = DummyCLK(
                glitch_width_min = cfg["fisetup"]["glitch_width_min"],
                glitch_width_max = cfg["fisetup"]["glitch_width_max"],
                glitch_width_step = cfg["fisetup"]["glitch_width_step"],
                trigger_delay_min = cfg["fisetup"]["trigger_delay_min"],
                trigger_delay_max = cfg["fisetup"]["trigger_delay_max"],
                trigger_step = cfg["fisetup"]["trigger_step"],
                num_iterations = cfg["fisetup"]["num_iterations"],
                parameter_generation = cfg["fisetup"]["parameter_generation"])
        elif self.gear_type == "dummy" and self.fi_type == "lfi":
            self.gear = DummyLFI(
                x_position_min = cfg["fisetup"]["x_position_min"],
                x_position_max = cfg["fisetup"]["x_position_max"],
                x_position_step = cfg["fisetup"]["x_position_step"],
                y_position_min = cfg["fisetup"]["y_position_min"],
                y_position_max = cfg["fisetup"]["y_position_max"],
                y_position_step = cfg["fisetup"]["y_position_step"],
                attenuation_min = cfg["fisetup"]["attenuation_min"],
                attenuation_max = cfg["fisetup"]["attenuation_max"],
                attenuation_step = cfg["fisetup"]["attenuation_step"],
                pulse_width_min = cfg["fisetup"]["pulse_width_min"],
                pulse_width_max = cfg["fisetup"]["pulse_width_max"],
                pulse_width_step = cfg["fisetup"]["pulse_width_step"],
                trigger_delay_min = cfg["fisetup"]["trigger_delay_min"],
                trigger_delay_max = cfg["fisetup"]["trigger_delay_max"],
                trigger_step = cfg["fisetup"]["trigger_step"],
                num_iterations = cfg["fisetup"]["num_iterations"],
                parameter_generation = cfg["fisetup"]["parameter_generation"])
        elif self.gear_type == "dummy" and self.fi_type == "emfi":
            self.gear = DummyEMFI(
                x_position_min = cfg["fisetup"]["x_position_min"],
                x_position_max = cfg["fisetup"]["x_position_max"],
                x_position_step = cfg["fisetup"]["x_position_step"],
                y_position_min = cfg["fisetup"]["y_position_min"],
                y_position_max = cfg["fisetup"]["y_position_max"],
                y_position_step = cfg["fisetup"]["y_position_step"],
                voltage_min = cfg["fisetup"]["voltage_min"],
                voltage_max = cfg["fisetup"]["voltage_max"],
                voltage_step = cfg["fisetup"]["voltage_step"],
                pulse_width_min = cfg["fisetup"]["pulse_width_min"],
                pulse_width_max = cfg["fisetup"]["pulse_width_max"],
                pulse_width_step = cfg["fisetup"]["pulse_width_step"],
                trigger_delay_min = cfg["fisetup"]["trigger_delay_min"],
                trigger_delay_max = cfg["fisetup"]["trigger_delay_max"],
                trigger_step = cfg["fisetup"]["trigger_step"],
                num_iterations = cfg["fisetup"]["num_iterations"],
                parameter_generation = cfg["fisetup"]["parameter_generation"])
        elif self.gear_type == "husky" and self.fi_type == "voltage_glitch":
            self.gear = HuskyVCC(
                pll_frequency = cfg["target"]["pll_frequency"],
                glitch_width_min = cfg["fisetup"]["glitch_width_min"],
                glitch_width_max = cfg["fisetup"]["glitch_width_max"],
                glitch_width_step = cfg["fisetup"]["glitch_width_step"],
                trigger_delay_min = cfg["fisetup"]["trigger_delay_min"],
                trigger_delay_max = cfg["fisetup"]["trigger_delay_max"],
                trigger_step = cfg["fisetup"]["trigger_step"],
                num_iterations = cfg["fisetup"]["num_iterations"],
                parameter_generation = cfg["fisetup"]["parameter_generation"])

    def arm_trigger(self, fi_parameters: dict) -> None:
        """ Arm the trigger.

        Args:
            A dict containing the FI parameters.
        """
        self.gear.arm_trigger(fi_parameters)

    def generate_fi_parameters(self) -> dict:
        """ Generate random fault parameters within the provided limits.

        Returns:
            A dict containing the FI parameters.
        """
        return self.gear.generate_fi_parameters()

    def reset(self) -> None:
        """ Reset the FI gear.

        Some FI gear (Husky) needs to be resettet.
        """
        self.gear.reset()

    def get_num_fault_injections(self) -> int:
        """ Get number of fault injections.

        Returns: The total number of fault injections performed with the FI gear.
        """
        return self.gear.get_num_fault_injections()
