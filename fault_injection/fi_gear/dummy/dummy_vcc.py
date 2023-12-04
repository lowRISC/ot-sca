# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from fi_gear.utility import random_float_range

""" Template for VCC glitching gear.
Acts as a template for voltage glitching FI gear.
"""


class DummyVCC:
    def __init__(self, glitch_voltage_min: float, glitch_voltage_max: float,
                 glitch_voltage_step: float, glitch_width_min: float,
                 glitch_width_max: float, glitch_width_step: float,
                 trigger_delay_min: int, trigger_delay_max: int, trigger_step: int):
        self.glitch_voltage_min = glitch_voltage_min
        self.glitch_voltage_max = glitch_voltage_max
        self.glitch_voltage_step = glitch_voltage_step
        self.glitch_width_min = glitch_width_min
        self.glitch_width_max = glitch_width_max
        self.glitch_width_step = glitch_width_step
        self.trigger_delay_min = trigger_delay_min
        self.trigger_delay_max = trigger_delay_max
        self.trigger_step = trigger_step

    def arm_trigger(self, fault_parameters: dict) -> None:
        """ Arm the trigger.

        Args:
            A dict containing the FI parameters.
        """
        print(f"Arming DummyVCC trigger with glitch_voltage={fault_parameters['glitch_voltage']} glitch_width={fault_parameters['glitch_width']}, and trigger_delay={fault_parameters['trigger_delay']}")  # noqa: E501

    def generate_fi_parameters(self) -> dict:
        """ Generate random voltage glitch parameters within the provided
            limits.

        Returns:
            A dict containing the FI parameters.
        """
        parameters = {}
        parameters["glitch_voltage"] = random_float_range(self.glitch_voltage_min,
                                                          self.glitch_voltage_max,
                                                          self.glitch_voltage_step)
        parameters["glitch_width"] = random_float_range(self.glitch_width_min,
                                                        self.glitch_width_max,
                                                        self.glitch_width_step)
        parameters["trigger_delay"] = random_float_range(self.trigger_delay_min,
                                                         self.trigger_delay_max,
                                                         self.trigger_step)
        return parameters

    def reset(self) -> None:
        """ No reset is required for dummy VCC glitch gear.
        """
        pass
