# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
""" Template for LFI gear.
Acts as a template for laser FI gear with XY table.
"""

from fi_gear.utility import random_float_range


class DummyLFI:

    def __init__(
        self,
        x_position_min: int,
        x_position_max: int,
        x_position_step: int,
        y_position_min: int,
        y_position_max: int,
        y_position_step: int,
        attenuation_min: int,
        attenuation_max: int,
        attenuation_step: int,
        pulse_width_min: int,
        pulse_width_max: int,
        pulse_width_step: int,
        trigger_delay_min: int,
        trigger_delay_max: int,
        trigger_step: int,
        num_iterations: int,
        parameter_generation: str,
    ):
        self.x_position_min = x_position_min
        self.x_position_max = x_position_max
        self.x_position_step = x_position_step
        self.y_position_min = y_position_min
        self.y_position_max = y_position_max
        self.y_position_step = y_position_step
        self.attenuation_min = attenuation_min
        self.attenuation_max = attenuation_max
        self.attenuation_step = attenuation_step
        self.pulse_width_min = pulse_width_min
        self.pulse_width_max = pulse_width_max
        self.pulse_width_step = pulse_width_step
        self.trigger_delay_min = trigger_delay_min
        self.trigger_delay_max = trigger_delay_max
        self.trigger_step = trigger_step
        self.num_iterations = num_iterations
        self.parameter_generation = parameter_generation

        # Current x & y position.
        self.curr_x_position = self.x_position_min
        self.curr_y_position = self.y_position_min

        # Current FI parameter iteration.
        self.curr_iteration = 0

    def arm_trigger(self, fault_parameters: dict) -> None:
        """Arm the trigger.

        Args:
            A dict containing the FI parameters.
        """
        print(f"Arming DummyLFI trigger with"
              f"x_position={fault_parameters['x_position']} "
              f"y_position={fault_parameters['y_position']}, "
              f"attenuation={fault_parameters['attenuation']}, "
              f"pulse_width={fault_parameters['pulse_width']}, and "
              f"trigger_delay={fault_parameters['trigger_delay']}")

    def generate_fi_parameters(self) -> dict:
        """Generate LFI parameters within the provided limits.

        Returns:
            A dict containing the FI parameters.
        """
        parameters = {}
        if self.parameter_generation == "random":
            parameters["x_position"] = random_float_range(
                self.x_position_min, self.x_position_max, self.x_position_step)
            parameters["y_position"] = random_float_range(
                self.y_position_min, self.y_position_max, self.y_position_step)
        elif self.parameter_generation == "deterministic":
            if self.curr_iteration == self.num_iterations:
                self.curr_iteration = 0
                if self.curr_y_position < self.y_position_max:
                    self.curr_y_position += self.y_position_step
                else:
                    self.curr_y_position = self.y_position_min
                    self.curr_x_position += self.x_position_step

            parameters["x_position"] = self.curr_x_position
            parameters["y_position"] = self.curr_y_position
            self.curr_iteration += 1
        else:
            raise Exception(
                "DummyLFI only supports random/deterministic parameter generation"
            )

        parameters["attenuation"] = random_float_range(self.attenuation_min,
                                                       self.attenuation_max,
                                                       self.attenuation_step)
        parameters["pulse_width"] = random_float_range(self.pulse_width_min,
                                                       self.pulse_width_max,
                                                       self.pulse_width_step)
        parameters["trigger_delay"] = random_float_range(
            self.trigger_delay_min, self.trigger_delay_max, self.trigger_step)
        return parameters

    def reset(self) -> None:
        """No reset is required for dummy LFI gear."""
        pass

    def get_num_fault_injections(self) -> int:
        """Get number of fault injections.

        Returns: The total number of fault injections performed with DummyEMFI.
        """
        if self.parameter_generation == "random":
            return self.num_iterations
        elif self.parameter_generation == "deterministic":
            return (((self.x_position_max - self.x_position_min + 1) /
                     self.x_position_step) *
                    ((self.y_position_max - self.y_position_min + 1) /
                     self.y_position_step) * (self.num_iterations))
        else:
            raise Exception(
                "DummyEMFI only supports random/deterministic parameter generation"
            )
