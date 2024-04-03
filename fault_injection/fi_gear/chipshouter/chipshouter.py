# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import time

from chipshouter import ChipSHOUTER
from chipshover import ChipShover
from fi_gear.utility import random_float_range

""" EMFI with the ChipShouter and the ChipShover XZY table.
"""


class ChipShouterEMFI:
    def __init__(self, x_position_min: float, x_position_max: float,
                 x_position_step: float, y_position_min: float, y_position_max: float,
                 y_position_step: float, z_position: float, voltage_min: float,
                 voltage_max: float, voltage_step: float, pulse_width_min: float,
                 pulse_width_max: float, pulse_width_step: float,
                 trigger_delay_min: float, trigger_delay_max: float,
                 trigger_step: float, num_iterations: float,
                 parameter_generation: str, chipshover_port: str,
                 chipshouter_port: str):
        self.x_position_min = x_position_min
        self.x_position_max = x_position_max
        self.x_position_step = x_position_step
        self.y_position_min = y_position_min
        self.y_position_max = y_position_max
        self.y_position_step = y_position_step
        self.z_position = z_position
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.voltage_step = voltage_step
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

        # Init ChipShover and set XZY to home position.
        self.shv = ChipShover(chipshover_port)
        print("Putting XZY table to home position.")
        self.shv.home()

        # Init ChipShouter.
        self.cs = ChipSHOUTER(chipshouter_port)
        # Active high trigger with high impedance.
        self.cs.hwtrig_mode = True
        self.cs.hwtrig_term = False
        # Mute the ChipShouter.
        self.cs.mute = True

    def arm_trigger(self, fault_parameters: dict) -> None:
        """ Arm the trigger.

        Args:
            A dict containing the FI parameters.
        """
        # Move ChipShouter to XZY position.
        self.shv.move(fault_parameters['x_position'],
                      fault_parameters['y_position'], self.z_position)
        # Set the EMFI parameter.
        self.cs.voltage = fault_parameters["voltage"]
        self.cs.pulse.width = fault_parameters["pulse_width"]
        self.cs.pulse.repeat = 1
        # Arm the trigger
        self.cs.armed = True
        # Wait and disarm to prepare for the next EMFI.
        print(self.cs.faults_current)
        time.sleep(1)
        self.cs.armed = False

    def generate_fi_parameters(self) -> dict:
        """ Generate EMFI parameters within the provided limits.

        Returns:
            A dict containing the FI parameters.
        """
        parameters = {}
        if self.parameter_generation == "random":
            parameters["x_position"] = random_float_range(self.x_position_min,
                                                          self.x_position_max,
                                                          self.x_position_step)
            parameters["y_position"] = random_float_range(self.y_position_min,
                                                          self.y_position_max,
                                                          self.y_position_step)
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
            raise Exception("ChipShouter EMFI only supports random/deterministic"
                            "parameter generation")

        parameters["voltage"] = random_float_range(self.voltage_min,
                                                   self.voltage_max,
                                                   self.voltage_step)
        parameters["pulse_width"] = random_float_range(self.pulse_width_min,
                                                       self.pulse_width_max,
                                                       self.pulse_width_step)
        parameters["trigger_delay"] = random_float_range(self.trigger_delay_min,
                                                         self.trigger_delay_max,
                                                         self.trigger_step)
        return parameters

    def reset(self) -> None:
        """ Reset XZY position.
        """
        self.shv.stop()
        self.shv.home()

    def get_num_fault_injections(self) -> int:
        """ Get number of fault injections.

        Returns: The total number of fault injections performed with ChipShouter EMFI.
        """
        if self.parameter_generation == "random":
            return self.num_iterations
        elif self.parameter_generation == "deterministic":
            return ((self.x_position_max - self.x_position_min + 1) /
                    self.x_position_step) * ((self.y_position_max - self.y_position_min + 1) /
                                             self.y_position_step) * (self.num_iterations)
        else:
            raise Exception("ChipShouter EMFI only supports random/deterministic"
                            "parameter generation")
