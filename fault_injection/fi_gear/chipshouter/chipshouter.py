# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
""" EMFI with the ChipShouter and the ChipShover XZY table.
"""

import sys
import time

import chipwhisperer as cw
from fi_gear.utility import random_float_range

# We allow to skip this loading if not used.
try:
    from chipshover import ChipShover
except ImportError:
    print("ChipShover library not found. Skipping its use.")
try:
    from chipshouter import ChipSHOUTER
    from chipshouter.com_tools import Reset_Exception
except ImportError:
    print("ChipShouter library not found. Skipping its use.")

sys.path.append("../")
from util import check_version  # noqa: E402


class ChipShouterEMFI:

    def __init__(
        self,
        x_position_min: float,
        x_position_max: float,
        x_position_step: float,
        y_position_min: float,
        y_position_max: float,
        y_position_step: float,
        z_position: float,
        voltage_min: float,
        voltage_max: float,
        voltage_step: float,
        pulse_width_min: float,
        pulse_width_max: float,
        pulse_width_step: float,
        trigger_delay_min: float,
        trigger_delay_max: float,
        trigger_step: float,
        num_iterations: float,
        parameter_generation: str,
        chipshover_port: str,
        chipshouter_port: str,
        husky_sn: str,
    ):
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
        self.husky_sn = husky_sn

        # Current x & y position determined by the FI parameter generation
        # function.
        self.curr_x_position = self.x_position_min
        self.curr_y_position = self.y_position_min
        # Current x & y position the XYZ table is programmed to.
        self.pos_y = self.y_position_min
        self.pos_x = self.x_position_min
        # When in deterministic mode, increment or decrement the Y position.
        self.increment = True

        # Current FI parameter iteration.
        self.curr_iteration = 0

        # Init ChipShover and set XZY to home position.
        self.shv = ChipShover(chipshover_port)
        print("Putting XZY table to home position.")
        self.shv.home()
        print("Putting XZY table to init position.")
        self.shv.move(self.x_position_min, self.y_position_min,
                      self.z_position)

        # Init ChipShouter.
        self.cs = ChipSHOUTER(chipshouter_port)
        # Init the ChipShouter.
        self.init_cs()

        # Init ChipWhisperer Husky.
        self.cwh = None
        if self.husky_sn is not None:
            check_version.check_husky("1.5.0", sn=str(self.husky_sn))
            self.cwh = cw.scope(sn=str(self.husky_sn))
        else:
            check_version.check_husky("1.5.0")
            self.cwh = cw.scope()
        if not self.cwh._is_husky:
            raise RuntimeError("No ChipWhisperer Husky is attached!")
        # Configure the Husky to generate the pulse that is fed into the ChipShouter.
        self.cwh.clock.clkgen_src = "system"
        self.cwh.clock.clkgen_freq = 50e6
        self.cwh.clock.adc_mul = 1
        self.cwh.adc.basic_mode = "rising_edge"
        self.cwh.io.hs2 = "glitch"
        self.cwh.glitch.enabled = True
        self.cwh.glitch.clk_src = "pll"
        self.cwh.clock.pll.update_fpga_vco(600e6)
        # Each glitch is one clock cycle long (e.g., 1/50MHz = 20ns)
        # Use repeat to have repeat * 20ns glitch lengths and ext_offset to
        # have a delay of ext_offset * 20ns trigger delay.
        self.cwh.glitch.output = "enable_only"
        # Route the pulse to the HS2 (Trigger/Glitch out SMB) output.
        self.cwh.io.aux_io_mcx = "hs2"
        self.cwh.io.glitch_trig_mcx = "glitch"
        # Trigger by using the IO4 input pin that comes from the DUT.
        self.cwh.trigger.triggers = "tio4"
        self.cwh.glitch.trigger_src = "ext_single"
        assert self.cwh.glitch.mmcm_locked

        # Check pulse width min.
        min_ns = int((1 / self.cwh.clock.clkgen_freq) * 1e9)
        print(min_ns)
        if self.pulse_width_min < min_ns:
            raise RuntimeError("Min pulse width shorter than supported (" +
                               str(min_ns) + " ns)")
        # Check pulse step.
        if (self.pulse_width_min
                != self.pulse_width_max) and (self.pulse_width_step % min_ns
                                              != 0):
            raise RuntimeError("Only a pulse step width of " + str(min_ns) +
                               " ns is supported")

        # Check trigger delay min.
        if self.trigger_delay_min < min_ns:
            raise RuntimeError("Min trigger delay shorter than supported (" +
                               str(min_ns) + " ns)")

        # Check trigger delay step.
        if (self.trigger_delay_min
                != self.trigger_delay_max) and (self.trigger_step % min_ns
                                                != 0):
            raise RuntimeError("Only a trigger delay step  of " + str(min_ns) +
                               " ns is supported")

    def init_cs(self) -> None:
        time.sleep(1)
        self.cs.reset = True
        time.sleep(5)
        # Mute the ChipShouter.
        self.cs.mute = True
        # Clear errors.
        self.cs.faults_current = 0
        # Set active high trigger with 50R impedance.
        self.cs.hwtrig_term = True
        self.cs.hwtrig_mode = True
        # Set default voltage to 250.
        self.cs.voltage = 250

        # Arm and add a small delay before continuing.
        self.cs.armed = 1
        time.sleep(1)

    def arm_trigger(self, fault_parameters: dict) -> None:
        """Arm the trigger.

        Args:
            A dict containing the FI parameters.
        """
        # Signal the CS that it is now safe to do a self-check.
        self.cs.trigger_safe

        # Check for CS errors.
        if self.cs.faults_current:
            print("ChipShouter fault occured, reboot and reconfigure it!")
            print(self.cs.faults_current)
            self.init_cs()

        # Move ChipShouter to XZY position.
        if (self.pos_y != fault_parameters["y_pos"] or
                self.pos_x != fault_parameters["x_pos"]):
            self.pos_y = fault_parameters["y_pos"]
            self.pos_x = fault_parameters["x_pos"]
            self.shv.move(fault_parameters["x_pos"], fault_parameters["y_pos"],
                          self.z_position)

        # Set the EMFI voltage parameter.
        try:
            self.cs.voltage = fault_parameters["glitch_voltage"]
        except Reset_Exception:
            print("ChipShouter fault occured, reboot and reconfigure it!")
            self.init_cs()

        # Configure the glitch delay and length.
        self.cwh.glitch.ext_offset = int(fault_parameters["trigger_delay"] *
                                         1e-9 /
                                         (1 / self.cwh.clock.clkgen_freq))
        self.cwh.glitch.repeat = int(fault_parameters["glitch_width"] * 1e-9 /
                                     (1 / self.cwh.clock.clkgen_freq))

        # Arm the ChipWhisperer Husky.
        self.cwh.arm()
        # The Husky now waits for the trigger signal of the DUT, sends the
        # configured pulse after trigger_delay ns to the ChipShouter, which then
        # uses this pulse as a template to generate the EM pulse.

    def generate_fi_parameters(self) -> dict:
        """Generate EMFI parameters within the provided limits.

        Returns:
            A dict containing the FI parameters.
        """
        parameters = {}
        if self.parameter_generation == "random":
            parameters["x_pos"] = random_float_range(self.x_position_min,
                                                     self.x_position_max,
                                                     self.x_position_step)
            parameters["y_pos"] = random_float_range(self.y_position_min,
                                                     self.y_position_max,
                                                     self.y_position_step)
        elif self.parameter_generation == "deterministic":
            if self.curr_iteration == self.num_iterations:
                self.curr_iteration = 0
                if self.curr_y_position >= self.y_position_max:
                    self.increment = False
                    self.curr_x_position += self.x_position_step
                elif self.curr_y_position <= self.y_position_min:
                    self.increment = True

                if self.increment:
                    self.curr_y_position += self.y_position_step
                else:
                    self.curr_y_position -= self.y_position_step

            parameters["x_pos"] = self.curr_x_position
            parameters["y_pos"] = self.curr_y_position
            self.curr_iteration += 1
        else:
            raise Exception(
                "ChipShouter EMFI only supports random/deterministic"
                "parameter generation")

        parameters["glitch_voltage"] = random_float_range(
            self.voltage_min, self.voltage_max, self.voltage_step)
        parameters["glitch_width"] = random_float_range(
            self.pulse_width_min, self.pulse_width_max, self.pulse_width_step)
        parameters["trigger_delay"] = random_float_range(
            self.trigger_delay_min, self.trigger_delay_max, self.trigger_step)
        return parameters

    def reset(self) -> None:
        """Reset XZY position."""
        self.shv.stop()
        self.shv.home()
        self.init_cs()

    def get_num_fault_injections(self) -> int:
        """Get number of fault injections.

        Returns: The total number of fault injections performed with ChipShouter EMFI.
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
                "ChipShouter EMFI only supports random/deterministic"
                "parameter generation")
