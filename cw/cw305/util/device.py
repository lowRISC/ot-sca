# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

r"""CW305 utility functions. Used to configure FPGA with OpenTitan design."""

import inspect
import time

import chipwhisperer as cw


class RuntimePatchFPGAProgram:
    """Replaces the FPGAProgram method of an FPGA object with a function
    that calls the given callback before calling the original method.

    This class can be used to detect if the FPGA was actually programmed or not.
    """
    def __init__(self, fpga, callback):
        """Inits a RuntimePatchFPGAProgram.

        Args:
            fpga: An FPGA object.
            callback: Callback to call when fpga.FPGAProgram() is called.
        """
        self._fpga = fpga
        self._callback = callback
        self._orig_fn = fpga.FPGAProgram

    def __enter__(self):
        def wrapped_fn(*args, **kwargs):
            self._callback()
            return self._orig_fn(*args, **kwargs)
        self._fpga.FPGAProgram = wrapped_fn

    def __exit__(self, exc_type, exc_value, traceback):
        self._fpga.FPGAProgram = self._orig_fn


class OpenTitan(object):
    def __init__(self, fw_programmer, bitstream, pll_frequency, baudrate, scope_gain, num_samples):
        self.fpga = self.initialize_fpga(bitstream, pll_frequency)
        self.scope = self.initialize_scope(scope_gain, num_samples)
        self.target = self.initialize_target(self.scope, fw_programmer, baudrate)

    def initialize_fpga(self, bitstream, pll_frequency):
        """Initializes FPGA bitstream and sets PLL frequency."""
        # Do not program the FPGA if it is already programmed.
        # Note: Set this to True to force programming the FPGA when using a new
        # bitstream.
        # TODO: We should have this in the CLI.
        force_programming = False
        print('Connecting and loading FPGA... ', end='')
        fpga = cw.capture.targets.CW305()
        # Runtime patch fpga.fpga.FPGAProgram to detect if it was actually called.
        # Note: This is fragile and may break but it is easy to miss that the FPGA
        # was not programmed.
        programmed = False

        def program_callback():
            nonlocal programmed
            programmed = True

        with RuntimePatchFPGAProgram(fpga.fpga, program_callback):
            # Connect to the FPGA and program it.
            fpga.con(bsfile=bitstream, force=force_programming)
            if not programmed:
                # TODO: Update this message when we have this in the CLI.
                stack_top = inspect.stack()[0]
                print(f"SKIPPED! (see: {stack_top[1]}:{stack_top[3]})")
            else:
                print("Done!")
        fpga.vccint_set(1.0)

        print('Initializing PLL1')
        fpga.pll.pll_enable_set(True)
        fpga.pll.pll_outenable_set(False, 0)
        fpga.pll.pll_outenable_set(True, 1)
        fpga.pll.pll_outenable_set(False, 2)
        fpga.pll.pll_outfreq_set(pll_frequency, 1)

        # 1ms is plenty of idling time
        fpga.clkusbautooff = True
        fpga.clksleeptime = 1
        return fpga

    def initialize_scope(self, scope_gain, num_samples):
        """Initializes chipwhisperer scope."""
        scope = cw.scope()
        scope.gain.db = scope_gain
        scope.adc.samples = num_samples
        scope.adc.offset = 0
        scope.adc.basic_mode = "rising_edge"
        scope.clock.clkgen_freq = 100000000
        # We sample using the target clock (100 MHz).
        scope.clock.adc_src = "extclk_dir"
        scope.trigger.triggers = "tio4"
        scope.io.tio1 = "serial_tx"
        scope.io.tio2 = "serial_rx"
        scope.io.hs2 = "disabled"

        # TODO: Need to update error handling.
        scope.clock.reset_adc()
        time.sleep(0.5)
        assert (scope.clock.adc_locked), "ADC failed to lock"
        return scope

    def initialize_target(self, scope, fw_programmer, baudrate):
        """Loads firmware image and initializes test target."""
        fw_programmer.run(self.fpga)
        time.sleep(0.5)
        target = cw.target(scope)
        target.output_len = 16
        target.baud = baudrate
        target.flush()
        return target
