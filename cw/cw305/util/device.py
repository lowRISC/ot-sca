# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

r"""CW305 utility functions. Used to configure FPGA with OpenTitan design."""

import inspect
import time
import re

import chipwhisperer as cw
from util.spiflash import SpiProgrammer


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
    def __init__(self, bitstream, firmware, pll_frequency, baudrate, scope_gain,
                 num_samples, offset, output_len):

        # Extract target board type from bitstream name.
        m = re.search('cw305|cw310', bitstream)
        if m:
            if m.group() == 'cw305':
                fpga = cw.capture.targets.CW305()
            else:
                assert m.group() == 'cw310'
                fpga = cw.capture.targets.CW310()
            programmer = SpiProgrammer(fpga)
        else:
            raise ValueError('Could not infer target board type from bistream name')

        self.fpga = self.initialize_fpga(fpga, bitstream, pll_frequency)
        self.scope = self.initialize_scope(scope_gain, num_samples, offset)
        self.target = self.initialize_target(programmer, firmware, baudrate, output_len)

    def initialize_fpga(self, fpga, bitstream, pll_frequency):
        """Initializes FPGA bitstream and sets PLL frequency."""
        # Do not program the FPGA if it is already programmed.
        # Note: Set this to True to force programming the FPGA when using a new
        # bitstream.
        # Unfortunately, this doesn't seem to work for the CW310 yet,
        # see https://github.com/lowRISC/ot-sca/issues/48.
        # TODO: We should have this in the CLI.
        force_programming = False
        print('Connecting and loading FPGA... ', end='')

        # Runtime patch fpga.fpga.FPGAProgram to detect if it was actually called.
        # Note: This is fragile and may break but it is easy to miss that the FPGA
        # was not programmed.
        programmed = False

        def program_callback():
            nonlocal programmed
            programmed = True

        with RuntimePatchFPGAProgram(fpga.fpga, program_callback):
            # Connect to the FPGA and program it.
            fpga.con(bsfile=bitstream, force=force_programming, slurp=False)
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

        # Disable USB clock to reduce noise in power traces.
        fpga.clkusbautooff = True

        # 1ms is plenty of idling time
        fpga.clksleeptime = 1

        return fpga

    def initialize_scope(self, scope_gain, num_samples, offset):
        """Initializes chipwhisperer scope."""
        scope = cw.scope()
        scope.gain.db = scope_gain
        scope.adc.basic_mode = "rising_edge"
        if hasattr(scope, '_is_husky') and scope._is_husky:
            # We sample using the target clock * 2 (200 MHz).
            scope.clock.adc_mul = 2
            scope.clock.clkgen_freq = 100000000
            scope.adc.samples = num_samples
            scope.clock.clkgen_src = 'extclk'
            husky = True
        else:
            # We sample using the target clock (100 MHz).
            scope.clock.adc_mul = 1
            scope.clock.clkgen_freq = 100000000
            scope.adc.samples = num_samples // 2
            offset = offset // 2
            scope.clock.adc_src = 'extclk_dir'
            husky = False
        if offset >= 0:
            scope.adc.offset = offset
        else:
            scope.adc.offset = 0
            scope.adc.presamples = -offset
        scope.trigger.triggers = "tio4"
        scope.io.tio1 = "serial_tx"
        scope.io.tio2 = "serial_rx"
        scope.io.hs2 = "disabled"

        # TODO: Need to update error handling.
        if not husky:
            scope.clock.reset_adc()
            time.sleep(0.5)
        assert (scope.clock.adc_locked), "ADC failed to lock"
        return scope

    def initialize_target(self, programmer, firmware, baudrate, output_len):
        """Loads firmware image and initializes test target."""
        programmer.bootstrap(firmware)
        time.sleep(0.5)
        target = cw.target(self.scope)
        target.output_len = output_len
        target.baud = baudrate
        target.flush()
        return target
