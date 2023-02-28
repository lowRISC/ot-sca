# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
r"""CW305 utility functions. Used to configure FPGA with OpenTitan design."""

import inspect
import re
import time

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

    def __init__(self, bitstream, firmware, pll_frequency, baudrate,
                 scope_gain, num_samples, offset, output_len):

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
            raise ValueError(
                'Could not infer target board type from bistream name')

        # Added `pll_frequency` to handle frequencies other than 100MHz.
        # Needed this for OTBN ECDSA.
        # TODO: Remove these comments after discussion
        self.fpga = self.initialize_fpga(fpga, bitstream, pll_frequency)
        self.scope = self.initialize_scope(scope_gain, num_samples, offset,
                                           pll_frequency)
        self.target = self.initialize_target(programmer, firmware, baudrate,
                                             output_len, pll_frequency)

    def initialize_fpga(self, fpga, bitstream, pll_frequency):
        """Initializes FPGA bitstream and sets PLL frequency."""
        # Do not program the FPGA if it is already programmed.
        # Note: Set this to True to force programming the FPGA when using a new
        # bitstream.
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
        # Added `pll_frequency` to handle frequencies other than 100MHz.
        # Needed this for OTBN ECDSA.
        # TODO: Remove these comments after discussion
        fpga.pll.pll_outfreq_set(pll_frequency, 1)

        # Disable USB clock to reduce noise in power traces.
        fpga.clkusbautooff = True

        # 1ms is plenty of idling time
        fpga.clksleeptime = 1

        return fpga

    def initialize_scope(self, scope_gain, num_samples, offset, pll_frequency):
        """Initializes chipwhisperer scope."""
        scope = cw.scope()
        scope.gain.db = scope_gain
        scope.adc.basic_mode = "rising_edge"
        if hasattr(scope, '_is_husky') and scope._is_husky:
            # We sample using the target clock * 2 (200 MHz).
            scope.clock.clkgen_src = 'extclk'
            # To fully capture the long OTBN applications,
            # we may need to use pll_frequencies other than 100 MHz.
            scope.clock.clkgen_freq = pll_frequency
            scope.clock.adc_mul = 2
            scope.clock.extclk_monitor_enabled = False
            scope.adc.samples = num_samples

            husky = True
            print(f"Husky? = {husky}")
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

        # Wait for ADC to lock.
        ping_cnt = 0
        while not scope.clock.adc_locked:
            if ping_cnt == 3:
                raise RuntimeError(
                    f'ADC failed to lock (attempts: {ping_cnt}).')
            ping_cnt += 1
            time.sleep(0.5)

        return scope

    def initialize_target(self, programmer, firmware, baudrate, output_len,
                          pll_frequency):
        """Loads firmware image and initializes test target."""
        # To fully capture the long OTBN applications,
        # we may need to use pll_frequencies other than 100 MHz.
        # As the programming works at 100MHz, we set pll_frequency to 100MHz
        if pll_frequency != 100e6:
            self.fpga.pll.pll_outfreq_set(100e6, 1)

        programmer.bootstrap(firmware)

        # To handle the PLL frequencies other than 100e6,after programming is done,
        # we switch the pll frequency back to its original value
        if pll_frequency != 100e6:
            self.fpga.pll.pll_outfreq_set(pll_frequency, 1)

        time.sleep(0.5)
        target = cw.target(self.scope)
        target.output_len = output_len
        # Added `pll_frequency` to handle frequencies other than 100MHz.
        # Needed this for OTBN ECDSA.
        # TODO: Remove these comments after discussion
        target.baud = int(baudrate * pll_frequency / 100e6)
        target.flush()

        return target

    def program_target(self, fw, pll_frequency=100e6):
        """Loads firmware image """
        programmer1 = SpiProgrammer(self.fpga)
        # To fully capture the long OTBN applications,
        # we may need to use pll_frequencies other than 100 MHz.
        # As the programming works at 100MHz, we set pll_frequency to 100MHz
        if self.scope.clock.clkgen_freq != 100e6:
            self.fpga.pll.pll_outfreq_set(100e6, 1)

        programmer1.bootstrap(fw)

        # To handle the PLL frequencies other than 100e6,after programming is done,
        # we switch the pll frequency back to its original value
        if self.scope.clock.clkgen_freq != 100e6:
            self.fpga.pll.pll_outfreq_set(pll_frequency, 1)

        time.sleep(0.5)
