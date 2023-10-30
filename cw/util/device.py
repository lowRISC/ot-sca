# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
r"""CW305 utility functions. Used to configure FPGA with OpenTitan design."""

import inspect
import re
import time

import chipwhisperer as cw

from util.spiflash import SpiProgrammer

PLL_FREQUENCY_DEFAULT = 100e6
SAMPLING_RATE_MAX = 200e6


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

    def __init__(self, bitstream, force_programming, firmware, pll_frequency, target_clk_mult,
                 baudrate, scope_gain, num_cycles, offset_cycles, output_len):

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

        self.fpga = self.initialize_fpga(fpga, bitstream, force_programming,
                                         pll_frequency)

        # In our setup, Husky operates on the PLL frequency of the target and
        # multiplies that by an integer number to obtain the sampling rate.
        # Note that the sampling rate must be at most 200 MHz.
        self.clkgen_freq = pll_frequency
        self.adc_mul = int(SAMPLING_RATE_MAX // pll_frequency)
        self.sampling_rate = self.clkgen_freq * self.adc_mul

        # The target runs on the PLL clock but uses internal clock dividers and
        # multiplier to produce the clock of the target block.
        self.target_freq = pll_frequency * target_clk_mult

        # The scope is configured in terms of samples. For Husky, the number of
        # samples must be divisble by 3 for batch captures.
        # TODO: For WaveRunner, we need to read the configured sampling rate here
        # and use that to compute the horizontal offset and number of samples to
        # capture.
        sampling_target_ratio = self.sampling_rate / self.target_freq
        self.offset_samples = int(offset_cycles * sampling_target_ratio)
        self.num_samples = int(num_cycles * sampling_target_ratio)
        if self.num_samples % 3:
            self.num_samples = self.num_samples + 3 - (self.num_samples % 3)

        self.scope = self.initialize_scope(scope_gain, self.num_samples, self.offset_samples,
                                           self.clkgen_freq, self.adc_mul)
        print(f'Scope setup with sampling rate {self.scope.clock.adc_freq} S/s')
        print(f'Resulting oversampling ratio {sampling_target_ratio}')

        self.target = self.initialize_target(programmer, firmware, baudrate,
                                             output_len, pll_frequency)

        self._test_read_version_from_target()

    def initialize_fpga(self, fpga, bitstream, force_programming,
                        pll_frequency):
        """Initializes FPGA bitstream and sets PLL frequency."""
        # Do not program the FPGA if it is already programmed.
        # Note: Set this to True to force programming the FPGA when using a new
        # bitstream.
        print('Connecting and loading FPGA... ', end='', flush = True)

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

    def initialize_scope(self, scope_gain, num_samples, offset_samples, clkgen_freq, adc_mul):
        """Initializes chipwhisperer scope."""
        scope = cw.scope()
        scope.gain.db = scope_gain
        scope.adc.basic_mode = "rising_edge"

        scope.clock.clkgen_src = 'extclk'
        scope.clock.clkgen_freq = clkgen_freq
        scope.clock.adc_mul = adc_mul
        scope.clock.extclk_monitor_enabled = False
        scope.adc.samples = num_samples
        if offset_samples >= 0:
            scope.adc.offset = offset_samples
        else:
            scope.adc.offset = 0
            scope.adc.presamples = -offset_samples
        scope.trigger.triggers = "tio4"
        scope.io.tio1 = "serial_tx"
        scope.io.tio2 = "serial_rx"
        scope.io.hs2 = "disabled"

        # Make sure that clkgen_locked is true.
        scope.clock.clkgen_src = 'extclk'

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
        # The bootstrapping always runs at 100 MHz.
        if pll_frequency != PLL_FREQUENCY_DEFAULT:
            self.fpga.pll.pll_outfreq_set(PLL_FREQUENCY_DEFAULT, 1)

        programmer.bootstrap(firmware)

        # After boostrapping, we can again configure the actually desired
        # PLL frequency.
        if pll_frequency != PLL_FREQUENCY_DEFAULT:
            self.fpga.pll.pll_outfreq_set(pll_frequency, 1)

        time.sleep(0.5)
        target = cw.target(self.scope)
        target.output_len = output_len
        target.baud = int(baudrate * pll_frequency / PLL_FREQUENCY_DEFAULT)
        target.flush()

        return target

    def _test_read_version_from_target(self):
        version = None
        ping_cnt = 0
        while not version:
            if ping_cnt == 3:
                raise RuntimeError(
                    f'No response from the target (attempts: {ping_cnt}).')
            self.target.write('v' + '\n')
            ping_cnt += 1
            time.sleep(0.5)
            version = self.target.read().strip()
        print(f'Target simpleserial version: {version} (attempts: {ping_cnt}).')

    def program_target(self, fw, pll_frequency=PLL_FREQUENCY_DEFAULT):
        """Loads firmware image """
        programmer1 = SpiProgrammer(self.fpga)
        # To fully capture the long OTBN applications,
        # we may need to use pll_frequencies other than 100 MHz.
        # As the programming works at 100MHz, we set pll_frequency to 100MHz
        if self.scope.clock.clkgen_freq != PLL_FREQUENCY_DEFAULT:
            self.fpga.pll.pll_outfreq_set(PLL_FREQUENCY_DEFAULT, 1)

        programmer1.bootstrap(fw)

        # To handle the PLL frequencies other than PLL_FREQUENCY_DEFAULT, after
        # programming is done, we switch the pll frequency back to its original
        # value
        if self.scope.clock.clkgen_freq != PLL_FREQUENCY_DEFAULT:
            self.fpga.pll.pll_outfreq_set(pll_frequency, 1)

        time.sleep(0.5)
