# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""CW utility functions. Used to configure FPGA with OpenTitan design."""

import inspect
import re
import time

import chipwhisperer as cw

from util.spiflash import SpiProgrammer

PLL_FREQUENCY_DEFAULT = 100e6


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


class CWFPGA(object):
    """Class for the CW FPGA. Initializes the FPGA with the bitstream and the
    target binary.
    """

    def __init__(self, bitstream, force_programming, firmware, pll_frequency,
                 baudrate, output_len, protocol):
        self.bitstream = bitstream
        self.firmware = firmware
        self.pll_frequency = pll_frequency
        self.baudrate = baudrate
        self.output_len = output_len

        # Extract target board type from bitstream name.
        m = re.search('cw305|cw310', bitstream)
        if m:
            if m.group() == 'cw305':
                self.fpga_type = cw.capture.targets.CW305()
            else:
                assert m.group() == 'cw310'
                self.fpga_type = cw.capture.targets.CW310()
            programmer = SpiProgrammer(self.fpga_type)
        else:
            raise ValueError(
                'Could not infer target board type from bistream name')
        self.prot_simple_serial = True
        if protocol == "ujson":
            self.prot_simple_serial = False
        # Initialize ChipWhisperer scope. This is needed to program the binary.
        # Note that the actual scope config for capturing traces is later
        # initialized.
        self.scope = cw.scope()
        # Sometimes CW-Husky blocks USB communication after power cycling
        # Initializing the scope twice seems to solve the problem.
        self.scope = cw.scope()

        self.fpga = self.initialize_fpga(self.fpga_type, bitstream,
                                         force_programming, pll_frequency)

        self.target = self.initialize_target(programmer, firmware, baudrate,
                                             output_len, pll_frequency)

        # TODO: add version check also for uJson binary.
        if self.prot_simple_serial:
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

    def reset_target(self):
        """Resets the target. """
        # Check if FPGA bitstream is corrupted by reading the state of INITB.
        if self.fpga.INITB_state() is not True:
            print("Reprogram the FPGA.")
            # Reprogram the bitstream.
            programmer = SpiProgrammer(self.fpga_type)

            self.scope = cw.scope()

            self.fpga = self.initialize_fpga(self.fpga_type, self.bitstream,
                                             True, self.pll_frequency)

            self.target = self.initialize_target(programmer, self.firmware,
                                                 self.baudrate, self.output_len,
                                                 self.pll_frequency)
            # TODO: add version check also for uJson binary.
            if self.prot_simple_serial:
                self._test_read_version_from_target()
        else:
            # Bitstream seems to be OK, reset OpenTitan.
            print("Reset OpenTitan.")
            # POR_N (OpenTitan) is connected to USB_A14 (CW310 SAM3X)
            io = self.fpga.gpio_mode()
            io.pin_set_output("USB_A14")
            io.pin_set_state("USB_A14", 1)
            # Trigger reset for 100ms.
            io.pin_set_state("USB_A14", 0)
            time.sleep(0.1)
            # Deactivate reset.
            io.pin_set_state("USB_A14", 1)
            # Add a small delay to allow OpenTitan to boot up.
            time.sleep(0.1)
