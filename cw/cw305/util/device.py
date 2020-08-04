# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

r"""CW305 utility functions. Used to configure FPGA with OpenTitan design."""

import subprocess
import time

import chipwhisperer as cw


SPIFLASH=r'bin/linux/spiflash'


class OpenTitan(object):
  def __init__(self, bitstream, fw_image, pll_frequency, baudrate):
      self.fpga = self.initialize_fpga(bitstream, pll_frequency)
      self.scope = self.initialize_scope()
      self.target = self.initialize_target(self.scope, fw_image, baudrate)

  def initialize_fpga(self, bitstream, pll_frequency):
    """Initializes FPGA bitstream and sets PLL frequency."""
    print('Connecting and loading FPGA')
    fpga = cw.capture.targets.CW305()
    fpga.con(bsfile=bitstream, force=True)
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

  def initialize_scope(self):
    """Initializes chipwhisperer scope."""
    scope = cw.scope()
    scope.gain.db = 15
    scope.adc.samples =  200
    scope.adc.offset = 600
    scope.adc.basic_mode = "rising_edge"
    scope.clock.clkgen_freq = 18425000
    scope.clock.adc_src = "extclk_x4"
    scope.trigger.triggers = "tio4"
    scope.io.tio1 = "serial_tx"
    scope.io.tio2 = "serial_rx"
    scope.io.hs2 = "disabled"

    # TODO: Need to update error handling.
    scope.clock.reset_adc()
    time.sleep(0.5)
    assert (scope.clock.adc_locked), "ADC failed to lock"
    return scope

  def load_fw(self, fw_image):
    """Loads firmware image."""
    # TODO: Make settings configurable.
    command = [SPIFLASH, '--dev-id=0403:6014', '--dev-sn=FT2U2SK1',
           '--input=' + fw_image]
    subprocess.check_call(command)

  def initialize_target(self, scope, fw_image, baudrate):
    """Loads firmware image and initializes test target."""
    self.load_fw(fw_image)
    time.sleep(0.5)
    target = cw.target(scope)
    target.output_len = 16
    target.baud = baudrate
    target.flush()
    return target
