# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

r"""CW305 utility functions. Used to configure FPGA with OpenTitan design."""

import os
import subprocess
import time

import chipwhisperer as cw


class OpenTitan(object):
  def __init__(self, fw_programmer, bitstream, pll_frequency, baudrate, scope):
      self.fpga = self.initialize_fpga(bitstream, pll_frequency)
      self.target = self.initialize_target(scope, fw_programmer, baudrate)

  def initialize_fpga(self, bitstream, pll_frequency):
    """Initializes FPGA bitstream and sets PLL frequency."""
    print('Connecting and loading FPGA')
    fpga = cw.capture.targets.CW305()
    # Do not program the FPGA if it is already programmed.
    fpga.con(bsfile=bitstream, force=False)
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

  def initialize_target(self, scope, fw_programmer, baudrate):
    """Loads firmware image and initializes test target."""
    fw_programmer.run()
    time.sleep(0.5)
    target = cw.target(scope)
    target.output_len = 16
    target.baud = baudrate
    target.flush()
    return target
