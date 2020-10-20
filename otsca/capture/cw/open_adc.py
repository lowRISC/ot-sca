# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""ChipWhisperer Lite and ChipWhisperer Pro capture board utility functions."""
import time

import chipwhisperer as cw


def initialize_scope():
  """Initializes chipwhisperer scope."""
  scope = cw.scope()
  scope.gain.db = 30
  # Samples per trace - We oversample by 10x and AES is doing ~12/16 cycles per
  # encryption.
  scope.adc.samples = 180
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
