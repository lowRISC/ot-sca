# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
from cw_segmented import CwSegmented


class HuskyDispatcher:

    def __init__(self, opentitan_device, num_segments):
        self.num_segments = num_segments
        self.opentitan_device = opentitan_device

        if self.num_segments == 1:
            self.scope = opentitan_device.scope
        else:  # Batch mode
            self.scope = CwSegmented(num_samples=opentitan_device.num_samples,
                                     offset_samples=opentitan_device.offset_samples,
                                     scope_gain=opentitan_device.scope.gain.db,
                                     scope=opentitan_device.scope,
                                     clkgen_freq=opentitan_device.scope.clock.clkgen_freq,
                                     adc_mul=opentitan_device.adc_mul)
            self.scope.num_segments = num_segments

    def arm(self):
        self.scope.arm()

    def capture_and_transfer_waves(self):
        if self.num_segments == 1:
            ret = self.scope.capture(poll_done=False)
            i = 0
            while not self.opentitan_device.target.is_done():
                i += 1
                time.sleep(0.05)
                if i > 100:
                    print("Warning: Target did not finish operation")
            if ret:
                print("Warning: Timeout happened during capture")

            # Get Husky trace (single mode only) and return as array with one item
            return np.array([self.scope.get_last_trace(as_int=True)])

        # Batch mode
        return self.scope.capture_and_transfer_waves()
