# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Support for capturing traces using ChipWhisperer-Lite in segmented mode."""

import logging
import re
import time

import chipwhisperer as cw
import numpy as np
from numpy.lib.stride_tricks import as_strided


class CwLiteSegmented:
    """Class for capturing traces using a ChipWhisperer-Lite.

    This class uses segmented traces mode to improve capture performance.

    When in segmented mode, ChipWhisperer-Lite captures multiple segments each starting
    at a trigger event. This is much more efficient than sending a separate command for
    each segment.

    Due to current limitations of the firmware, ChipWhisperer-Lite needs to fill the
    entire sample buffer when running in batch mode. Therefore, the number of segments
    that need to be captured may be fewer or more than the desired number of segments
    depending on the number of samples per segment. Additionally, the last segment may
    be captured only partially. This class handles all the associated processing,
    including discarding partially captured segments, internally.

    However, the target must generate the correct number of triggers so that the entire
    sample buffer is filled. This number is made available via the read-only
    ``num_segments_actual`` attribute so that the target can be configured accordingly.

    Typical usage:
    >>> cw_lite = CwLiteSegmented()
    >>> while foo:
    >>>     ...
    >>>     cw_lite.num_segments = desired_num_segments
    >>>     # Note: cw_lite.num_segments_actual gives the actual number of segments that
    >>>     # will be captured.
    >>>     cw_lite.arm()
    >>>     # Configure the target using cw_lite.num_segments_actual.
    >>>     target.configure(cw_lite.num_segments_actual)
    >>>     # This returns cw_lite.num_segments number of waves each with
    >>>     # cw_lite.num_samples samples.
    >>>     waves = cw_lite.capture_and_transfer_waves()
    >>>     ...

    Attributes:
        num_samples_min: Minimum number of samples per segment. Read-only.
        num_samples_max: Maximum number of samples per segment. Read-only.
        num_samples: Number of samples per segment, must be in [``num_samples_min``,
            ``num_samples_max``].
        num_segments_min: Minimum number of segments per capture. Read-only.
        num_segments_max: Maximum number of segments per capture, depends on the number
            of samples per segment. Read-only.
        num_segments: Number of segments per capture, must be in [``num_segments_min``,
            ``num_segments_max``].  This number determines the number of segments
            returned by ``capture_and_transfer_segments``.
        num_segments_actual: Actual number of segments that will be captured. This
            number depends on the number of samples per segment. The target must create
            this many trigger events. Read-only.
    """

    def __init__(self, num_samples=740):
        """Inits a CwLiteSegmented.

        Args:
            num_samples: Number of samples per segment, must be in [``num_samples_min``,
                ``num_samples_max``].
        """
        self._scope = cw.scope()
        self._configure_scope()
        self.num_segments = 1
        self.num_samples = num_samples
        self._print_device_info()

    @property
    def num_segments_min(self):
        return 1

    @property
    def num_segments_max(self):
        return (self._scope.adc.oa.hwMaxSamples // self._scope.adc.samples) - 1

    @property
    def num_segments_actual(self):
        # Must round-up to fill the entire buffer.
        return round(self._scope.adc.oa.hwMaxSamples / self._scope.adc.samples) + 1

    @property
    def num_segments(self):
        return self._num_segments

    @num_segments.setter
    def num_segments(self, num_segments):
        if not self.num_segments_min <= num_segments <= self.num_segments_max:
            raise RuntimeError(
                f"num_segments must be in [{self.num_segments_min}, {self.num_segments_max}]."
            )
        self._num_segments = num_segments

    @property
    def num_samples_min(self):
        return 1

    @property
    def num_samples_max(self):
        # TODO: is this correct?
        return self._scope.adc.oa.hwMaxSamples

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, num_samples):
        if not self.num_samples_min <= num_samples <= self.num_samples_max:
            raise RuntimeError(
                f"num_samples must be in [{self.num_samples_min}, {self.num_samples_max}]."
            )
        self._num_samples = num_samples
        # This should ideally be handled by the chipwhisperer library but setting the
        # number of samples smaller than 241 results in "received fewer points than
        # expected" error. This number is further rounded up by chipwhisperer so that
        # (num_samples-1) is divisible by 3. We get the actual number of samples from
        # adc.
        self._scope.adc.samples = max(241, num_samples)
        # Note: CW-Lite actually returns one less than adc.samples.
        self._num_samples_actual = self._scope.adc.samples - 1
        if self.num_segments > self.num_segments_max:
            print(f"Warning: Adjusting number of segments to {self.num_segments_max}.")
            self.num_segments = self.num_segments_max

    def _configure_scope(self):
        self._scope.gain.db = 23
        self._scope.adc.offset = 0
        self._scope.adc.basic_mode = "rising_edge"
        self._scope.adc.fifo_fill_mode = "segment"
        self._scope.clock.clkgen_freq = 100000000
        # We sample using the target clock (100 MHz).
        self._scope.clock.adc_src = "extclk_dir"
        self._scope.trigger.triggers = "tio4"
        self._scope.io.tio1 = "serial_tx"
        self._scope.io.tio2 = "serial_rx"
        self._scope.io.hs2 = "disabled"
        self._scope.clock.reset_adc()
        time.sleep(0.5)
        assert self._scope.clock.adc_locked, "ADC failed to lock"

    def _print_device_info(self):
        print(
            (
                "Connected to ChipWhisperer-Lite ("
                f"num_samples: {self.num_samples}, "
                f"num_samples_actual: {self._num_samples_actual}, "
                f"num_segments_actual: {self.num_segments_actual})"
            )
        )

    def arm(self):
        """Arms ChipWhisperer-Lite."""
        self._scope.arm()

    def _parse_waveform(self, data):
        shape = (self.num_segments, self.num_samples)
        strides = (data.itemsize * self._num_samples_actual, data.itemsize)
        return as_strided(data, shape, strides, writeable=False)

    def capture_and_transfer_waves(self):
        """Waits until the acquisition is complete and transfers waves.

        Returns:
            Waves.
        """
        self._scope.capture_segmented()
        data = self._scope.get_last_trace()
        waves = self._parse_waveform(data)
        return waves
