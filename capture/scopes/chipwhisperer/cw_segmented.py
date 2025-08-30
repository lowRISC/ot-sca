# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Support for capturing traces using ChipWhisperer-Husky in segmented mode."""

import time

import chipwhisperer as cw
from numpy.lib.stride_tricks import as_strided


class CwSegmented:
    """Class for capturing traces using a ChipWhisperer-Husky.

    This class uses segmented traces mode to improve capture performance.

    When in segmented mode, ChipWhisperer-Husky captures multiple segments each starting
    at a trigger event. This is much more efficient than sending a separate command for
    each segment.

    Typical usage:
    >>> cw_scope = CwSegmented()
    >>> while foo:
    >>>     ...
    >>>     cw_scope.num_segments = desired_num_segments
    >>>     # Note: cw_scope.num_segments_actual gives the actual number of segments that
    >>>     # will be captured.
    >>>     cw_scope.arm()
    >>>     # Configure the target using cw_scope.num_segments_actual.
    >>>     target.configure(cw_scope.num_segments_actual)
    >>>     # This returns cw_scope.num_segments number of waves each with
    >>>     # cw_scope.num_samples samples.
    >>>     waves = cw_scope.capture_and_transfer_waves()
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

    def __init__(
        self,
        num_samples=1200,
        offset_samples=0,
        scope_gain=23,
        scope=None,
        clkgen_freq=100e6,
        adc_mul=2,
    ):
        """Inits a CwSegmented.

        Args:
            num_samples: Number of samples per segment, must be in [``num_samples_min``,
                ``num_samples_max``].
        """
        if scope:
            self._scope = scope
        else:
            self._scope = cw.scope()

        self._configure_scope(scope_gain, offset_samples, clkgen_freq, adc_mul)

        self.num_segments = 1
        self.num_samples = num_samples
        self._print_device_info()

    @property
    def num_segments_min(self):
        return 1

    @property
    def num_segments_max(self):
        return self._scope.adc.oa.hwMaxSegmentSamples // self._scope.adc.samples

    @property
    def num_segments_actual(self):
        return self._scope.adc.segments

    @property
    def num_segments(self):
        return self._num_segments

    @num_segments.setter
    def num_segments(self, num_segments):
        self._scope.adc.segments = num_segments
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
        self._scope.adc.samples = num_samples
        self._num_samples_actual = num_samples

        if self.num_segments > self.num_segments_max:
            print(f"Warning: Adjusting number of segments to {self.num_segments_max}.")
            self.num_segments = self.num_segments_max

    def _configure_scope(self, scope_gain, offset_samples, clkgen_freq, adc_mul):
        self._scope.gain.db = scope_gain
        if offset_samples >= 0:
            self._scope.adc.offset = offset_samples
        else:
            self._scope.adc.offset = 0
            self._scope.adc.presamples = -offset_samples
        self._scope.adc.basic_mode = "rising_edge"
        self._scope.clock.adc_mul = adc_mul
        self._scope.clock.clkgen_freq = clkgen_freq
        self._scope.clock.clkgen_src = "extclk"

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
                "Connected to ChipWhisperer ("
                f"num_samples: {self.num_samples}, "
                f"num_samples_actual: {self._num_samples_actual}, "
                f"num_segments_actual: {self.num_segments_actual})"
            )
        )

    def arm(self):
        """Arms ChipWhisperer."""
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
        self._scope.capture()
        data = self._scope.get_last_trace(as_int=True)
        waves = self._parse_waveform(data)
        return waves
