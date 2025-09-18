#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from typing import Optional

from scopes.chipwhisperer.husky import Husky
from scopes.waverunner.waverunner import WaveRunner


@dataclass
class ScopeConfig:
    """Scope configuration.
    Stores information about the scope.
    """

    scope_type: str
    num_segments: int
    batch_mode: bool
    bit: Optional[int] = 8
    acqu_channel: Optional[str] = None
    ip: Optional[str] = None
    num_samples: Optional[int] = 0
    offset_samples: Optional[int] = 0
    sparsing: Optional[int] = 1
    scope_gain: Optional[float] = 1
    pll_frequency: Optional[int] = 1
    sampling_rate: Optional[int] = 0
    scope_sn: Optional[str] = None


class Scope:
    """Scope class.

    Represents a Husky or WaveRunner scope and provides functionality to
    initialize and use the scope.
    """

    def __init__(self, scope_cfg: ScopeConfig) -> None:
        self.scope_cfg = scope_cfg
        self.scope = self._init_scope()

    def _init_scope(self):
        """Init scope.

        Configure Husky or WaveRunner scope with the user provided scope
        settings.
        """
        # Check if num_segments is 1 in non-batch mode.
        self._sanitize_num_segments()
        # Configure Scopes.
        if self.scope_cfg.scope_type == "husky":
            scope = Husky(
                scope_gain=self.scope_cfg.scope_gain,
                batch_mode=self.scope_cfg.batch_mode,
                num_samples=self.scope_cfg.num_samples,
                num_segments=self.scope_cfg.num_segments,
                offset_samples=self.scope_cfg.offset_samples,
                sampling_rate=self.scope_cfg.sampling_rate,
                pll_frequency=self.scope_cfg.pll_frequency,
                scope_sn=self.scope_cfg.scope_sn,
            )
            scope.initialize_scope()
            scope.configure_batch_mode()
            # Update num_segments.
            self.scope_cfg.num_segments = scope.num_segments
            return scope
        elif self.scope_cfg.scope_type == "waverunner":
            if self.scope_cfg.ip:
                resolution = self.scope_cfg.bit
                if self.scope_cfg.bit is None:
                    resolution = 8
                scope = WaveRunner(self.scope_cfg.ip, resolution)

                # Get current sampling rate from scope compare to cfg for sanity
                setup_data = scope._ask("PANEL_SETUP?")
                scope._get_and_print_cmd_error()
                tmp_sampling_rate = int(
                    re.findall(r"SampleRate = \d+", setup_data)[0][13:])
                # Sanitize inputs.
                if self.scope_cfg.sampling_rate is not None:
                    if self.scope_cfg.sampling_rate != tmp_sampling_rate:
                        raise RuntimeError(
                            "WAVERUNNER: Error: WaveRunner sampling "
                            "rate does not match given configuration!")
                if self.scope_cfg.num_segments is None:
                    raise RuntimeError(
                        "WAVERUNNER: Error: num_segments needs to "
                        "be provided!")
                # Configure WaveRunner.
                scope.configure_waveform_transfer_general(
                    num_segments=self.scope_cfg.num_segments,
                    sparsing=self.scope_cfg.sparsing,
                    num_samples=self.scope_cfg.num_samples,
                    first_point=self.scope_cfg.offset_samples,
                    acqu_channel=self.scope_cfg.acqu_channel,
                )
                return scope
            else:
                raise RuntimeError("Error: No WaveRunner IP provided!")
        else:
            raise RuntimeError("Error: Scope not supported!")

    def _sanitize_num_segments(self) -> None:
        """Sanitize num_segments.

        When in non-batch mode, num_segments needs to be 1.
        """
        if not self.scope_cfg.batch_mode and self.scope_cfg.num_segments != 1:
            self.scope_cfg.num_segments = 1
            print("Warning: num_segments needs to be 1 in non-batch mode. "
                  "Setting num_segments=1.")

    def arm(self) -> None:
        """Arm the scope."""
        self.scope.arm()

    def capture_and_transfer_waves(self, target=None):
        """Wait until capture is finished and return traces."""
        if self.scope_cfg.scope_type == "husky":
            return self.scope.capture_and_transfer_waves(target)
        else:
            return self.scope.capture_and_transfer_waves()


def convert_num_cycles(cfg: dict, scope_type: str) -> int:
    """Converts number of cycles to number of samples if samples not given.

    As the scopes are configured in number of samples, this function converts
    the number of cycles to samples.
    The number of samples must be divisble by 3 for batch captures on Husky
    and is adjusted accordingly.

    Args:
        dict: The scope configuration.
        scope_type: The used scope (Husky or WaveRunner).

    Returns:
        The number of samples.
    """
    if cfg[scope_type].get("num_samples") is None:
        sampl_target_rat = cfg[scope_type].get(
            "sampling_rate") / cfg["target"].get("target_freq")
        num_samples = int(cfg[scope_type].get("num_cycles") * sampl_target_rat)

        if scope_type == "husky":
            if num_samples % 3:
                num_samples = num_samples + 3 - (num_samples % 3)

        return num_samples
    else:
        return cfg[scope_type].get("num_samples")


def convert_offset_cycles(cfg: dict, scope_type: str) -> int:
    """Converts offset in cycles to offset in samples if not given in samples.

    Args:
        dict: The scope configuration.
        scope_type: The used scope (Husky or WaveRunner).

    Returns:
        The offset in samples.
    """
    if cfg[scope_type].get("offset_samples") is None:
        sampl_target_rat = cfg[scope_type].get(
            "sampling_rate") / cfg["target"].get("target_freq")
        return int(cfg[scope_type].get("offset_cycles") * sampl_target_rat)
    else:
        return cfg[scope_type].get("offset_samples")


def determine_sampling_rate(cfg: dict, scope_type: str) -> int:
    """Determine sampling rate.

    If no sampling rate is provided, calculate for Husky or receive from
    WaveRunner.

    Args:
        dict: The scope configuration.
        scope_type: The used scope (Husky or WaveRunner).

    Returns:
        Sampling rate
    """
    if cfg[scope_type].get("sampling_rate") is None:
        if scope_type == "husky":
            # If no sampling rate is provided, calculte it. Max. sampling
            # rate of Husky is 200e6. As adc_mul needs to be an integer,
            # calculate the maximum possible sampling rate.
            SAMPLING_RATE_MAX = 200e6
            adc_mul = int(SAMPLING_RATE_MAX // cfg["target"]["pll_frequency"])
            return adc_mul * cfg["target"]["pll_frequency"]
        else:
            # Waverunner init not done yet, so cannot be read from WaveRunner.
            raise RuntimeError(
                "WAVERUNNER: ERROR: Sampling rate for WaveRunner "
                "not given in configuration.")
    else:
        return cfg[scope_type].get("sampling_rate")
