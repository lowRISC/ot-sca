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
    """ Scope configuration.
    Stores information about the scope.
    """
    scope_type: str
    acqu_channel: Optional[str] = None
    ip: Optional[str] = None
    num_samples: Optional[int] = 0
    offset_samples: Optional[int] = 0
    num_segments: Optional[int] = 1
    sparsing: Optional[int] = 1
    scope_gain: Optional[float] = 1
    pll_frequency: Optional[int] = 1
    sampling_rate: Optional[int] = 0


def determine_sampling_rate(cfg: dict, scope_type: str) -> int:
    """ Determine sampling rate.

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
            return (adc_mul * cfg["target"]["pll_frequency"])
        else:
            # Waverunner init not done yet, so cannot be read from WaveRunner.
            raise RuntimeError("WAVERUNNER: ERROR: Sampling rate for WaveRunner "
                               "not given in configuration.")
    else:
        return cfg[scope_type].get("sampling_rate")


class Scope:
    """ Scope class.

    Represents a Husky or WaveRunner scope and provides functionality to
    initialize and use the scope.
    """
    def __init__(self, scope_cfg: ScopeConfig) -> None:
        self.scope_cfg = scope_cfg
        self.scope = self._init_scope()

    def _init_scope(self):
        """ Init scope.

        Configure Husky or WaveRunner scope with the user provided scope
        settings.
        """
        if self.scope_cfg.scope_type == "husky":
            scope = Husky(scope_gain = self.scope_cfg.scope_gain,
                          num_samples = self.scope_cfg.num_samples,
                          num_segments = self.scope_cfg.num_segments,
                          offset_samples = self.scope_cfg.offset_samples,
                          sampling_rate = self.scope_cfg.sampling_rate,
                          pll_frequency = self.scope_cfg.pll_frequency
                          )
            scope.initialize_scope()
            scope.configure_batch_mode()
            return scope
        elif self.scope_cfg.scope_type == "waverunner":
            if self.scope_cfg.ip:
                scope = WaveRunner(self.scope_cfg.ip)

                # Get current sampling rate from scope compare to cfg for sanity
                setup_data = scope._ask("PANEL_SETUP?")
                scope._get_and_print_cmd_error()
                tmp_sampling_rate = int(re.findall(r'SampleRate = \d+', setup_data)[0][13:])
                if self.scope_cfg.sampling_rate is not None:
                    if self.scope_cfg.sampling_rate != tmp_sampling_rate:
                        raise RuntimeError("WAVERUNNER: Error: WaveRunner sampling "
                                           "rate does not match given configuration!")

                scope.configure_waveform_transfer_general(
                    num_segments = self.scope_cfg.num_segments,
                    sparsing = self.scope_cfg.sparsing,
                    num_samples = self.scope_cfg.num_samples,
                    first_point = self.scope_cfg.offset_samples,
                    acqu_channel = self.scope_cfg.acqu_channel
                )
                return scope
            else:
                raise RuntimeError("Error: No WaveRunner IP provided!")
        else:
            raise RuntimeError("Error: Scope not supported!")

    def arm(self) -> None:
        """ Arm the scope.
        """
        self.scope.arm()

    def capture_and_transfer_waves(self, target=None):
        """ Wait until capture is finished and return traces.
        """
        if self.scope_cfg.scope_type == "husky":
            return self.scope.capture_and_transfer_waves(target)
        else:
            return self.scope.capture_and_transfer_waves()
