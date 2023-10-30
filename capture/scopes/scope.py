#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

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
    num_cycles: Optional[int] = 0
    num_samples: Optional[int] = 0
    offset_cycles: Optional[int] = 0
    offset_samples: Optional[int] = 0
    target_clk_mult: Optional[int] = 0
    num_segments: Optional[int] = 1
    sparsing: Optional[int] = 1
    scope_gain: Optional[float] = 1
    pll_frequency: Optional[int] = 1


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
                          num_cycles = self.scope_cfg.num_cycles,
                          num_segments = self.scope_cfg.num_segments,
                          offset_cycles = self.scope_cfg.offset_cycles,
                          pll_frequency = self.scope_cfg.pll_frequency,
                          target_clk_mult = self.scope_cfg.target_clk_mult
                          )
            scope.initialize_scope()
            scope.configure_batch_mode()
            return scope
        elif self.scope_cfg.scope_type == "waverunner":
            # TODO WaveRunner needs to be adapted to new cycle-based config.
            # (Issue lowrisc/ot-sca#210)
            if self.scope_cfg.ip:
                scope = WaveRunner(self.scope_cfg.ip)
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

    def capture_and_transfer_waves(self):
        """ Wait until capture is finished and return traces.
        """
        return self.scope.capture_and_transfer_waves()
