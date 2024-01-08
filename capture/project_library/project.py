#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chipwhisperer as cw
import numpy as np

sys.path.append("../")
from capture.project_library.ot_trace_library.trace_library import (  # noqa: E402
    Trace, TraceLibrary)


@dataclass
class ProjectConfig:
    """ Project configuration.
    Stores information about the project.
    """
    type: str
    path: Path
    wave_dtype: np.dtype
    overwrite: bool
    trace_threshold: Optional[int] = 1


class SCAProject:
    """ Project class.

    Used to manage a SCA project.
    """
    def __init__(self, project_cfg: ProjectConfig) -> None:
        self.project_cfg = project_cfg
        self.project = None

    def create_project(self):
        """ Create project.

        Create project using the provided path.
        """
        if self.project_cfg.type == "cw":
            self.project = cw.create_project(self.project_cfg.path,
                                             overwrite=self.project_cfg.overwrite)
        elif self.project_cfg.type == "ot_trace_library":
            self.project = TraceLibrary(
                str(self.project_cfg.path),
                trace_threshold=self.project_cfg.trace_threshold,
                wave_datatype=self.project_cfg.wave_dtype,
                overwrite=self.project_cfg.overwrite)
        else:
            raise RuntimeError("Only trace_db='cw' or trace_db='ot_trace_library' supported.")

    def open_project(self) -> None:
        """ Open project.
        """
        if self.project_cfg.type == "cw":
            self.project = cw.open_project(self.project_cfg.path)
        elif self.project_cfg.type == "ot_trace_library":
            self.project = TraceLibrary(
                str(self.project_cfg.path),
                trace_threshold=self.project_cfg.trace_threshold,
                wave_datatype=self.project_cfg.wave_dtype,
                overwrite=self.project_cfg.overwrite)

    def close(self, save: bool) -> None:
        """ Close project.
        """
        if self.project_cfg.type == "cw":
            self.project.close(save = save)
        elif self.project_cfg.type == "ot_trace_library":
            self.project.flush_to_disk()

        self.project = None

    def save(self) -> None:
        """ Save project.
        """
        if self.project_cfg.type == "cw":
            self.project.save()
        elif self.project_cfg.type == "ot_trace_library":
            self.project.flush_to_disk()

    def append_trace(self, wave, plaintext, ciphertext, key) -> None:
        """ Append trace to trace storage in project.
        """
        if self.project_cfg.type == "cw":
            trace = cw.Trace(wave, plaintext, ciphertext, key)
            self.project.traces.append(trace, dtype=self.project_cfg.wave_dtype)
        elif self.project_cfg.type == "ot_trace_library":
            trace = Trace(wave=wave.tobytes(),
                          plaintext=plaintext,
                          ciphertext=ciphertext,
                          key=key)
            self.project.write_to_buffer(trace)

    def get_waves(self, start: Optional[int] = None, end: Optional[int] = None):
        """ Get waves from project.
        """
        if self.project_cfg.type == "cw":
            return self.project.waves[start:end]
        elif self.project_cfg.type == "ot_trace_library":
            return self.project.get_waves(start, end)

    def get_keys(self, start: Optional[int] = None, end: Optional[int] = None):
        """ Get keys[start, end] from project.
        """
        if self.project_cfg.type == "cw":
            if start and end:
                return self.project.keys[start:end]
            else:
                return self.project.keys
        elif self.project_cfg.type == "ot_trace_library":
            return self.project.get_keys(start, end)

    def get_plaintexts(self, start: Optional[int] = None, end: Optional[int] = None):
        """ Get plaintexts[start, end] from project.
        """
        if self.project_cfg.type == "cw":
            if start and end:
                return self.project.textins[start:end]
            else:
                return self.project.textins
        elif self.project_cfg.type == "ot_trace_library":
            return self.project.get_plaintexts(start, end)

    def write_metadata(self, metadata: dict) -> None:
        """ Write metadata to project.
        """
        if self.project_cfg.type == "cw":
            self.project.settingsDict.update(metadata)
        elif self.project_cfg.type == "ot_trace_library":
            self.project.write_metadata(metadata)

    def get_metadata(self) -> dict:
        """ Get metadata from project.
        """
        if self.project_cfg.type == "cw":
            return self.project.settingsDict
        elif self.project_cfg.type == "ot_trace_library":
            return self.project.get_metadata()

    def optimize_capture(self, num_segments_storage):
        """Optimize CW capture by managing API."""
        # Make sure to allocate sufficient memory for the storage segment array during the
        # first resize operation. By default, the ChipWhisperer API starts every new segment
        # with 1 trace and then increases it on demand by 25 traces at a time. This results in
        # frequent array resizing and decreasing capture rate.
        # See addWave() in chipwhisperer/common/traces/_base.py.
        if self.project_cfg.type == "cw":
            if self.project.traces.cur_seg.tracehint < self.project.traces.seg_len:
                self.project.traces.cur_seg.setTraceHint(self.project.traces.seg_len)
            # Only keep the latest two trace storage segments enabled. By default the ChipWhisperer
            # API keeps all segments enabled and after appending a new trace, the trace ranges are
            # updated for all segments. This leads to a decreasing capture rate after time.
            # See:
            # - _updateRanges() in chipwhisperer/common/api/TraceManager.py.
            # - https://github.com/newaetech/chipwhisperer/issues/344
            #
            # Before saving the CW project to disk, all trace storage segments need to be
            # re-enabled using finalize_capture().
            if num_segments_storage != len(self.project.segments):
                if num_segments_storage >= 2:
                    self.project.traces.tm.setTraceSegmentStatus(num_segments_storage - 2, False)
                num_segments_storage = len(self.project.segments)
            return num_segments_storage

    def finalize_capture(self, num_traces):
        """Before saving the CW project to disk, re-enable all trace storage segments."""
        # The function optimize_capture above disables all but the most recent two trace storage
        # segments to maintain the capture rate. Before saving the CW project to disk, this
        # needs to be undone.
        if self.project_cfg.type == "cw":
            for s in range(len(self.project.segments)):
                self.project.traces.tm.setTraceSegmentStatus(s, True)
            assert len(self.project.traces) == num_traces
