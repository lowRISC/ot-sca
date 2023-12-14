#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

from fault_injection.project_library.ot_fi_library.fi_library import (
    FILibrary, FIResult)


class FISuccess(IntEnum):
    SUCCESS = 1
    EXPRESPONSE = 2
    NORESPONSE = 3


@dataclass
class ProjectConfig:
    """ Project configuration.
    Stores information about the project.
    """
    type: str
    path: Path
    overwrite: bool
    fi_threshold: Optional[int] = 1


class FIProject:
    """ Project class.

    Used to manage a FI project.
    """
    def __init__(self, project_cfg: ProjectConfig) -> None:
        self.project_cfg = project_cfg
        self.project = None

    def create_project(self):
        """ Create project.

        Create project using the provided path.
        """
        if self.project_cfg.type == "ot_fi_project":
            self.project = FILibrary(
                str(self.project_cfg.path),
                fi_threshold=self.project_cfg.fi_threshold,
                overwrite=self.project_cfg.overwrite)
        else:
            raise RuntimeError("Only project_db='ot_fi_project' supported.")

    def open_project(self) -> None:
        """ Open project.
        """
        if self.project_cfg.type == "ot_fi_project":
            self.project = FILibrary(
                str(self.project_cfg.path),
                fi_threshold=self.project_cfg.fi_threshold,
                overwrite=self.project_cfg.overwrite)

    def close(self, save: bool) -> None:
        """ Close project.
        """
        if self.project_cfg.type == "ot_fi_project":
            self.project.flush_to_disk()
        self.project = None

    def save(self) -> None:
        """ Save project.
        """
        if self.project_cfg.type == "ot_fi_project":
            self.project.flush_to_disk()

    def append_firesult(self, response: str, fi_result: int, trigger_delay: int,
                        glitch_voltage: Optional[float] = 0,
                        glitch_width: Optional[float] = 0,
                        x_pos: Optional[int] = 0,
                        y_pos: Optional[int] = 0) -> None:
        """ Append FI result to storage in project.
        """
        if self.project_cfg.type == "ot_fi_project":
            firesult = FIResult(response = response,
                                fi_result = int(fi_result),
                                trigger_delay = trigger_delay,
                                glitch_width = glitch_width,
                                glitch_voltage = glitch_voltage,
                                x_pos = x_pos, y_pos = y_pos)
            self.project.write_to_buffer(firesult)

    def get_firesults(self, start: Optional[int] = None,
                      end: Optional[int] = None):
        """ Get FI results from database and stored into RAM.

        Fetch FI results from start to end from database storage into RAM.

        Args:
            start: fetch FI result from trace index start
            end: to FI result index end. If no start and end is provided, all
                 FI result are returned.
        Returns:
            The FI results from the database.
        """
        if self.project_cfg.type == "ot_fi_project":
            return self.project.get_firesults(start, end)

    def write_metadata(self, metadata: dict) -> None:
        """ Write metadata to project.
        """
        if self.project_cfg.type == "ot_fi_project":
            self.project.write_metadata(metadata)

    def get_metadata(self) -> dict:
        """ Get metadata from project.
        """
        if self.project_cfg.type == "ot_fi_project":
            return self.project.get_metadata()
