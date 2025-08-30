# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to run tests in the context of this repository."""

from pathlib import Path
from typing import Union

from .cmd import Args, Cmd

REPO_PATH = Path(__file__).parent.parent.resolve()


class RepoCmd(Cmd):
    def __init__(self, args: Args):
        # Prepend absolute path to repository to the command.
        args[0] = str(REPO_PATH / args[0])
        super().__init__(args)


class TestDataPath:
    """Absolute path to a test data file"""

    # This is not a test class.
    __test__ = False

    def __init__(self, subpath: Union[str, Path]):
        """Create an absolute path to a test data file from a subpath.

        The subpath argument is a string or a Path object, relative to the test data directory.
        """
        self._path = REPO_PATH / "test" / "data" / subpath

    def __str__(self) -> str:
        return str(self._path)
