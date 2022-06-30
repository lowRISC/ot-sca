# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to run tests in the context of this repository."""

import pathlib

from .cmd import Args, Cmd

REPO_PATH = pathlib.Path(__file__).parent.parent.resolve()


class RepoCmd(Cmd):
    def __init__(self, args: Args):
        # Prepend absolute path to repository to the command.
        args[0] = str(REPO_PATH / args[0])
        super().__init__(args)
