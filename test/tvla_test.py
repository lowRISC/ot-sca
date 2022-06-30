# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from .cmd import Args
from .repo import RepoCmd


class TvlaCmd(RepoCmd):
    def __init__(self, args: Args):
        # Insert (relative) path to TVLA before the given arguments.
        args = Args('cw/cw305/tvla.py') + args
        super().__init__(args)


def test_help():
    tvla = TvlaCmd(Args('--help')).run()
    # Assert that a message is printed on stdout or stderr.
    assert(len(tvla.stdout()) != 0 or len(tvla.stderr()) != 0)
