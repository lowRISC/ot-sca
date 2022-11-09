# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .cmd import Args
from .repo import RepoCmd, TestDataPath


class TvlaCmd(RepoCmd):
    def __init__(self, args: Args):
        # Insert (relative) path to TVLA before the given arguments.
        args = Args('cw/tvla.py') + args
        super().__init__(args)


def test_help():
    tvla = TvlaCmd(Args('--help')).run()
    # Assert that a message is printed on stdout or stderr.
    assert (len(tvla.stdout()) != 0 or len(tvla.stderr()) != 0)


def ttest_significant(ttest_trace) -> bool:
    """Determine if a t-test trace contains a significant deviation from the mean."""
    threshold = 4.5
    abs_max = np.max(np.abs(ttest_trace), axis=3)
    return np.any(abs_max > threshold)


def test_general_leaking_histogram():
    hist_path = TestDataPath('tvla_general/kmac_hist_leaking.npz')
    tvla = TvlaCmd(Args(['--input-file', str(hist_path),
                        '--mode', 'kmac', 'run-tvla'])).run()
    assert ttest_significant(np.load('tmp/ttest.npy')), (
           f"{tvla} did not find significant leakage, which is unexpected")


def test_general_nonleaking_histogram():
    hist_path = TestDataPath('tvla_general/kmac_hist_nonleaking.npz')
    tvla = TvlaCmd(Args(['--input-file', str(hist_path),
                        '--mode', 'kmac', 'run-tvla'])).run()
    assert not ttest_significant(np.load('tmp/ttest.npy')), (
           f"{tvla} did find significant leakage, which is unexpected")
