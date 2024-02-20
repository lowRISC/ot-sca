# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .cmd import Args
from .repo import RepoCmd, TestDataPath


class TvlaCmd(RepoCmd):
    def __init__(self, args: Args):
        # Insert (relative) path to TVLA before the given arguments.
        args = Args('analysis/tvla.py') + args
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


def ttest_compare_results(expected, received, delta) -> bool:
    """
    Determine if the numerical values of two ttest traces match.

    Checks if all nan values are in the same positions in both arrays.
    Checks if all numerical values match within precision delta.
    """
    nan_match = np.all(np.isnan(expected) == np.isnan(received))
    numbers_match = np.all(np.logical_or(np.isnan(expected), np.abs(expected - received) < delta))
    return nan_match and numbers_match


def test_general_kmac_nonleaking_project():
    project_path = TestDataPath('tvla_general/ci_opentitan_simple_kmac.cwp')
    tvla = TvlaCmd(Args(['--project-file', str(project_path),
                         '--mode', 'kmac', '--save-to-disk-ttest', '--test-type', 'GENERAL_KEY',
                         '--number-of-steps', '10', 'run-tvla'])).run()
    expected_path = TestDataPath('tvla_general/ttest-step-golden-kmac.npy.npz')
    expected_file = np.load(str(expected_path))
    expected_trace = expected_file['ttest_step']
    received_file = np.load('tmp/ttest-step.npy.npz')
    received_trace = received_file['ttest_step']
    # Expected and received traces should be equal within precision delta.
    # Small mismatch is possible due to the differences in floating point operations
    # on different machines.
    delta = 0.001
    assert ttest_compare_results(expected_trace, received_trace, delta), (
           f"{tvla} generated ttest_step values that don't match the expected ones")


def test_general_aes_nonleaking_project():
    project_path = TestDataPath('tvla_general/ci_opentitan_simple_aes_fvsr.cwp')
    tvla = TvlaCmd(Args(['--project-file', str(project_path),
                         '--mode', 'aes', '--save-to-disk-ttest', '--test-type', 'GENERAL_KEY',
                         '--number-of-steps', '10', 'run-tvla'])).run()
    expected_path = TestDataPath('tvla_general/ttest-step-golden-aes.npy.npz')
    expected_file = np.load(str(expected_path))
    expected_trace = expected_file['ttest_step']
    received_file = np.load('tmp/ttest-step.npy.npz')
    received_trace = received_file['ttest_step']
    delta = 0.001
    assert ttest_compare_results(expected_trace, received_trace, delta), (
           f"{tvla} generated ttest_step values that don't match the expected ones")


def test_general_leaking_histogram():
    hist_path = TestDataPath('tvla_general/kmac_hist_leaking.npz')
    tvla = TvlaCmd(Args(['--input-histogram-file', str(hist_path),
                        '--mode', 'kmac', '--save-to-disk-ttest', '--test-type', 'GENERAL_KEY',
                         'run-tvla'])).run()
    assert ttest_significant(np.load('tmp/ttest.npy')), (
           f"{tvla} did not find significant leakage, which is unexpected")


def test_general_nonleaking_histogram():
    hist_path = TestDataPath('tvla_general/kmac_hist_nonleaking.npz')
    tvla = TvlaCmd(Args(['--input-histogram-file', str(hist_path),
                        '--mode', 'kmac', '--save-to-disk-ttest', '--test-type', 'GENERAL_KEY',
                         'run-tvla'])).run()
    assert not ttest_significant(np.load('tmp/ttest.npy')), (
           f"{tvla} did find significant leakage, which is unexpected")


def test_aes_byte_filtering():
    project_path = TestDataPath('tvla_aes_byte/ci_opentitan_simple_aes.cwp')
    tvla = TvlaCmd(Args(['--project-file', str(project_path),
                         '--mode', 'aes', '--round-select', '0',
                         '--byte-select', '0', '--save-to-disk', '--test-type', 'SPECIFIC_BYTE',
                         'run-tvla'])).run()
    received_file = np.load('tmp/traces.npy.npz')
    traces_to_use = received_file['traces_to_use']
    assert sum(traces_to_use) <= 100, (
           f"{tvla} filtered less than 90 % of the input traces, which is unexpected.")
