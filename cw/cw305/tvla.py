#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Test Vector Leakage Assessment



Typical usage:

To run the analysis without loading or saving the histograms:
>>> ./tvla.py

To save histograms in the OUTPUT_FILE for later use:
>>> ./tvla.py -o OUTPUT_FILE

To load histograms from the INPUT_FILE
>>> ./tvla.py -i INPUT_FILE

"""

import argparse
import logging as log
import multiprocessing
import os
from pathlib import Path

import chipwhisperer as cw
import matplotlib.pyplot as plt
import numpy as np
from chipwhisperer.analyzer import aes_funcs
from joblib import Parallel, delayed
from scipy.stats import ttest_ind_from_stats


class UnformattedLog(object):
    def __init__(self):
        self.logger = log.getLogger()
        self.formatters = [handler.formatter for handler in self.logger.handlers]

    def __enter__(self):
        for i in range(len(self.formatters)):
            self.logger.handlers[i].setFormatter(log.Formatter())

    def __exit__(self, exc_type, exc_value, traceback):
        for i in range(len(self.formatters)):
            self.logger.handlers[i].setFormatter(self.formatters[i])


def bit_count(int_no):
    """Computes Hamming weight of a number."""
    c = 0
    while int_no:
        int_no &= int_no - 1
        c += 1
    return c


# A set of functions for working with histograms.
# The distributions are stored in two matrices x and y with dimensions (M, N) where:
# - M equals the number of time samples times the number of orders, and
# - N equals the number of values (i.e. the resolution).
# The matrices hold the following data:
# - x holds the values (all rows are the same for 1st order), and
# - y holds the probabilities (one probability distribution per row/time sample).


def mean_hist_xy(x, y):
    """
    Computes mean values for a set of distributions.

    Both x and y are (M, N) matrices, the return value is a (M, ) vector.
    """
    return np.divide(np.sum(x * y, axis=1), np.sum(y, axis=1))


def var_hist_xy(x, y, mu):
    """
    Computes variances for a set of distributions.

    This amounts to E[(X - E[X])**2].

    Both x and y are (M, N) matrices, mu is a (M, ) vector, the return value is a (M, ) vector.
    """

    # Replicate mu.
    num_values = x.shape[1]
    mu = np.transpose(np.tile(mu, (num_values, 1)))

    # Compute the variances.
    x_mu_2 = np.power(x - mu, 2)
    return mean_hist_xy(x_mu_2, y)


def ttest1_hist_xy(x_a, y_a, x_b, y_b):
    """
    Basic first-order t-test.

    Everything needs to be a matrix.
    """
    mu1 = mean_hist_xy(x_a, y_a)
    mu2 = mean_hist_xy(x_b, y_b)
    std1 = np.sqrt(var_hist_xy(x_a, y_a, mu1))
    std2 = np.sqrt(var_hist_xy(x_b, y_b, mu2))
    N1 = np.sum(y_a, axis=1)
    N2 = np.sum(y_b, axis=1)
    return ttest_ind_from_stats(mu1,
                                std1,
                                N1,
                                mu2,
                                std2,
                                N2,
                                equal_var=False,
                                alternative='two-sided')[0]


def ttest_hist_xy(x_a, y_a, x_b, y_b, num_orders):
    """
    Welch's t-test for orders 1,..., num_orders.

    For more details see: Reparaz et. al. "Fast Leakage Assessment", CHES 2017.
    available at: https://eprint.iacr.org/2017/624.pdf

    x_a and x_b are (M/num_orders, N) matrices holding the values, one value vector per row.
    y_a and y_b are (M/num_orders, N) matrices holding the distributions, one distribution per row.

    The return value is (num_orders, M/num_orders)
    """

    num_values = x_a.shape[1]
    num_samples = y_a.shape[0]

    #############
    # y_a / y_b #
    #############

    # y_a and y_b are the same for all orders and can simply be replicated along the first axis.
    y_a_ord = np.tile(y_a, (num_orders, 1))
    y_b_ord = np.tile(y_b, (num_orders, 1))

    #############
    # x_a / x_b #
    #############

    # x_a and x_b are different on a per-order basis. Start with an empty array.
    x_a_ord = np.zeros((num_samples * num_orders, num_values))
    x_b_ord = np.zeros((num_samples * num_orders, num_values))

    # Compute shareable intermediate results.
    if num_orders > 1:
        mu_a = mean_hist_xy(x_a, y_a)
        mu_b = mean_hist_xy(x_b, y_b)
    if num_orders > 2:
        var_a = var_hist_xy(x_a, y_a, mu_a)
        var_b = var_hist_xy(x_b, y_b, mu_b)
        sigma_a = np.transpose(np.tile(np.sqrt(var_a), (num_values, 1)))
        sigma_b = np.transpose(np.tile(np.sqrt(var_b), (num_values, 1)))

    # Fill in the values.
    for i_order in range(num_orders):
        if i_order == 0:
            # First order takes the values as is.
            x_a_ord[0:num_samples, :] = x_a
            x_b_ord[0:num_samples, :] = x_b
        else:
            # Second order takes the variance.
            tmp_a = x_a - np.transpose(np.tile(mu_a, (num_values, 1)))
            tmp_b = x_b - np.transpose(np.tile(mu_b, (num_values, 1)))
            if i_order > 1:
                # Higher orders take the higher order moments, and also divide by sigma.
                tmp_a = np.divide(tmp_a, sigma_a)
                tmp_b = np.divide(tmp_b, sigma_b)

            # Take the power and fill in the values.
            tmp_a = np.power(tmp_a, i_order + 1)
            tmp_b = np.power(tmp_b, i_order + 1)
            x_a_ord[i_order * num_samples:(i_order + 1) * num_samples, :] = tmp_a
            x_b_ord[i_order * num_samples:(i_order + 1) * num_samples, :] = tmp_b

    # Compute Welch's t-test for all requested orders.
    ttest = ttest1_hist_xy(x_a_ord, y_a_ord, x_b_ord, y_b_ord)

    return np.reshape(ttest, (num_orders, num_samples))


def compute_statistics(num_orders, rnd_list, byte_list, histograms, x_axis):
    """ Computing t-test statistics for a set of time samples.
    """
    num_rnds = len(rnd_list)
    num_bytes = len(byte_list)
    num_samples = histograms.shape[3]
    ttest_trace = np.zeros((num_orders, num_rnds, num_bytes, num_samples))

    # Replicate the x_axis such that x has the same dimensions as fixed_set/random_set below.
    x = np.tile(x_axis, (num_samples, 1))

    # Compute statistics.
    for i_rnd in range(num_rnds):
        for i_byte in range(num_bytes):
            # We do fixed vs. random.
            fixed_set = histograms[i_rnd, i_byte, 0, :, :]
            random_set = np.sum(histograms[i_rnd, i_byte, 1:, :, :], 0)
            if not np.any(fixed_set != 0.0) or not np.any(random_set != 0.0):
                # In case any of the sets is empty, the statistics can't be computed. This can
                # happen if for example:
                # - Few traces are used only.
                # - The hamming distance is used as sensitive variable and the initial round is
                #   analyzed. Then the hamming distance can only be zero (fixed_set) or non-zero
                #   (random_set) if the corresponding key byte is zero or non-zero, respectively.
                #   Thus, either of the sets must be empty.
                # We return NaN and handle it when checking all results.
                ttest_trace[:, i_rnd, i_byte, :] = np.nan
                continue
            tmp = ttest_hist_xy(x, fixed_set, x, random_set, num_orders)
            ttest_trace[:, i_rnd, i_byte, :] = tmp

    return ttest_trace


def compute_histograms_general(trace_resolution, traces, leakage):
    """ Building histograms for general fixed-vs-random TVLA.

    For each time sample we make two histograms, one for the fixed and one for the random group.
    Whether a trace belongs to the fixed or random group is indicated in the leakage input
    variable. The value stored in histograms[v][w][x][y][z] shows how many traces have value z at
    time y, given that trace is in the fixed (x = 1) or random (x = 0) group. The v and w indices
    are not used but we keep them for code compatiblitly with non-general AES TVLA.
    """
    num_leakages = 2
    num_rnds = 1
    num_bytes = 1
    num_samples = traces.shape[1]
    histograms = np.zeros((num_rnds, num_bytes, num_leakages, num_samples, trace_resolution),
                          dtype=np.uint32)

    for i_sample in range(num_samples):
        histograms[0, 0, :, i_sample, :] = np.histogram2d(
            leakage, traces[:, i_sample],
            bins=[range(num_leakages + 1), range(trace_resolution + 1)])[0]

    return histograms


def compute_histograms_aes(trace_resolution, rnd_list, byte_list, traces, leakage):
    """ Building histograms for AES.

    For each time sample we make two histograms, one for Hamming weight of the sensitive variable
    = 0 (fixed set) and one for Hamming weight > 0 (random set). The value stored in
    histograms[v][w][x][y][z] shows how many traces have value z at time y, given that
    HW(state byte w in AES round v) = 0 (fixed set, x = 0) or > 0 (random set, x = 1).
    """
    num_leakages = 2
    num_rnds = len(rnd_list)
    num_bytes = len(byte_list)
    num_samples = traces.shape[1]
    histograms = np.zeros((num_rnds, num_bytes, num_leakages, num_samples, trace_resolution),
                          dtype=np.uint32)

    for i_rnd in range(num_rnds):
        for i_byte in range(num_bytes):
            for i_sample in range(num_samples):
                histograms[i_rnd, i_byte, :, i_sample, :] = np.histogram2d(
                    leakage[rnd_list[i_rnd], byte_list[i_byte], :], traces[:, i_sample],
                    bins=[np.append(range(num_leakages), 9), range(trace_resolution + 1)])[0]

    return histograms


def compute_leakage_aes(keys, plaintexts, leakage_model):
    """
    Sensitive variable is always byte-sized.

    Two leakage models are available:
    HAMMING_WEIGHT - based on the hamming weight of the state register byte.
    HAMMING_DISTANCE - based on the hamming distance between the curent and previous state
                       for a specified byte.
    """
    num_traces = len(keys)
    leakage = np.zeros((11, 16, num_traces), dtype=np.uint8)

    # Checks if all keys in the list are the same.
    key_fixed = np.all(keys == keys[0])
    subkey = np.zeros((11, 16))

    if key_fixed:
        for j in range(11):
            subkey[j] = np.asarray(
                aes_funcs.key_schedule_rounds(keys[0], 0, j))
        subkey = subkey.astype(int)

    for i in range(num_traces):

        if not key_fixed:
            for j in range(11):
                subkey[j] = np.asarray(
                    aes_funcs.key_schedule_rounds(keys[i], 0, j))
            subkey = subkey.astype(int)

        # Init
        state = plaintexts[i]

        # Round 0
        old_state = state
        state = np.bitwise_xor(state, subkey[0])
        for k in range(16):
            if leakage_model == 'HAMMING_DISTANCE':
                leakage[0][k][i] = bit_count(
                    np.bitwise_xor(state[k], old_state[k]))
            else:
                leakage[0][k][i] = bit_count(state[k])

        # Round 1 - 10
        for j in range(1, 11):
            old_state = state
            state = aes_funcs.subbytes(state)
            state = aes_funcs.shiftrows(state)
            if (j < 10):
                state = aes_funcs.mixcolumns(state)
            state = np.bitwise_xor(state, subkey[j])
            for k in range(16):
                if leakage_model == 'HAMMING_DISTANCE':
                    leakage[j][k][i] = bit_count(
                        np.bitwise_xor(state[k], old_state[k]))
                else:
                    leakage[j][k][i] = bit_count(state[k])

    return leakage


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""A histogram-based TVLA described in "Fast Leakage Assessment" by O. Reparaz,
        B. Gierlichs and I. Verbauwhede (https://eprint.iacr.org/2017/624.pdf)."""
    )

    parser.add_argument(
        "-p",
        "--project-file",
        default="projects/opentitan_simple_aes.cwp",
        help="""Name of the ChipWhisperer project file to use. Not required. If not provided,
        projects/opentitan_simple_aes.cwp is used.""",
    )
    parser.add_argument(
        "-t",
        "--trace-file",
        help="""Name of the trace file containing the numpy array with all traces in 16-bit integer
        format. Not required. If not provided, the data from the ChipWhisperer project file
        is used. Ignored for number-of-steps > 1.""",
    )
    parser.add_argument(
        "-s",
        "--trace-start",
        help="""Index of the first trace to use. Not required. If not provided, starts at the first
        trace.""",
    )
    parser.add_argument(
        "-e",
        "--trace-end",
        help="""Index of the last trace to use. Not required. If not provided, ends at the last
        trace.""",
    )
    parser.add_argument(
        "-l",
        "--leakage-file",
        help="""Name of the leakage file containing the numpy array with the leakage model for all
        rounds, all bytes, and all traces. Not required. If not provided, the leakage is computed
        from the data in the ChipWhisperer project file. Ignored for number-of-steps > 1.""",
    )
    parser.add_argument(
        "-d",
        "--save-to-disk",
        action="store_true",
        default=False,
        help="""Save trace, leakage and t-test files to disk. Ignored for trace and leakage files
        when number-of-steps > 1.""",
    )
    parser.add_argument(
        "-r",
        "--round-select",
        help="""Index of the AES round for which the histograms are to be computed: 0-10. Not
        required. If not provided, the histograms for all AES rounds are computed.""",
    )
    parser.add_argument(
        "-b",
        "--byte-select",
        help="""Index of the AES state byte for which the histograms are to be computed: 0-15. Not
        required. If not provided, the histograms for all AES state bytes are computed.""",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        help="""Name of the input file containing the histograms. Not required. If both -i and -o
        are provided, the input file is appended with more data to produce the output file.""",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="""Name of the output file to store generated histograms. Not required. If both -i and
        -o are provided, the input file is appended with more data to produce the output file.""",
    )
    parser.add_argument(
        "-n",
        "--number-of-steps",
        type=int,
        default="1",
        help="""Number of steps to breakdown the analysis into. For every step, traces are
        separately filtered and the leakage is computed. The histograms are appended to the
        ones of the previous step. This is useful when operating on very large trace sets and/or
        when analyzing how results change with the number of traces used.""",
    )
    parser.add_argument(
        "-a",
        "--ttest-step-file",
        help="""Name of the t-test step file containing one t-test analysis per step. Not
        required. If not provided, the data is recomputed.""",
    )
    parser.add_argument(
        "-f",
        "--plot-figures",
        action="store_true",
        default=False,
        help="""Plot figures and save them to disk. Not required.""",
    )
    parser.add_argument(
        "-g",
        "--general-test",
        action="store_true",
        default=False,
        help="""Perform general fixed-vs-random TVLA without leakage model. Odd traces are grouped
        in the fixed set while even traces are grouped in the random set. Not required.""",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="aes",
        help="""Select mode: can be either "aes" or "sha3". Not required. If not provided or if a
        another string is provided, "aes" is used.""",
    )

    return parser.parse_args()


def main():

    Path("tmp").mkdir(exist_ok=True)
    log_format = "%(asctime)s %(levelname)s: %(message)s"
    log.basicConfig(format=log_format,
                    datefmt="%Y-%m-%d %I:%M:%S",
                    handlers=[
                        log.FileHandler("tmp/log.txt"),
                        log.StreamHandler()
                    ],
                    level=log.INFO,
                    force=True,)

    args = parse_args()

    if args.mode != "sha3" and args.mode != "aes":
        log.info("Unsupported mode {args.mode}, falling back to \"aes\"")

    if args.mode == "sha3" or args.general_test is True:
        general_test = True
    else:
        general_test = False

    if args.mode == "sha3" or general_test is True:
        # We don't care about the round select in this mode. Set it to 0 for code compatibility.
        rnd_list = [0]
    elif args.round_select is None:
        rnd_list = list(range(11))
    else:
        rnd_list = [int(args.round_select)]
    assert all(rnd >= 0 and rnd < 11 for rnd in rnd_list)

    num_rnds = len(rnd_list)

    if args.mode == "sha3" or general_test is True:
        # We don't care about the byte select in this mode. Set it to 0 for code compatibility.
        byte_list = [0]
    elif args.byte_select is None:
        byte_list = list(range(16))
    else:
        byte_list = [int(args.byte_select)]
    assert all(byte >= 0 and byte < 16 for byte in byte_list)

    num_bytes = len(byte_list)

    num_steps = int(args.number_of_steps)
    assert num_steps >= 1

    save_to_disk_trace = args.save_to_disk
    save_to_disk_leakage = args.save_to_disk
    save_to_disk_ttest = args.save_to_disk

    # Step-wise processing isn't compatible with a couple of other arguments.
    if num_steps > 1:
        args.trace_file = None
        args.leakage_file = None
        save_to_disk_trace = False
        save_to_disk_leakage = False

    if args.input_file is not None:
        # Load previously generated histograms.
        histograms_file = np.load(args.input_file)
        histograms_in = histograms_file['histograms']
        num_samples = histograms_in.shape[3]
        trace_resolution = histograms_in.shape[4]
        # If previously generated histograms are loaded, the rounds and bytes of interest must
        # match. Otherwise, indices would get mixed up.
        assert rnd_list == histograms_file['rnd_list']
        assert byte_list == histograms_file['byte_list']

    if (args.input_file is None or args.output_file is not None) and args.ttest_step_file is None:
        # Either don't have previously generated histograms or we need to append previously
        # generated histograms.

        # Make sure the project file is compatible with the previously generated histograms.
        project = cw.open_project(args.project_file)
        if args.input_file is None:
            num_samples = len(project.waves[0])
        else:
            assert num_samples == len(project.waves[0])

        if args.input_file is None:
            adc_bits = 12
            trace_resolution = 2**adc_bits

        # When doing general fixed-vs-random TVLA, the first trace is using the fixed key.
        if general_test is True:
            fixed_key = np.copy(project.keys[0])

        # Amount of tolerable deviation from average during filtering.
        num_sigmas = 3.5

        # Overall number of traces, trace start and end indices.
        num_traces_tot = len(project.waves)
        if args.trace_start is None:
            trace_start_tot = 0
        else:
            trace_start_tot = int(args.trace_start)
        if args.trace_end is None:
            trace_end_tot = num_traces_tot - 1
        else:
            trace_end_tot = int(args.trace_end)
        assert trace_end_tot - trace_start_tot < num_traces_tot
        num_traces_tot = trace_end_tot - trace_start_tot + 1

        # Generate indices for step-wise processing.
        num_traces_vec = []
        trace_start_vec = []
        trace_end_vec = []
        num_traces_step = num_traces_tot // num_steps
        num_traces_rem = num_traces_tot % num_steps
        for i_step in range(num_steps):
            trace_start_vec.append(trace_start_tot + i_step * num_traces_step)
            if i_step < num_steps - 1 or num_traces_rem == 0:
                num_traces_vec.append(num_traces_step)
                trace_end_vec.append((i_step + 1) * num_traces_step - 1)
            else:
                num_traces_vec.append(num_traces_step + num_traces_rem)
                trace_end_vec.append(trace_end_tot)

        # The number of parallel jobs to use for the processing-heavy tasks.
        num_jobs = multiprocessing.cpu_count()

        # The number of traces/samples processed by each parallel job at a time.
        trace_step_leakage = min(10000, num_traces_step // num_jobs)
        sample_step_hist = 1
        # Increase work per thread to amortize parallelization overhead.
        if len(rnd_list) == 1 and len(byte_list) == 1:
            if general_test is True:
                sample_step_hist = min(10000, num_samples // num_jobs)
            else:
                sample_step_hist = 5

        for i_step in range(num_steps):
            num_traces = num_traces_vec[i_step]
            trace_start = trace_start_vec[i_step]
            trace_end = trace_end_vec[i_step]

            log.info("Processing Step %i/%i: Trace %i - %i",
                     i_step + 1, num_steps, trace_start, trace_end)

            if args.trace_file is None:

                # Make sure to re-open the project file as we close it during the operation to free
                # up some memory.
                if i_step > 0:
                    project = cw.open_project(args.project_file)

                # Converting traces from floating point to integer and creating a dense copy.
                log.info("Converting Traces")
                if project.waves[0].dtype == 'uint16':
                    traces = np.empty((num_traces, num_samples), dtype=np.uint16)
                    for i_trace in range(num_traces):
                        traces[i_trace] = project.waves[i_trace + trace_start]
                else:
                    traces = np.empty((num_traces, num_samples), dtype=np.double)
                    for i_trace in range(num_traces):
                        traces[i_trace] = (project.waves[i_trace +
                                                         trace_start] + 0.5) * trace_resolution
                    traces = traces.astype('uint16')

                if general_test is False:
                    # Filter out noisy traces.
                    log.info("Filtering Traces")

                    # Get the mean and standard deviation.
                    mean = traces.mean(axis=0)
                    std = traces.std(axis=0)

                    # Define upper and lower limits.
                    max_trace = mean + num_sigmas * std
                    min_trace = mean - num_sigmas * std

                    # Filtering of converted traces (len = num_samples). traces_to_use itself can be
                    # used to index the entire project file (len >= num_samples).
                    traces_to_use = np.zeros(len(project.waves), dtype=bool)
                    traces_to_use[trace_start:trace_end + 1] = np.all((traces >= min_trace) &
                                                                      (traces <= max_trace), axis=1)
                    traces = traces[traces_to_use[trace_start:trace_end + 1]]
                else:
                    # For now, don't perform any filtering when doing general fixed-vs-random TVLA.
                    traces_to_use = np.zeros(len(project.waves), dtype=bool)
                    traces_to_use[trace_start:trace_end + 1] = True

                    if i_step == 0:
                        # Keep a single trace to create the figures.
                        single_trace = traces[1]

                if save_to_disk_trace:
                    log.info("Saving Traces")
                    np.savez('tmp/traces.npy', traces=traces, traces_to_use=traces_to_use,
                             trace_start=trace_start, trace_end=trace_end)

                if ((save_to_disk_trace is True or save_to_disk_ttest is True) and
                        general_test is True and i_step == 0):
                    np.save('tmp/single_trace.npy', single_trace)

            else:
                trace_file = np.load(args.trace_file)
                traces = trace_file['traces']
                traces_to_use = trace_file['traces_to_use']
                assert num_samples == traces.shape[1]
                # If a trace range is specified, it must match the range in the trace file.
                # Otherwise, we might end up using a leakage model that doesn't match the actual
                # traces.
                if args.trace_start is None:
                    trace_start = trace_file['trace_start']
                assert trace_start == trace_file['trace_start']
                if args.trace_end is None:
                    trace_end = trace_file['trace_end']
                assert trace_end == trace_file['trace_end']
                num_traces = trace_end - trace_start + 1
                # The project file must match the trace file.
                assert len(project.waves) == len(traces_to_use)

            # Correct num_traces based on filtering.
            num_traces_orig = num_traces
            num_traces = np.sum(traces_to_use)
            log.info(
                f"Will use {num_traces} traces "
                f"({100*num_traces/num_traces_orig:.1f}%)"
            )

            if args.leakage_file is None:
                # Create local, dense copies of keys and plaintexts. This allows the leakage
                # computation to be parallelized.
                keys = np.empty((num_traces_orig, 16), dtype=np.uint8)

                if general_test is False:
                    keys[:] = project.keys[trace_start:trace_end + 1]
                else:
                    # Existing KMAC trace sets use a mix of bytes strings and ChipWhisperer byte
                    # arrays. For compatiblity, we need to convert everything to numpy arrays.
                    # Eventually, we can drop this.
                    if i_step == 0:
                        # Convert all keys from the project file to numpy arrays once.
                        keys_nparrays = []
                        for i in range(num_traces_tot):
                            keys_nparrays.append(np.frombuffer(project.keys[i], dtype=np.uint8))

                        # In addition, for some existing trace sets the fixed key is used for the
                        # second instead of the first trace. For compatibility, compare a couple of
                        # keys and then select the fixed one. Eventually, we can drop this.
                        for i_key in range(10):
                            fixed_key = keys_nparrays[i_key]
                            num_hits = 0
                            for i in range(10):
                                num_hits += np.array_equal(fixed_key, keys_nparrays[i])
                            if num_hits > 1:
                                break

                    # Select the correct slice of keys for each step.
                    keys[:] = keys_nparrays[trace_start:trace_end + 1]

                # Only select traces to use.
                keys = keys[traces_to_use[trace_start:trace_end + 1]]
                if general_test is False:
                    # The plaintexts are only required for non-general AES TVLA.
                    plaintexts = np.empty((num_traces_orig, 16), dtype=np.uint8)
                    plaintexts[:] = project.textins[trace_start:trace_end + 1]
                    plaintexts = plaintexts[traces_to_use[trace_start:trace_end + 1]]

            # We don't need the project file anymore after this point. Close it together with all
            # trace files opened in the background.
            project.close(save=False)

            if general_test is False:
                # Compute or load prevsiously computed leakage model.
                if args.leakage_file is None:
                    # leakage models: HAMMING_WEIGHT (default), HAMMING_DISTANCE
                    log.info("Computing Leakage")
                    leakage = Parallel(n_jobs=num_jobs)(
                        delayed(compute_leakage_aes)(keys[i:i + trace_step_leakage],
                                                     plaintexts[i:i + trace_step_leakage],
                                                     'HAMMING_WEIGHT')
                        for i in range(0, num_traces, trace_step_leakage))
                    leakage = np.concatenate((leakage[:]), axis=2)
                    if save_to_disk_leakage:
                        log.info("Saving Leakage")
                        np.save('tmp/leakage.npy', leakage)
                else:
                    leakage = np.load(args.leakage_file)
                    assert num_traces == leakage.shape[2]
            else:
                log.info("Computing Leakage")
                # We do general fixed-vs-random TVLA. The "leakage" is indicating whether a trace
                # belongs to the fixed (1) or random (0) group.
                leakage = np.zeros((num_traces), dtype=np.uint8)
                for i in range(num_traces):
                    leakage[i] = np.array_equal(fixed_key, keys[i])

            log.info("Building Histograms")
            if general_test is False:
                # For every time sample we make two histograms, one for Hamming weight of the
                # sensitive variable = 0 (fixed set) and one for Hamming weight > 0 (random set).
                # histograms has dimensions [num_rnds, num_bytes, 2, num_samples, trace_resolution]
                # The value stored in histograms[v][w][x][y][z] shows how many traces have value z
                # at sample y, given that HW(state byte w in AES round v) = 0 (fixed set, x = 0) or
                # > 0 (random set, x = 1).
                # The computation is parallelized over the samples.
                histograms = Parallel(n_jobs=num_jobs)(
                    delayed(compute_histograms_aes)(trace_resolution, rnd_list, byte_list,
                                                    traces[:, i:i + sample_step_hist], leakage)
                    for i in range(0, num_samples, sample_step_hist))
                histograms = np.concatenate((histograms[:]), axis=3)
            else:
                # For every time sample we make 2 histograms, one for the fixed set and one for the
                # random set.
                # histograms has dimensions [0, 0, 2, num_samples, trace_resolution]
                # The value stored in histograms[v][w][x][y][z] shows how many traces have value z
                # at time y, given that trace is in the fixed (x = 0) or random (x = 1) group. The
                # v and w indices are not used but we keep them for code compatiblitly with
                # non-general AES TVLA.
                histograms = Parallel(n_jobs=num_jobs)(
                    delayed(compute_histograms_general)(trace_resolution,
                                                        traces[:, i:i + sample_step_hist],
                                                        leakage)
                    for i in range(0, num_samples, sample_step_hist))
                histograms = np.concatenate((histograms[:]), axis=3)

            # Add up new data to potential, previously generated histograms.
            if args.input_file is not None or i_step > 0:
                histograms = histograms + histograms_in

            # Move current histograms to temp variable for next step.
            if num_steps > 1 and i_step < num_steps - 1:
                histograms_in = histograms

            # Histograms can be saved for later use if output file name is passed.
            if args.output_file is not None:
                log.info("Saving Histograms")
                np.savez(args.output_file, histograms=histograms, rnd_list=rnd_list,
                         byte_list=byte_list)

            # Computing the t-test statistics vs. time.
            log.info("Computing T-test Statistics")

            # The number of samples processed by each parallel job at a time.
            sample_step_ttest = num_samples // num_jobs

            # By default, the first two moments are computed. This can be modified to any order.
            num_orders = 2

            x_axis = np.arange(trace_resolution)

            # Compute statistics.
            # ttest_trace has dimensions [num_orders, num_rnds, num_bytes, num_samples].
            ttest_trace = Parallel(n_jobs=num_jobs)(
                delayed(compute_statistics)(num_orders, rnd_list, byte_list,
                                            histograms[:, :, :, i:i + sample_step_ttest, :],
                                            x_axis)
                for i in range(0, num_samples, sample_step_ttest))
            ttest_trace = np.concatenate((ttest_trace[:]), axis=3)

            # Building the t-test statistics vs. number of traces used. ttest_step has dimensions
            # [num_orders, num_rnds, num_bytes, num_samples, num_steps], i.e., for every order,
            # every round, every byte, every sample and every step, we track the t-test value.
            log.info("Updating T-test Statistics vs. Number of Traces")
            if i_step == 0:
                ttest_step = np.empty((num_orders, num_rnds, num_bytes, num_samples,
                                       num_steps))
            ttest_step[:, :, :, :, i_step] = ttest_trace

        if save_to_disk_ttest:
            log.info("Saving T-test Step")
            np.savez('tmp/ttest-step.npy',
                     ttest_step=ttest_step,
                     trace_end_vec=trace_end_vec,
                     rnd_list=rnd_list,
                     byte_list=byte_list)

        rnd_ext = list(range(num_rnds))
        byte_ext = list(range(num_bytes))

    elif args.ttest_step_file is not None:
        # Load previously generated t-test results.
        ttest_step_file = np.load(args.ttest_step_file)
        ttest_step = ttest_step_file['ttest_step']
        num_orders = ttest_step.shape[0]
        num_samples = ttest_step.shape[3]
        num_steps = ttest_step.shape[4]
        trace_end_vec = ttest_step_file['trace_end_vec']
        # The rounds and bytes of interests must be available in the previously generated t-test
        # results. In addition, we may need to translate indices to extract the right portion of
        # of the loaded results.
        rnd_ext = np.zeros((num_rnds), dtype=np.uint8)
        byte_ext = np.zeros((num_bytes), dtype=np.uint8)
        for i_rnd in range(num_rnds):
            assert rnd_list[i_rnd] in ttest_step_file['rnd_list']
            rnd_ext[i_rnd] = np.where(ttest_step_file['rnd_list'] == rnd_list[i_rnd])[0][0]
        for i_byte in range(num_bytes):
            assert byte_list[i_byte] in ttest_step_file['byte_list']
            byte_ext[i_byte] = np.where(ttest_step_file['byte_list'] == byte_list[i_byte])[0][0]

        # Plot the t-test vs. time figures for the maximum number of traces.
        ttest_trace = ttest_step[:, :, :, :, num_steps - 1]

        if general_test is True:
            single_trace_file = os.path.dirname(args.ttest_step_file)
            single_trace_file += "/" if single_trace_file else ""
            single_trace_file += "single_trace.npy"
            single_trace = np.load(single_trace_file)
            assert num_samples == single_trace.shape[0]

    # Check ttest results.
    threshold = 4.5
    failure = np.any(np.abs(ttest_trace) >= threshold, axis=3)
    nan = np.isnan(np.sum(ttest_trace, axis=3))

    if not np.any(failure):
        log.info("No leakage above threshold identified.")
    if np.any(failure) or np.any(nan):
        if general_test is False:
            if np.any(failure):
                log.info("Leakage above threshold identified in the following order(s), round(s) "
                         "and byte(s) marked with X:")
            if np.any(nan):
                log.info("Couldn't compute statistics for order(s), round(s) and byte(s) marked "
                         "with O:")
            with UnformattedLog():
                byte_str = "Byte     |"
                dash_str = "----------"
                for i_byte in range(num_bytes):
                    byte_str += str(byte_list[i_byte]).rjust(5)
                    dash_str += "-----"

                for i_order in range(num_orders):
                    log.info(f"Order {i_order + 1}:")
                    log.info(f"{byte_str}")
                    log.info(f"{dash_str}")
                    for i_rnd in range(num_rnds):
                        result_str = "Round " + str(rnd_list[i_rnd]).rjust(2) + " |"
                        for i_byte in range(num_bytes):
                            if failure[i_order, rnd_ext[i_rnd], byte_ext[i_byte]]:
                                result_str += str("X").rjust(5)
                            elif nan[i_order, rnd_ext[i_rnd], byte_ext[i_byte]]:
                                result_str += str("O").rjust(5)
                            else:
                                result_str += "     "
                        log.info(f"{result_str}")
                    log.info("")
        else:
            log.info("Leakage above threshold identified in the following order(s) marked with X")
            if np.any(nan):
                log.info("Couldn't compute statistics for order(s) marked with O:")
            with UnformattedLog():
                for i_order in range(num_orders):
                    result_str = "Order " + str(i_order + 1) + ": "
                    if failure[i_order, 0, 0]:
                        result_str += "X"
                    elif nan[i_order, 0, 0]:
                        result_str += "O"
                    else:
                        result_str += " "
                    log.info(f"{result_str}")
                log.info("")

    if args.plot_figures:
        log.info("Plotting Figures to tmp/figures")
        Path("tmp/figures").mkdir(exist_ok=True)

        # Plotting figures for t-test statistics vs. time.
        log.info("Plotting T-test Statistics vs. Time.")
        if args.mode == "aes" and general_test is False:
            # By default the figures are saved under tmp/t_test_round_x_byte_y.png.
            for i_rnd in range(num_rnds):
                for i_byte in range(num_bytes):

                    c = np.ones(num_samples)
                    fig, axs = plt.subplots(1, num_orders, figsize=(16, 5), sharey=True)

                    for i_order in range(num_orders):
                        axs[i_order].plot(ttest_trace[i_order, rnd_ext[i_rnd], byte_ext[i_byte]],
                                          'k')
                        axs[i_order].plot(c * threshold, 'r')
                        axs[i_order].plot(-threshold * c, 'r')
                        axs[i_order].set_xlabel('time')
                        axs[i_order].set_ylabel('t-test ' + str(i_order + 1))

                    filename = "aes_t_test_round_" + str(rnd_list[i_rnd])
                    filename += "_byte_" + str(byte_list[i_byte]) + ".png"
                    plt.savefig("tmp/figures/" + filename)
                    if num_rnds == 1 and num_bytes == 1:
                        plt.show()
                    else:
                        plt.close()
        else:
            #
            c = np.ones(num_samples)
            fig, axs = plt.subplots(3, sharex=True)
            axs[0].plot(single_trace, "k")
            for i_order in range(num_orders):
                axs[1 + i_order].plot(ttest_trace[i_order, 0, 0], "k")
                axs[1 + i_order].plot(c * threshold, "r")
                axs[1 + i_order].plot(-threshold * c, "r")
                axs[1 + i_order].set_ylabel('t-test ' + str(i_order + 1))
            plt.xlabel("time [samples]")
            plt.savefig('tmp/figures/' + args.mode + '_fixed_vs_random.png')
            plt.show()

        # Plotting figures for t-test statistics vs. number of traces used.
        # For now, do a single figure per round and per order. Every line corresponds to the t-test
        # result of one time sample for one byte and one round.
        if num_steps > 1:

            log.info("Plotting T-test Statistics vs. Number of Traces, this may take a while.")
            xticks = [np.around(trace_end / 100000) for trace_end in trace_end_vec]
            xticklabels = [str(int(tick)) for tick in xticks]

            # Empty every second label if we got more than 10 steps.
            if num_steps > 10:
                for i_step in range(num_steps):
                    xticklabels[i_step] = "" if (i_step % 2 == 0) else xticklabels[i_step]

            for i_rnd in range(num_rnds):

                c = np.ones(num_steps)
                fig, axs = plt.subplots(1, num_orders, figsize=(16, 5), sharey=True)

                # To reduce the number of lines in the plot, we only plot those samples where
                # leakage is expected in the first place. This might need tuning if the design
                # is altered.

                if general_test is False:
                    # Each regular round lasts for 100 samples.
                    samples_per_rnd = 100
                    # We have a negative trigger offset of 20 samples. The initial key and data
                    # loading takes another 20 samples, the initial round lasts for 100 samples.
                    # Then center the window around the middle of the round. The effective
                    # numbers are best tuned by doing a capture with masking switched off.
                    rnd_offset = 150 + samples_per_rnd // 2
                    # The widnow width is 100 samples + 40 samples extended on each side.
                    half_window = samples_per_rnd // 2 + 40

                    samples = range(max(rnd_offset + (rnd_list[i_rnd] * samples_per_rnd) -
                                        half_window, 0),
                                    min(rnd_offset + (rnd_list[i_rnd] * samples_per_rnd) +
                                        half_window, num_samples))

                else:
                    if args.mode == "aes":
                        # Simply plot everything.
                        samples = range(0, num_samples)
                    else:
                        # For now, let's focus on the key absorption only.
                        samples = range(520, 2460)
                        # Numbers for the eval branch:

                for i_order in range(num_orders):
                    for i_byte in range(num_bytes):
                        for i_sample in samples:
                            axs[i_order].plot(ttest_step[i_order,
                                                         rnd_ext[i_rnd],
                                                         byte_ext[i_byte],
                                                         i_sample],
                                              'k')
                            axs[i_order].plot(c * threshold, 'r')
                            axs[i_order].plot(-threshold * c, 'r')
                            axs[i_order].set_xlabel('number of traces [100k]')
                            axs[i_order].set_xticks(range(num_steps))
                            axs[i_order].set_xticklabels(xticklabels)
                            axs[i_order].set_ylabel('t-test ' + str(i_order + 1) +
                                                    "\nfor samples " + str(samples[0]) +
                                                    ' to ' + str(samples[-1]))

                filename = args.mode + "_t_test_steps_round_" + str(rnd_list[i_rnd]) + ".png"
                plt.savefig("tmp/figures/" + filename)
                if num_rnds == 1:
                    plt.show()
                else:
                    plt.close()


if __name__ == "__main__":
    main()
