#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


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


def compute_histograms_aes_byte(trace_resolution, rnd_list, byte_list, traces, leakage):
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


def compute_histograms_aes_bit(trace_resolution, rnd_list, bit_list, traces, leakage):
    """ Building histograms for AES.

    For each time sample we make two histograms, one for selected bit value = 0 (fixed set) and one
    for selected bit value = 1 (random set).
    The value stored in histograms[v][w][x][y][z] shows how many traces have value z at time y,
    given that selected bit = 0 (fixed set, x = 0) or 1 (random set, x = 1).
    """
    num_leakages = 2
    num_rnds = len(rnd_list)
    num_bits = len(bit_list)
    num_samples = traces.shape[1]
    histograms = np.zeros((num_rnds, num_bits, num_leakages, num_samples, trace_resolution),
                          dtype=np.uint32)

    for i_rnd in range(num_rnds):
        for i_bit in range(num_bits):
            for i_sample in range(num_samples):
                histograms[i_rnd, i_bit, :, i_sample, :] = np.histogram2d(
                    leakage[rnd_list[i_rnd], bit_list[i_bit], :], traces[:, i_sample],
                    bins=[range(num_leakages + 1), range(trace_resolution + 1)])[0]

    return histograms
