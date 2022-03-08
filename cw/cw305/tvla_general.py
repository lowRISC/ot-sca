#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Test Vector Leakage Assessment



Typical usage:

>>> ./tvla_general.py


"""

import argparse
import binascii
from Crypto.Cipher import AES
import numpy as np
import time
from tqdm import tqdm
import yaml
from types import SimpleNamespace
import typer
from pathlib import Path
import chipwhisperer as cw
import matplotlib.pyplot as plt 
from chipwhisperer.analyzer import aes_funcs

def bit_count(int_no):
    """Computes Hamming weight of a number."""
    c = 0
    while int_no:
        int_no &= int_no - 1
        c += 1
    return c

# A set of functions for working with histograms.
# Each distribution should be stored as two vectors x and y.
# x - A vector of  values.
# y - A vector of probabilities.


def mean_hist_xy(x, y):
    """Computes mean value of a distribution."""
    return np.dot(x, y) / sum(y)


def var_hist_xy(x, y):
    """Computes variance of a distribution."""
    mu = mean_hist_xy(x, y)
    new_x = (x - mu)**2
    return mean_hist_xy(new_x, y)


def ttest1_hist_xy(x_a, y_a, x_b, y_b):
    """
    Basic first-order t-test.
    """
    mu1 = mean_hist_xy(x_a, y_a)
    mu2 = mean_hist_xy(x_b, y_b)
    var1 = var_hist_xy(x_a, y_a)
    var2 = var_hist_xy(x_b, y_b)
    N1 = sum(y_a)
    N2 = sum(y_b)
    num = np.sqrt(var1 / N1 + var2 / N2)
    diff_flag = 0 if (mu1 == mu2) else 100
    return (mu1 - mu2) / num if num > 0 else diff_flag


def ttest_hist_xy(x_a, y_a, x_b, y_b, order):
    """ General t-test of any order.
    For more details see: Reparaz et. al. "Fast Leakage Assessment", CHES 2017.
    available at: https://eprint.iacr.org/2017/624.pdf
    """
    mu_a = mean_hist_xy(x_a, y_a)
    mu_b = mean_hist_xy(x_b, y_b)

    if order == 1:
        new_x_a = x_a
        new_x_b = x_b
    elif order == 2:
        new_x_a = (x_a - mu_a)**2
        new_x_b = (x_b - mu_b)**2
    else:
        var_a = var_hist_xy(x_a, y_a)
        var_b = var_hist_xy(x_b, y_b)
        sigma_a = np.sqrt(var_a)
        sigma_b = np.sqrt(var_b)

        if sigma_a == 0:
            new_x_a = 0
        else:
            new_x_a = (((x_a - mu_a) / sigma_a)**order)
        if sigma_b == 0:
            new_x_b = 0
        else:
            new_x_b = (((x_b - mu_b) / sigma_b)**order)

    return ttest1_hist_xy(new_x_a, y_a, new_x_b, y_b)


def compute_histograms(trace_resolution, traces, N):
    """ Building histograms.
    """
    num_samples = traces.shape[1]
    num_traces = traces.shape[0]
    histograms = np.zeros((2, num_samples, trace_resolution), dtype=np.uint32)

    for i_sample in range(num_samples):
        for i_trace in range(N):
            sample_fixed = traces[2*i_trace][i_sample]
            sample_random = traces[2*i_trace+1][i_sample]
            histograms[0][i_sample][sample_fixed]+=1;
            histograms[1][i_sample][sample_random]+=1;
    return histograms


def main():
    project_file = "./projects/opentitan_simple_sha3.cwp"
    project = cw.open_project(project_file)
    A = project.waves
    num_samples = len(project.waves[0])
    num_traces = len(project.waves)
    adc_bits = 10
    trace_resolution = 2**adc_bits
    # Set size is at most half of the number of traces
    set_size = 5000

    traces = np.empty((num_traces, num_samples), dtype=np.double)
    for i_trace in range(num_traces):
        traces[i_trace] = A[i_trace] * trace_resolution
    offset = traces.min().astype('uint16')
    traces = traces.astype('uint16') - offset


    histograms = np.zeros((2, num_samples, trace_resolution), dtype=np.uint32)
    H = compute_histograms(trace_resolution,traces,set_size)


    sample_length=H.shape[1]
    x_axis = np.arange(trace_resolution)
    result = np.zeros((4,sample_length))
    for i_samples in range(sample_length):
        fixed_set=H[0][i_samples][:]
        random_set=H[1][i_samples][:]
        for order in range(4):
            result[order][i_samples] = ttest_hist_xy(x_axis, fixed_set, x_axis, random_set,order+1)

    
    x1 = np.arange(sample_length)
    y1 = np.ones(sample_length)
    fig, axs = plt.subplots(3,sharex = True)
    axs[0].plot(x1,traces[1],"k")
    axs[1].plot(x1,result[0],"k",x1,y1*4.5,"r",x1,-y1*4.5,"r")
    axs[2].plot(x1,result[1],"k",x1,y1*4.5,"r",x1,-y1*4.5,"r")
    plt.xlabel("time [samples]")
    plt.savefig('results.png')

if __name__ == "__main__":
    main()
