#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy.stats import ttest_ind_from_stats

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
    return ttest_ind_from_stats(
        mu1, std1, N1, mu2, std2, N2, equal_var=False, alternative="two-sided"
    )[0]


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
            x_a_ord[i_order * num_samples: (i_order + 1) * num_samples, :] = tmp_a
            x_b_ord[i_order * num_samples: (i_order + 1) * num_samples, :] = tmp_b

    # Compute Welch's t-test for all requested orders.
    ttest = ttest1_hist_xy(x_a_ord, y_a_ord, x_b_ord, y_b_ord)

    return np.reshape(ttest, (num_orders, num_samples))
