#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys

import chipwhisperer as cw
import numpy as np
import scipy.stats


def analyze_traces(file_proj, file_gold_proj, corr_coeff) -> bool:
    """Performs a correlation between golden and new traces.

    This function:
        - Computes the mean of the golden and new traces,
        - Computes the pearson coefficient of these means,
        - Compares the coefficient with the user provided threshold.

    Args:
        file_proj: The new Chipwhisperer project file.
        file_gold_proj: The golden Chipwhisperer project file.
        corr_coeff: User defined correlation threshold.

    Returns:
        True if trace comparison succeeds, False otherwise.
    """
    # Open the current project
    proj_curr = cw.open_project(file_proj)
    # Calculate mean of new traces
    curr_trace = np.mean(proj_curr.waves, axis=0)

    # Import the golden project
    proj_gold = cw.import_project(file_gold_proj)
    # Calculate mean of golden traces
    gold_trace = np.mean(proj_gold.waves, axis=0)

    # Pearson correlation: golden trace vs. mean of new traces
    calc_coeff = scipy.stats.pearsonr(gold_trace, curr_trace).correlation
    print(f'Correlation={round(calc_coeff,3)}')
    # Fail / pass
    if calc_coeff < corr_coeff:
        return False
    else:
        return True


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""Calculate Pearson correlation between golden
            traces and captured traces. Failes when correlation
            coefficient is below user threshold."""
    )
    parser.add_argument(
        "-f",
        "--file_proj",
        required=True,
        help="chipwhisperer project file"
    )
    parser.add_argument(
        "-g",
        "--file_gold_proj",
        required=True,
        help="chipwhisperergolden project file"
    )
    parser.add_argument(
        "-c",
        "--corr_coeff",
        type=float,
        required=True,
        help="specifies the correlation coefficient threshold"
    )
    return parser.parse_args()


def main() -> int:
    """Parses command-line arguments and TODO"""
    args = parse_args()

    if analyze_traces(**vars(args)):
        print('Traces OK.')
    else:
        print('Traces correlation below threshold.')
        sys.exit(1)


if __name__ == "__main__":
    main()
