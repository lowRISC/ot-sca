#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys

import numpy as np
import scipy.stats

sys.path.append("../")
from capture.project_library.project import ProjectConfig  # noqa: E402
from capture.project_library.project import SCAProject  # noqa: E402


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
    project_curr_cfg = ProjectConfig(
        type="ot_trace_library",
        path=file_proj,
        wave_dtype=np.uint16,
        overwrite=False,
        trace_threshold=10000,
    )
    proj_curr = SCAProject(project_curr_cfg)
    proj_curr.open_project()
    # Calculate mean of new traces
    curr_waves = proj_curr.get_waves()
    curr_trace = np.mean(curr_waves, axis=0)

    # Import the golden project
    project_gold_cfg = ProjectConfig(
        type="ot_trace_library",
        path=file_gold_proj,
        wave_dtype=np.uint16,
        overwrite=False,
        trace_threshold=10000,
    )
    proj_gold = SCAProject(project_gold_cfg)
    proj_gold.open_project()
    # Calculate mean of golden traces
    gold_waves = proj_gold.get_waves()
    gold_trace = np.mean(gold_waves, axis=0)

    # Pearson correlation: golden trace vs. mean of new traces
    calc_coeff = scipy.stats.pearsonr(gold_trace, curr_trace).correlation
    print(f"Correlation={round(calc_coeff, 3)}")
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
        "-f", "--file_proj", required=True, help="chipwhisperer project file"
    )
    parser.add_argument(
        "-g", "--file_gold_proj", required=True, help="chipwhisperergolden project file"
    )
    parser.add_argument(
        "-c",
        "--corr_coeff",
        type=float,
        required=True,
        help="specifies the correlation coefficient threshold",
    )
    return parser.parse_args()


def main() -> int:
    """Parses command-line arguments and TODO"""
    args = parse_args()

    if analyze_traces(**vars(args)):
        print("Traces OK.")
    else:
        print("Traces correlation below threshold.")
        sys.exit(1)


if __name__ == "__main__":
    main()
