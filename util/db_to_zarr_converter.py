#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import sys

import helpers
import numpy as np
import zarr
from tqdm import tqdm

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_PATH + "/..")
from capture.project_library.project import ProjectConfig  # noqa: E402
from capture.project_library.project import SCAProject  # noqa: E402

logger = logging.getLogger()


def parse_arguments(argv):
    """Command line argument parsing.

    Args:
        argv: The command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Parse")
    parser.add_argument(
        "-i",
        "--input_db",
        dest="input_db",
        type=helpers.ap_check_file_exists,
        required=True,
        help="Path of the input database",
    )
    parser.add_argument(
        "-o",
        "--output_db",
        dest="output_db",
        type=helpers.ap_check_dir_exists,
        required=True,
        help="Path of the output database",
    )
    parser.add_argument(
        "-f",
        "--db_format",
        dest="db_format",
        type=str,
        required=True,
        help="'cw' or 'ot_trace_library'",
    )
    parser.add_argument(
        "-m",
        "--max_traces_mem",
        dest="max_traces_mem",
        type=int,
        required=True,
        help="Maximum number of traces held in memory",
    )

    args = parser.parse_args(argv)

    return args


def main(argv=None):
    # Configure the logger.
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    # Parse the provided arguments.
    args = parse_arguments(argv)

    # Init project.
    project_cfg = ProjectConfig(
        type=args.db_format,
        path=args.input_db,
        wave_dtype=np.uint16,
        overwrite=False,
        trace_threshold=args.max_traces_mem,
    )
    project_in = SCAProject(project_cfg)
    project_in.open_project()
    metadata = project_in.get_metadata()
    num_traces = metadata["num_traces"]

    # Init the output project.
    compressor = None
    compressor_metadata = None
    zarr_dir = zarr.DirectoryStore(args.output_db)
    zarr_group = zarr.hierarchy.group(store=zarr_dir)
    zarr_group_tile = zarr_group.require_group("0/0")
    print(project_in.get_waves(0))
    zarr_group_tile.zeros(
        name="traces",
        shape=(0, len(project_in.get_waves(0))),
        chunks=(num_traces, len(project_in.get_waves(0))),
        dtype=np.int16,
        compressor=compressor,
    )

    zarr_group_tile.zeros(
        name="plaintext",
        shape=(0, len(project_in.get_plaintexts(0))),
        chunks=(num_traces, len(project_in.get_plaintexts(0))),
        dtype=np.uint8,
        compressor=compressor_metadata,
    )

    zarr_group_tile.zeros(
        name="ciphertext",
        shape=(0, len(project_in.get_ciphertexts(0))),
        chunks=(num_traces, len(project_in.get_ciphertexts(0))),
        dtype=np.uint8,
        compressor=compressor_metadata,
    )

    zarr_group_tile.zeros(
        name="key",
        shape=(0, len(project_in.get_keys(0))),
        chunks=(num_traces, len(project_in.get_keys(0))),
        dtype=np.uint8,
        compressor=compressor_metadata,
    )

    trace_end = 0
    for trace_it in tqdm(range(0, num_traces, args.max_traces_mem),
                         desc="Converting trace"):
        trace_end += args.max_traces_mem
        # Fetch trace, plaintext, ciphertext, and key from DB.
        in_traces = np.array(project_in.get_waves(trace_it, trace_end))
        in_ptx = np.array(project_in.get_plaintexts(trace_it, trace_end))
        in_ctx = np.array(project_in.get_ciphertexts(trace_it, trace_end))
        in_k = np.array(project_in.get_keys(trace_it, trace_end))

        zarr_group_tile["traces"].append(in_traces)
        zarr_group_tile["plaintext"].append(in_ptx)
        zarr_group_tile["ciphertext"].append(in_ctx)
        zarr_group_tile["key"].append(in_k)

    zarr_dir.close()
    project_in.close()


if __name__ == "__main__":
    main()
