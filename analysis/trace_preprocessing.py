#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os
import sys
from pathlib import Path

import chipwhisperer as cw
import numpy as np

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_PATH + '/..')
import util.helpers as helpers  # noqa : E402
from capture.project_library.project import ProjectConfig  # noqa : E402
from capture.project_library.project import SCAProject  # noqa : E402
from util import check_version  # noqa : E402
from util import plot  # noqa : E402

logger = logging.getLogger()


def parse_arguments(argv):
    """ Command line argument parsing.

    Args:
        argv: The command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Parse")
    parser.add_argument("-i",
                        "--project_in",
                        dest="project_in",
                        type=helpers.ap_check_file_exists,
                        required=True,
                        help="Path of the input project")
    parser.add_argument("-o",
                        "--project_out",
                        dest="project_out",
                        type=helpers.ap_check_dir_exists,
                        required=True,
                        help="Path of the output project")
    parser.add_argument("-f",
                        "--filter_enable",
                        dest="filter_enable",
                        action='store_true',
                        help="Filter the traces")
    parser.add_argument("-a",
                        "--align_enable",
                        dest="align_enable",
                        action='store_true',
                        help="Aligtn the traces")
    parser.add_argument("-s",
                        "--num_sigmas",
                        dest="num_sigmas",
                        type=float,
                        required=False,
                        help="Amount of tolerable deviation from average during\
                            filtering")
    parser.add_argument("-m",
                        "--max_traces_mem",
                        dest="max_traces_mem",
                        type=int,
                        required=False,
                        default=10000,
                        help="Maximum amount of traces held in memory")
    parser.add_argument("-p",
                        "--print_num_traces",
                        dest="print_num_traces",
                        type=int,
                        required=False,
                        default=100,
                        help="Number of traces to print")
    parser.add_argument("-r",
                        "--ref_trace",
                        dest="ref_trace",
                        type=int,
                        required=False,
                        default=100,
                        help="Reference trace for trace aligning")
    parser.add_argument("-lw",
                        "--low_window",
                        dest="low_window",
                        type=int,
                        required=False,
                        help="Window low for trace aligning")
    parser.add_argument("-hw",
                        "--high_window",
                        dest="high_window",
                        type=int,
                        required=False,
                        help="Window high for trace aligning")
    parser.add_argument("-ms",
                        "--max_shift",
                        dest="max_shift",
                        type=int,
                        required=False,
                        help="Maximum shift for trace aligning. Would more "
                             "shift be needed, the trace gets discarded")

    args = parser.parse_args(argv)

    return args


def print_traces(args, project: SCAProject, filename: Path):
    """ Print traces to file.

    Args:
        args: The command line arguments.
        project_in: The project.
        filename: The filename of the plot file.
    """
    metadata = project.get_metadata()
    num_traces = args.print_num_traces
    if metadata["num_traces"] < args.print_num_traces:
        num_traces = metadata["num_traces"]
    logger.info(f"Printing {str(filename)}.html...")
    plot.save_plot_to_file(project.get_waves(0, num_traces),
                           set_indices = None,
                           num_traces = num_traces,
                           outfile = filename,
                           add_mean_stddev=False)


def filter_traces(args: dict, project_in: SCAProject, project_out: SCAProject):
    """ Filter traces.

    This function filters all traces that contain values over the tolerable
    deviation from average (num_sigmas). Only up to max_traces_mem traces are
    held in memory.

    Args:
        args: The command line arguments.
        project_in: The project provided.
        project_out: The new project containing the filtered traces.
    """
    if args.filter_enable:
        metadata_in = project_in.get_metadata()
        logger.info(f"Start filtering {metadata_in['num_traces']} traces with "
                    f"num_sigmas={args.num_sigmas}...")
        trace_end = 0
        num_filtered_traces = 0
        max_traces_mem = args.max_traces_mem
        if metadata_in["num_traces"] < args.max_traces_mem:
            max_traces_mem = metadata_in["num_traces"]
        # Iterate over the traces, keep max. max_traces_mem in memory.
        for trace_it in range(0, metadata_in["num_traces"], max_traces_mem):
            trace_end += max_traces_mem
            # Fetch trace, plaintext, ciphertext, and key from DB.
            in_traces = np.array(project_in.get_waves(trace_it, trace_end))
            in_ptx = np.array(project_in.get_plaintexts(trace_it, trace_end))
            in_ctx = np.array(project_in.get_ciphertexts(trace_it, trace_end))
            in_k = np.array(project_in.get_keys(trace_it, trace_end))
            # Calculate min and max value for current trace set.
            mean = in_traces.mean(axis=0)
            std = in_traces.std(axis=0)
            max_trace = mean + args.num_sigmas * std
            min_trace = mean - args.num_sigmas * std
            # Filter traces.
            traces_to_use = np.zeros(len(in_traces), dtype=bool)
            traces_to_use = np.all((in_traces >= min_trace) &
                                   (in_traces <= max_trace), axis=1)
            out_traces = in_traces[traces_to_use]
            out_ptx = in_ptx[traces_to_use]
            out_ctx = in_ctx[traces_to_use]
            out_k = in_k[traces_to_use]
            # Store into output project.
            for idx in range(len(out_traces)):
                project_out.append_trace(wave = out_traces[idx],
                                         plaintext = out_ptx[idx],
                                         ciphertext = out_ctx[idx],
                                         key = out_k[idx])
                num_filtered_traces += 1
            project_out.save()
            # Free memory.
            in_traces = None
            in_ptx = None
            in_ctx = None
            in_k = None
        project_in.close(save=False)
        # Update metadata.
        metadata_out = project_out.get_metadata()
        metadata_out["num_traces"] = num_filtered_traces
        project_out.write_metadata(metadata_out)
        # Logging.
        trace_diff = metadata_in["num_traces"] - num_filtered_traces
        logger.info(f"Filtered {trace_diff} traces, new trace set contains "
                    f"{num_filtered_traces} traces")
        print_traces(args, project_out, "traces_filtered")
        return project_out
    else:
        return project_in


def align_traces(args: dict, project_in: SCAProject, project_out: SCAProject):
    """ Align traces.

    This function aligns all traces using a window (defined with the
    low_window and high_window command line argument) with the "Sum of
    Absolute Difference (SAD)" algorithm provided in the ChipWhisperer ResyncSAD
    function.
    Args:
        args: The command line arguments.
        project_in: The project provided.
        project_out: The new project containing the filtered traces.
    """
    if args.align_enable:
        metadata = project_in.get_metadata()
        logger.info(f"Start aligning {metadata['num_traces']} traces...")
        # Store the reference trace and the corresponding crypto material.
        ref_trace = project_in.get_waves(args.ref_trace)
        ref_ptx = project_in.get_plaintexts(args.ref_trace)
        ref_ctx = project_in.get_ciphertexts(args.ref_trace)
        rf_k = project_in.get_keys(args.ref_trace)
        num_traces_aligned = 0
        trace_end = 0
        max_traces_mem = args.max_traces_mem
        if metadata["num_traces"] < args.max_traces_mem:
            max_traces_mem = metadata["num_traces"]
        # Iterate over the traces, keep max. max_traces_mem in memory.
        for trace_it in range(0, metadata["num_traces"], max_traces_mem):
            trace_end += max_traces_mem
            # Convert SCAProject into ChipWhisperer project.
            cw_project = cw.common.api.ProjectFormat.Project()
            in_traces = project_in.get_waves(trace_it, trace_end)
            in_ptx = project_in.get_plaintexts(trace_it, trace_end)
            in_ctx = project_in.get_ciphertexts(trace_it, trace_end)
            in_k = project_in.get_keys(trace_it, trace_end)
            cw_project.traces.append(cw.Trace(ref_trace, ref_ptx, ref_ctx, rf_k), dtype = np.uint16)
            for idx, trace in enumerate(in_traces):
                cw_project.traces.append(cw.Trace(trace,
                                                  in_ptx[idx],
                                                  in_ctx[idx], in_k[idx]),
                                         dtype = np.uint16)
            # Align traces using CW functionality.
            resync_traces = cw.analyzer.preprocessing.ResyncSAD(cw_project)
            resync_traces.ref_trace = 0
            resync_traces.target_window = (args.low_window, args.high_window)
            resync_traces.max_shift = args.max_shift
            aligned_traces = cw.common.api.ProjectFormat.Project()
            for i in range(resync_traces.num_traces()):
                if resync_traces.get_trace(i) is None:
                    continue
                aligned_traces.traces.append(cw.Trace(resync_traces.get_trace(i),
                                                      resync_traces.get_textin(i),
                                                      resync_traces.get_textout(i),
                                                      resync_traces.get_known_key(i)),
                                             dtype=np.uint16)
            # Write traces back to output project.
            for trace in aligned_traces.traces:
                project_out.append_trace(wave = trace.wave,
                                         plaintext = trace.textin,
                                         ciphertext = trace.textout,
                                         key = trace.key)
                num_traces_aligned += 1
            cw_project.close()
            aligned_traces.close()
            project_out.save()
            # Free memory.
            in_traces = None
            in_ptx = None
            in_ctx = None
            in_k = None
        metadata["num_traces"] = num_traces_aligned
        project_out.write_metadata(metadata)
        logger.info(f"Aligned {num_traces_aligned} traces")
        print_traces(args, project_out, "traces_aligned")


def main(argv=None):
    # Configure the logger.
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    # Check the ChipWhisperer version.
    check_version.check_cw("5.7.0")

    #  Parse the provided arguments.
    args = parse_arguments(argv)

    # Open the existing project and create the new project containing the
    # filtered traces.
    type = "ot_trace_library"
    if "cwp" in str(args.project_in):
        type = "cw"
    logger.info(f"Opening DB {args.project_out}")
    project_in_cfg = ProjectConfig(type = type,
                                   path = args.project_in,
                                   wave_dtype = np.uint16,
                                   overwrite = False,
                                   trace_threshold = args.max_traces_mem)

    project_in = SCAProject(project_in_cfg)
    project_in.create_project()
    metadata_in = project_in.get_metadata()
    num_traces = metadata_in.get("num_traces")
    if num_traces is None:
        # Database does not contain num_traces in the metadata entry.
        # Open the traces, get the number of traces, and close it.
        waves = project_in.get_waves()
        num_traces = len(waves)
        waves = None
    print_traces(args, project_in, "traces_input")

    # Create output database.
    logger.info(f"Creating new DB {args.project_out}")
    project_out_cfg = ProjectConfig(type = type,
                                    path = args.project_out,
                                    wave_dtype = np.uint16,
                                    overwrite = True,
                                    trace_threshold = args.max_traces_mem)
    project_out = SCAProject(project_out_cfg)
    project_out.create_project()
    project_out.write_metadata(metadata_in)

    # Trace filtering step.
    project_in = filter_traces(args, project_in, project_out)

    # If we modified the output database, reopen it.
    if args.filter_enable:
        project_out = SCAProject(project_out_cfg)
        project_out.create_project()
        project_out.write_metadata(project_in.get_metadata())

    # Trace aligning step. If we changed the DB in the trace filtering step,
    # use these traces.
    align_traces(args, project_in, project_out)
    project_in.close(save=False)
    project_out.close(save=True)


if __name__ == "__main__":
    main()
