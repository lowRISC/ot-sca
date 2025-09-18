#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import chipwhisperer as cw
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_PATH + '/..')
import util.helpers as helpers  # noqa : E402
from capture.project_library.project import ProjectConfig  # noqa : E402
from capture.project_library.project import SCAProject  # noqa : E402
from util import check_version  # noqa : E402
from util import plot  # noqa : E402

logger = logging.getLogger()


@dataclass
class Trace:
    """ Configuration class for the current capture.
    """
    wave: list
    plaintext: list
    ciphertext: list
    key: list


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
    parser.add_argument(
        "-s",
        "--num_sigmas",
        dest="num_sigmas",
        type=float,
        required=False,
        help="Amount of tolerable deviation from average during\
                            filtering")
    parser.add_argument(
        "-m",
        "--max_traces_mem",
        dest="max_traces_mem",
        type=int,
        required=False,
        default=10000,
        help="Maximum amount of traces held in memory per process")
    parser.add_argument("-c",
                        "--num_cores",
                        dest="num_cores",
                        type=int,
                        required=False,
                        default=1,
                        help="Number of cores used for the trace alignment")
    parser.add_argument("-p",
                        "--print_num_traces",
                        dest="print_num_traces",
                        type=int,
                        required=False,
                        default=100,
                        help="Number of traces to print")
    parser.add_argument(
        "-n",
        "--num_traces_mean",
        dest="num_traces_mean",
        type=int,
        required=False,
        default=100,
        help="Number of traces used for calculating mean for the\
                            trace algining reference trace")
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


def print_traces(args, project: SCAProject, filename: Path, ref_trace=None):
    """ Print traces to file.

    Args:
        args: The command line arguments.
        project_in: The project.
        filename: The filename of the plot file.
        ref_trace: Reference trace to highlight in the plot.
    """
    metadata = project.get_metadata()
    num_traces = args.print_num_traces
    if metadata["num_traces"] < args.print_num_traces:
        num_traces = metadata["num_traces"]
    logger.info(f"Printing {str(filename)}.html...")
    plot.save_plot_to_file(project.get_waves(0, num_traces),
                           set_indices=None,
                           num_traces=num_traces,
                           outfile=filename,
                           add_mean_stddev=False,
                           ref_trace=ref_trace)


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
            traces_to_use = np.all(
                (in_traces >= min_trace) & (in_traces <= max_trace), axis=1)
            out_traces = in_traces[traces_to_use]
            out_ptx = in_ptx[traces_to_use]
            out_ctx = in_ctx[traces_to_use]
            out_k = in_k[traces_to_use]
            # Store into output project.
            for idx in range(len(out_traces)):
                project_out.append_trace(wave=out_traces[idx],
                                         plaintext=out_ptx[idx],
                                         ciphertext=out_ctx[idx],
                                         key=out_k[idx])
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


def align_traces_process(args: dict, in_traces: list, in_ptx: list,
                         in_ctx: list, in_k: list, ref_trace: Trace,
                         job_id: str):
    """ Align traces function to be distributed to multiple processes.

    Args:
        args: The command line arguments.
        in_traces: The input traces.
        in_ptx: The input plaintexts.
        in_ctx: The input ciphertexts.
        in_k: The input keys.
        ref_trace: The reference trace.
        job_id: The identifier for the current job.
    """
    # Create configuration for a temporary project. Needed to convert DB to
    # CW database as we utilize functionality from CW for the trace
    # alignment.
    tmp_project_path = str(Path(
        args.project_out).parent) + "/tmp" + job_id + ".cwp"
    tmp_project = ProjectConfig(type="cw",
                                path=tmp_project_path,
                                wave_dtype=np.uint16,
                                overwrite=True,
                                trace_threshold=args.max_traces_mem)
    cw_project = SCAProject(tmp_project)
    cw_project.create_project()
    # Convert SCAProject into ChipWhisperer project.
    # Insert reference mean trace into position 0 of the CW library.
    # This trace will be removed after processing the trace set.
    cw_project.append_trace(wave=ref_trace.wave,
                            plaintext=ref_trace.plaintext,
                            ciphertext=ref_trace.ciphertext,
                            key=ref_trace.key)
    for idx, trace in enumerate(in_traces):
        cw_project.append_trace(wave=trace,
                                plaintext=in_ptx[idx],
                                ciphertext=in_ctx[idx],
                                key=in_k[idx])

    # Align traces using CW functionality.
    resync_traces = cw.analyzer.preprocessing.ResyncSAD(cw_project.project)
    resync_traces.ref_trace = 0
    resync_traces.target_window = (args.low_window, args.high_window)
    resync_traces.max_shift = args.max_shift
    # Append traces to array.
    traces = []
    for i in range(resync_traces.num_traces()):
        if resync_traces.get_trace(i) is None:
            continue
        # Write traces back to output project.
        # Do not include reference trace that we generated before.
        if (resync_traces.get_textin(i) is not ref_trace.plaintext and
                resync_traces.get_textout(i) is not ref_trace.ciphertext and
                resync_traces.get_known_key(i) is not ref_trace.key):
            trace = Trace(wave=resync_traces.get_trace(i).astype(np.uint16),
                          plaintext=resync_traces.get_textin(i),
                          ciphertext=resync_traces.get_textout(i),
                          key=resync_traces.get_known_key(i))
            traces.append(trace)
    cw_project.close(save=False)
    in_traces = []
    in_ptx = []
    in_ctx = []
    in_k = []
    return traces


def align_traces(args: dict, project_in: SCAProject, project_out: SCAProject):
    """ Align traces.

    This function aligns all traces using a window (defined with the
    low_window and high_window command line argument) with the "Sum of
    Absolute Difference (SAD)" algorithm provided in the ChipWhisperer ResyncSAD
    function. A reference trace is used that consists of the mean of
    num_traces_mean traces.
    Args:
        args: The command line arguments.
        project_in: The project provided.
        project_out: The new project containing the filtered traces.
    """
    if args.align_enable:
        metadata = project_in.get_metadata()
        logger.info(f"Start aligning {metadata['num_traces']} traces...")

        # Calculate the mean of num_traces_mean traces and use as reference
        # trace for the aligning.
        traces_mean = project_in.get_waves(1, args.num_traces_mean)
        traces_new = np.empty((len(traces_mean), len(traces_mean[0])),
                              dtype=np.uint16)
        for i_trace in range(len(traces_mean)):
            traces_new[i_trace] = traces_mean[i_trace]
        ref_mean_wave = traces_new.mean(axis=0).astype(np.uint16)
        ref_ptx = np.zeros(len(project_in.get_plaintexts(0)), dtype=np.uint16)
        ref_ctx = np.zeros(len(project_in.get_ciphertexts(0)), dtype=np.uint16)
        ref_k = np.zeros(len(project_in.get_keys(0)), dtype=np.uint16)
        ref_trace = Trace(wave=ref_mean_wave,
                          plaintext=ref_ptx,
                          ciphertext=ref_ctx,
                          key=ref_k)
        # Iterate over the traces, keep max. max_traces_mem in memory.
        num_traces_aligned = 0
        trace_end = 0
        max_traces_mem_core = args.max_traces_mem
        if metadata["num_traces"] < args.max_traces_mem:
            max_traces_mem_core = int(metadata["num_traces"] / args.num_cores)
        max_traces_mem_total = max_traces_mem_core * args.num_cores
        for trace_it in tqdm(range(0, metadata["num_traces"],
                                   max_traces_mem_total),
                             desc="Aligning",
                             ncols=80,
                             unit=str(max_traces_mem_total) + " traces"):
            # Get current trace set.
            trace_end = trace_it + max_traces_mem_total
            in_traces = project_in.get_waves(trace_it, trace_end)
            in_ptx = project_in.get_plaintexts(trace_it, trace_end)
            in_ctx = project_in.get_ciphertexts(trace_it, trace_end)
            in_k = project_in.get_keys(trace_it, trace_end)
            # Distribute trace aligning to multiple processes.
            aligned_traces_total = Parallel(n_jobs=args.num_cores)(
                delayed(align_traces_process)
                (args, in_traces[i:i + max_traces_mem_core],
                 in_ptx[i:i +
                        max_traces_mem_core], in_ctx[i:i +
                                                     max_traces_mem_core],
                 in_k[i:i + max_traces_mem_core], ref_trace, str(trace_it + i))
                for i in range(0, trace_end, max_traces_mem_core))
            # Store aligned traces in output project.
            for align_traces in aligned_traces_total:
                for aligned_trace in align_traces:
                    project_out.append_trace(
                        wave=aligned_trace.wave,
                        plaintext=aligned_trace.plaintext,
                        ciphertext=aligned_trace.ciphertext,
                        key=aligned_trace.key)
                    num_traces_aligned += 1
            project_out.save()
            # Free memory.
            in_traces = None
            in_ptx = None
            in_ctx = None
            in_k = None
        metadata["num_traces"] = num_traces_aligned
        project_out.write_metadata(metadata)
        logger.info(f"Aligned {num_traces_aligned} traces")
        print_traces(args, project_out, "traces_aligned", ref_mean_wave)


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
    project_in_cfg = ProjectConfig(type=type,
                                   path=args.project_in,
                                   wave_dtype=np.uint16,
                                   overwrite=False,
                                   trace_threshold=args.max_traces_mem)

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
    project_out_cfg = ProjectConfig(type=type,
                                    path=args.project_out,
                                    wave_dtype=np.uint16,
                                    overwrite=True,
                                    trace_threshold=args.max_traces_mem)
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
