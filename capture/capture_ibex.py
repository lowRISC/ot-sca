#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import logging
import random
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from project_library.project import ProjectConfig, SCAProject
from scopes.scope import (Scope, ScopeConfig, convert_num_cycles,
                          convert_offset_cycles, determine_sampling_rate)
from tqdm import tqdm

import util.helpers as helpers
from target.communication.sca_ibex_commands import OTIbex
from target.communication.sca_trigger_commands import OTTRIGGER
from target.targets import Target, TargetConfig
from util import check_version, plot

"""Ibex SCA capture script.

Captures power traces during different Ibex operations.

Typical usage:
>>> ./capture_ibex.py -c configs/ibex_sca_cw310.yaml -p projects/ibex_sca_capture
"""


logger = logging.getLogger()


def abort_handler_during_loop(this_project, sig, frame):
    """ Abort capture and store traces.

    Args:
        this_project: Project instance.
    """
    if this_project is not None:
        logger.info("\nHandling keyboard interrupt")
        this_project.close(save=True)
    sys.exit(0)


@dataclass
class CaptureConfig:
    """ Configuration class for the current capture.
    """
    capture_mode: str
    num_traces: int
    num_segments: int
    protocol: str
    port: Optional[str] = "None"


def setup(cfg: dict, project: Path):
    """ Setup target, scope, and project.

    Args:
        cfg: The configuration for the current experiment.
        project: The path for the project file.

    Returns:
        The target, scope, and project.
    """
    # Calculate pll_frequency of the target.
    # target_freq = pll_frequency * target_clk_mult
    # target_clk_mult is a hardcoded constant in the FPGA bitstream.
    cfg["target"]["pll_frequency"] = cfg["target"]["target_freq"] / cfg["target"]["target_clk_mult"]

    # Create target config & setup target.
    logger.info(f"Initializing target {cfg['target']['target_type']} ...")
    target_cfg = TargetConfig(
        target_type = cfg["target"]["target_type"],
        fw_bin = cfg["target"]["fw_bin"],
        protocol = cfg["target"]["protocol"],
        pll_frequency = cfg["target"]["pll_frequency"],
        bitstream = cfg["target"].get("fpga_bitstream"),
        force_program_bitstream = cfg["target"].get("force_program_bitstream"),
        baudrate = cfg["target"].get("baudrate"),
        port = cfg["target"].get("port"),
        output_len = cfg["target"].get("output_len_bytes"),
        usb_serial = cfg["target"].get("usb_serial")
    )
    target = Target(target_cfg)

    # Init scope.
    scope_type = cfg["capture"]["scope_select"]

    # Will determine sampling rate (for Husky only), if not given in cfg.
    cfg[scope_type]["sampling_rate"] = determine_sampling_rate(cfg, scope_type)
    # Will convert number of cycles into number of samples if they are not given in cfg.
    cfg[scope_type]["num_samples"] = convert_num_cycles(cfg, scope_type)
    # Will convert offset in cycles into offset in samples, if they are not given in cfg.
    cfg[scope_type]["offset_samples"] = convert_offset_cycles(cfg, scope_type)

    logger.info(f"Initializing scope {scope_type} with a sampling rate of {cfg[scope_type]['sampling_rate']}...")  # noqa: E501

    # Create scope config & setup scope.
    scope_cfg = ScopeConfig(
        scope_type = scope_type,
        batch_mode = True,
        bit = cfg[scope_type].get("bit"),
        acqu_channel = cfg[scope_type].get("channel"),
        ip = cfg[scope_type].get("waverunner_ip"),
        num_samples = cfg[scope_type]["num_samples"],
        offset_samples = cfg[scope_type]["offset_samples"],
        sampling_rate = cfg[scope_type].get("sampling_rate"),
        num_segments = cfg[scope_type].get("num_segments"),
        sparsing = cfg[scope_type].get("sparsing"),
        scope_gain = cfg[scope_type].get("scope_gain"),
        pll_frequency = cfg["target"]["pll_frequency"],
    )
    scope = Scope(scope_cfg)

    # Init project.
    project_cfg = ProjectConfig(type = cfg["capture"]["trace_db"],
                                path = project,
                                wave_dtype = np.uint16,
                                overwrite = True,
                                trace_threshold = cfg["capture"].get("trace_threshold")
                                )
    project = SCAProject(project_cfg)
    project.create_project()

    return target, scope, project


def establish_communication(target, capture_cfg: CaptureConfig):
    """ Establish communication with the target device.

    Args:
        target: The OT target.
        capture_cfg: The capture config.

    Returns:
        ot_ibex: The communication interface to the Ibex SCA application.
        ot_trig: The communication interface to the SCA trigger.
    """
    # Create communication interface to OT Ibex.
    ot_ibex = OTIbex(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to SCA trigger.
    ot_trig = OTTRIGGER(target=target, protocol=capture_cfg.protocol)

    return ot_ibex, ot_trig


def generate_data():
    """ Returns data used by the test.

    Either a fixed dataset or a random one is generated.

    Returns:
        data: Data used by the test.
    """
    fixed_data = random.randint(0, 1)
    if fixed_data:
        data = [0xDEADBEEF, 0xCDCDCDCD, 0xABADCAFE, 0x8BADF00D, 0xFDFDFDFD,
                0xA5A5A5A5, 0xABABABAB, 0xC00010FF]
    else:
        data = []
        for i in range(0, 8):
            data.append(random.randint(0, 65535))
    return data


def capture(scope: Scope, ot_ibex: OTIbex, capture_cfg: CaptureConfig,
            project: SCAProject, target: Target):
    """ Capture power consumption during execution of Ibex SCA penetration tests.

    Supports the following captures:
    * ibex.sca.register_file_read: Read data from registers.
    * ibex.sca.register_file_write: Write data to registers.
    * ibex.sca.tl_read: Read data from SRAM over TL-UL.
    * ibex.sca.tl_write: Write data over TL-UL to SRAM.

    Args:
        scope: The scope class representing a scope (Husky or WaveRunner).
        ot_ibex: The OpenTitan AES communication interface.
        capture_cfg: The configuration of the capture.
        project: The SCA project.
        target: The OpenTitan target.
    """
    ot_ibex.init()
    # Optimization for CW trace library.
    num_segments_storage = 1

    # Register ctrl-c handler to store traces on abort.
    signal.signal(signal.SIGINT, partial(abort_handler_during_loop, project))
    # Main capture with progress bar.
    remaining_num_traces = capture_cfg.num_traces
    with tqdm(total=remaining_num_traces, desc="Capturing", ncols=80, unit=" traces") as pbar:
        while remaining_num_traces > 0:
            # Arm the scope.
            scope.arm()
            data = generate_data()
            if capture_cfg.capture_mode == "ibex.sca.register_file_read":
                ot_ibex.register_file_read(capture_cfg.num_segments, data)
            elif capture_cfg.capture_mode == "ibex.sca.register_file_write":
                ot_ibex.register_file_write(capture_cfg.num_segments, data)
            elif capture_cfg.capture_mode == "ibex.sca.tl_read":
                ot_ibex.tl_read(capture_cfg.num_segments, data)
            elif capture_cfg.capture_mode == "ibex.sca.tl_write":
                ot_ibex.tl_write(capture_cfg.num_segments, data)

            # Capture traces.
            waves = scope.capture_and_transfer_waves(target)
            assert waves.shape[0] == capture_cfg.num_segments

            # Convert data into bytearray for storage in database.
            data_bytes = []
            for d in data:
                data_bytes.append(d.to_bytes(4, "little"))

            # Store traces.
            for i in range(capture_cfg.num_segments):
                # Sanity check retrieved data (wave).
                assert len(waves[i, :]) >= 1
                # Store trace into database.
                project.append_trace(wave = waves[i, :],
                                     plaintext = b''.join(data_bytes),
                                     ciphertext = None,
                                     key = None)

            # Memory allocation optimization for CW trace library.
            num_segments_storage = project.optimize_capture(num_segments_storage)

            # Update the loop variable and the progress bar.
            remaining_num_traces -= capture_cfg.num_segments
            pbar.update(capture_cfg.num_segments)


def print_plot(project: SCAProject, config: dict, file: Path) -> None:
    """ Print plot of traces.

    Printing the plot helps to adjust the scope gain and check for clipping.

    Args:
        project: The project containing the traces.
        config: The capture configuration.
        file: The output file path.
    """
    if config["capture"]["show_plot"]:
        plot.save_plot_to_file(project.get_waves(0, config["capture"]["plot_traces"]),
                               set_indices = None,
                               num_traces = config["capture"]["plot_traces"],
                               outfile = file,
                               add_mean_stddev=True)
        logger.info(f'Created plot with {config["capture"]["plot_traces"]} traces: '
                    f'{Path(str(file) + ".html").resolve()}')


def main(argv=None):
    # Configure the logger.
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    # Parse the provided arguments.
    args = helpers.parse_arguments(argv)

    # Check the ChipWhisperer version.
    check_version.check_cw("5.7.0")

    # Load configuration from file.
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Setup the target, scope and project.
    target, scope, project = setup(cfg, args.project)

    # Create capture config object.
    capture_cfg = CaptureConfig(capture_mode = cfg["test"]["which_test"],
                                num_traces = cfg["capture"]["num_traces"],
                                num_segments = scope.scope_cfg.num_segments,
                                protocol = cfg["target"]["protocol"],
                                port = cfg["target"].get("port"))
    logger.info(f"Setting up capture {capture_cfg.capture_mode}...")

    # Open communication with target.
    ot_ibex, ot_trig = establish_communication(target, capture_cfg)

    # Configure SW trigger.
    ot_trig.select_trigger(1)

    # Capture traces.
    capture(scope, ot_ibex, capture_cfg, project, target)

    # Print plot.
    print_plot(project, cfg, args.project)

    # Save metadata.
    metadata = {}
    metadata["datetime"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    metadata["cfg"] = cfg
    metadata["num_samples"] = scope.scope_cfg.num_samples
    metadata["offset_samples"] = scope.scope_cfg.offset_samples
    metadata["sampling_rate"] = scope.scope_cfg.sampling_rate
    metadata["num_traces"] = capture_cfg.num_traces
    metadata["scope_gain"] = scope.scope_cfg.scope_gain
    metadata["cfg_file"] = str(args.cfg)
    # Store bitstream information.
    metadata["fpga_bitstream_path"] = cfg["target"].get("fpga_bitstream")
    if cfg["target"].get("fpga_bitstream") is not None:
        metadata["fpga_bitstream_crc"] = helpers.file_crc(cfg["target"]["fpga_bitstream"])
    if args.save_bitstream:
        metadata["fpga_bitstream"] = helpers.get_binary_blob(cfg["target"]["fpga_bitstream"])
    # Store binary information.
    metadata["fw_bin_path"] = cfg["target"]["fw_bin"]
    metadata["fw_bin_crc"] = helpers.file_crc(cfg["target"]["fw_bin"])
    if args.save_binary:
        metadata["fw_bin"] = helpers.get_binary_blob(cfg["target"]["fw_bin"])
    # Store user provided notes.
    metadata["notes"] = args.notes
    # Store the Git hash.
    metadata["git_hash"] = helpers.get_git_hash()
    # Write metadata into project database.
    project.write_metadata(metadata)

    # Finale the capture.
    project.finalize_capture(capture_cfg.num_traces)
    # Save and close project.
    project.save()


if __name__ == "__main__":
    main()
