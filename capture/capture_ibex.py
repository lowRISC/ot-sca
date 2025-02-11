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
from target.communication.sca_prng_commands import OTPRNG
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
    test_mode: str
    num_traces: int
    num_segments: int
    protocol: str
    port: Optional[str] = "None"
    batch_prng_seed: Optional[str] = "None"


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
        usb_serial = cfg["target"].get("usb_serial"),
        interface = cfg["target"].get("interface"),
        husky_serial = cfg["husky"].get("usb_serial")
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

    # Determine if we are in batch mode or not.
    batch = False
    if "batch" in cfg["test"]["which_test"]:
        batch = True

    # Create scope config & setup scope.
    scope_cfg = ScopeConfig(
        scope_type = scope_type,
        batch_mode = batch,
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
        scope_sn = cfg[scope_type].get("usb_serial"),
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
        ot_prng: The communication interface to the PRNG SCA application.
        ot_trig: The communication interface to the SCA trigger.
    """
    # Create communication interface to OT Ibex.
    ot_ibex = OTIbex(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to OT PRNG.
    ot_prng = OTPRNG(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to SCA trigger.
    ot_trig = OTTRIGGER(target=target, protocol=capture_cfg.protocol)

    return ot_ibex, ot_prng, ot_trig


def generate_data(test_mode, num_data):
    """ Returns data used by the test.

    Either a fixed dataset or a random one is generated.

    Returns:
        data: The data set used for the test.
        data_fixed: The fixed data set.
    """
    data = []
    data_fixed = 0xABBABABE
    # First sample is always fixed.
    sample_fixed = True
    for i in range(num_data):
        if "fvsr" in test_mode:
            if sample_fixed:
                data.append(data_fixed)
            else:
                data.append(random.getrandbits(32))
            sample_fixed = random.getrandbits(32) & 0x1
        elif "random" in test_mode:
            tmp = random.getrandbits(32)
            data.append(tmp)
        else:
            raise RuntimeError("Error: Invalid test mode!")
    return data, data_fixed


def capture(scope: Scope, ot_ibex: OTIbex, ot_prng: OTPRNG,
            capture_cfg: CaptureConfig, project: SCAProject, target: Target):
    """ Capture power consumption during execution of Ibex SCA penetration tests.

    Supports the following captures:
    * ibex.sca.register_file_read: Read data from registers.
    * ibex.sca.register_file_write: Write data to registers.
    * ibex.sca.tl_read: Read data from SRAM over TL-UL.
    * ibex.sca.tl_write: Write data over TL-UL to SRAM.

    Args:
        scope: The scope class representing a scope (Husky or WaveRunner).
        ot_ibex: The communication interface to the Ibex SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
        capture_cfg: The configuration of the capture.
        project: The SCA project.
        target: The OpenTitan target.
    Returns:
        device_id: The ID of the target device.
    """
    device_id = ot_ibex.init()
    # Optimization for CW trace library.
    num_segments_storage = 1

    # Seed the PRNG used for generating random data.
    if "batch" in capture_cfg.test_mode:
        # Seed host's PRNG.
        random.seed(capture_cfg.batch_prng_seed)

        # Seed the target's PRNG.
        ot_prng.seed_prng(capture_cfg.batch_prng_seed.to_bytes(4, "little"))

    # Register ctrl-c handler to store traces on abort.
    signal.signal(signal.SIGINT, partial(abort_handler_during_loop, project))
    # Main capture with progress bar.
    remaining_num_traces = capture_cfg.num_traces
    with tqdm(total=remaining_num_traces, desc="Capturing", ncols=80, unit=" traces") as pbar:
        while remaining_num_traces > 0:
            # Arm the scope.
            scope.arm()
            if "batch" in capture_cfg.test_mode:
                num_data = capture_cfg.num_segments
            else:
                # In non-batch mode, 8 uint32 values are used.
                num_data = 8
            # Generate data set used for the test.
            data, data_fixed = generate_data(capture_cfg.test_mode, num_data)
            # Start the test based on the mode.
            if "batch" in capture_cfg.test_mode:
                if "fvsr" in capture_cfg.test_mode:
                    # In FvsR batch, the fixed dataset and the number of segments
                    # is transferred to the device. The rest of the dataset is
                    # generated on the device. Trigger is set number of segments.
                    ot_ibex.start_test(capture_cfg.test_mode, data_fixed, capture_cfg.num_segments)
                elif "random" in capture_cfg.test_mode:
                    # In Random batch, number of segments is transferred to the
                    # device. number of segments random datasets are generated
                    # on the device. Trigger is set number of segments.
                    ot_ibex.start_test(capture_cfg.test_mode,
                                       capture_cfg.num_segments)
            else:
                # In the non-batch mode, the dataset is generated in ot-sca and
                # transferred to the device. Trigger is set once.
                ot_ibex.start_test(capture_cfg.test_mode, data)

            # Capture traces.
            waves = scope.capture_and_transfer_waves(target)
            assert waves.shape[0] == capture_cfg.num_segments

            response = ot_ibex.ibex_sca_read_response()
            # Check response. 0 for non-batch and the last data element in
            # batch mode.
            if "batch" in capture_cfg.test_mode:
                assert response == data[-1]
            else:
                assert response == 0

            # Store traces.
            if "batch" in capture_cfg.test_mode:
                for i in range(capture_cfg.num_segments):
                    # Sanity check retrieved data (wave).
                    assert len(waves[i, :]) >= 1
                    # Store trace into database.
                    project.append_trace(wave = waves[i, :],
                                         plaintext = data[i].to_bytes(4, 'little'),
                                         ciphertext = None,
                                         key = None)
            else:
                # Convert data into bytearray for storage in database.
                data_bytes = []
                for d in data:
                    data_bytes.append(d.to_bytes(4, "little"))
                # Sanity check retrieved data (wave).
                assert len(waves[0, :]) >= 1
                # Store trace into database.
                project.append_trace(wave = waves[0, :],
                                     plaintext = b''.join(data_bytes),
                                     ciphertext = None,
                                     key = None)

            # Memory allocation optimization for CW trace library.
            num_segments_storage = project.optimize_capture(num_segments_storage)

            # Update the loop variable and the progress bar.
            remaining_num_traces -= capture_cfg.num_segments
            pbar.update(capture_cfg.num_segments)
    return device_id


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
    capture_cfg = CaptureConfig(test_mode = cfg["test"]["which_test"],
                                num_traces = cfg["capture"]["num_traces"],
                                num_segments = scope.scope_cfg.num_segments,
                                protocol = cfg["target"]["protocol"],
                                port = cfg["target"].get("port"),
                                batch_prng_seed = cfg["test"].get("batch_prng_seed"))
    logger.info(f"Setting up capture {capture_cfg.test_mode}...")

    # Open communication with target.
    ot_ibex, ot_prng, ot_trig = establish_communication(target, capture_cfg)

    # Configure SW trigger.
    ot_trig.select_trigger(1)

    # Capture traces.
    device_id = capture(scope, ot_ibex, ot_prng, capture_cfg, project, target)

    # Print plot.
    print_plot(project, cfg, args.project)

    # Save metadata.
    metadata = {}
    metadata["device_id"] = device_id
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
