#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Note: The word ciphertext refers to the tag in hmac
#       To be compatible to the other capture scripts, the variable is
#       called ciphertext

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
from Crypto.Hash import HMAC, SHA256
from project_library.project import ProjectConfig, SCAProject
from scopes.scope import (Scope, ScopeConfig, convert_num_cycles,
                          convert_offset_cycles, determine_sampling_rate)
from tqdm import tqdm

import util.helpers as helpers
from target.communication.sca_hmac_commands import OTHMAC
from target.communication.sca_prng_commands import OTPRNG
from target.targets import Target, TargetConfig
from util import check_version, plot

"""HMAC SCA capture script.

Captures power traces during HMAC operations.

The data format of the crypto material (ciphertext, plaintext, and key) inside
the script is stored in plain integer arrays.

Typical usage:
>>> ./capture_hmac.py -c configs/hmac_sca_cw310.yaml -p projects/hmac_sca_capture
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
    batch_mode: bool
    num_traces: int
    num_segments: int
    output_len: int
    mask_fixed: list[int]
    key_fixed: list[int]
    key_len_bytes: int
    msg_len_bytes: int
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
        usb_serial = cfg["target"].get("usb_serial"),
        interface = cfg["target"].get("interface")
    )
    target = Target(target_cfg)

    # Init scope.
    scope_type = cfg["capture"]["scope_select"]

    # Determine sampling rate, if necessary.
    cfg[scope_type]["sampling_rate"] = determine_sampling_rate(cfg, scope_type)
    # Convert number of cycles into number of samples, if necessary.
    cfg[scope_type]["num_samples"] = convert_num_cycles(cfg, scope_type)
    # Convert offset in cycles into offset in samples, if necessary.
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
        ot_hmac: The communication interface to the HMAC SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
    """
    # Create communication interface to OT HMAC.
    ot_hmac = OTHMAC(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to OT PRNG.
    ot_prng = OTPRNG(target=target, protocol=capture_cfg.protocol)

    return ot_hmac, ot_prng


def configure_cipher(cfg, capture_cfg, ot_hmac, ot_prng):
    """ Configure the HMAC cipher.

    Establish communication with the HMAC cipher and configure the seed.

    Args:
        cfg: The project config.
        capture_cfg: The capture config.
        ot_hmac: The communication interface to the HMAC SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
    """
    # Initialize HMAC on the target.
    ot_hmac.init()

    # Seed the PRNG used for generating keys and plaintexts in batch mode.
    if capture_cfg.batch_mode:
        # Seed host's PRNG.
        random.seed(cfg["test"]["batch_prng_seed"])

        # Seed the target's PRNG.
        ot_prng.seed_prng(cfg["test"]["batch_prng_seed"].to_bytes(4, "little"))


def generate_ref_crypto(num_segments, mode, batch, key_fixed, mask_fixed,
                        key_length, msg_length):
    """ Generate cipher material for the encryption.

    This function derives the next key as well as the plaintext for the next
    encryption.

    Args:
        num_segments: The number of iterations in batch mode.
        mode: The mode of the capture.
        batch: Batch or non-batch mode.
        key_fixed: The fixed key for FVSR.
        mask_fixed: The fixed mask for FVSR.
        key_length: The length of the key.
        msg_length: The length of the message.

    Returns:
        msg: The next message.
        key: The next key.
        mask: The next mask.
        tag: The next tag.
    """
    # First sample is always fixed.
    sample_fixed = True
    # Arrays for storing num_segments crypto material.
    key_array = []
    mask_array = []
    msg_array = []
    for it in range(0, num_segments):
        if mode == "hmac_random":
            # Generate random message and key/mask.
            key = []
            for i in range(0, key_length):
                key.append(random.randint(0, 255))
            mask = []
            for i in range(0, key_length):
                mask.append(random.randint(0, 255))
            msg = []
            for i in range(0, msg_length):
                msg.append(random.randint(0, 255))
        else:
            # Generate FvsR key/mask and message.
            if sample_fixed:
                key = key_fixed
                mask = mask_fixed
            else:
                key = []
                for i in range(0, key_length):
                    key.append(random.randint(0, 255))
                mask = []
                for i in range(0, key_length):
                    mask.append(random.randint(0, 255))
            msg = []
            for i in range(0, msg_length):
                msg.append(random.randint(0, 255))
            # The next sample is either fixed or random.
            sample_fixed = msg[0] & 0x1
        # Generate expected tag for comparison. We only compare the last
        # tag.
        mac_fixed = HMAC.new(key=bytes(key), digestmod=SHA256)
        mac_fixed.update(bytes(msg))
        tag = bytearray(mac_fixed.digest())
        # Append generated material to arrays.
        key_array.append(key)
        mask_array.append(mask)
        msg_array.append(msg)

    return msg_array, key_array, mask_array, tag


def check_ciphertext(ot_hmac, expected_last_ciphertext):
    """ Compares the received with the generated ciphertext.

    Ciphertext is read from the device and compared against the pre-computed
    generated ciphertext. In batch mode, only the last ciphertext is compared.
    Asserts on mismatch.

    Args:
        ot_hmac: The OpenTitan HMAC communication interface.
        expected_last_ciphertext: The pre-computed ciphertext.
    """
    actual_last_ciphertext = bytearray(ot_hmac.read_tag())
    assert actual_last_ciphertext == expected_last_ciphertext, (
        f"Incorrect encryption result!\n"
        f"actual:   {actual_last_ciphertext}\n"
        f"expected: {expected_last_ciphertext}"
    )


def capture(scope: Scope, ot_hmac: OTHMAC, capture_cfg: CaptureConfig,
            project: SCAProject, target: Target):
    """ Capture power consumption during HMAC Tag computation.

    Supports four different capture types:
    * hmac_batch_random: Random key, mask, and message in batch mode.
    * hmac_batch_fvsr: Fixed key, random plaintext in batch mode
    * hmac_random: Random key, mask, and message.
    * hmac_fvsr: Fixed key, random plaintext.

    Args:
        scope: The scope class representing a scope (Husky or WaveRunner).
        ot_hmac: The OpenTitan HMAC communication interface.
        capture_cfg: The configuration of the capture.
        project: The SCA project.
        target: The OpenTitan target.
    """
    # Load fixed key and mask.
    key_fixed = capture_cfg.key_fixed
    mask_fixed = capture_cfg.mask_fixed

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

            # Generate data for the HMAC test.
            msg, key, mask, tag_expected = generate_ref_crypto(
                num_segments = capture_cfg.num_segments,
                mode = capture_cfg.capture_mode,
                batch = capture_cfg.batch_mode,
                key_fixed = key_fixed,
                mask_fixed = mask_fixed,
                key_length = capture_cfg.key_len_bytes,
                msg_length = capture_cfg.msg_len_bytes)

            if capture_cfg.batch_mode:
                if capture_cfg.capture_mode == "hmac_fvsr":
                    ot_hmac.fvsr_batch(key_fixed, mask_fixed,
                                       capture_cfg.num_segments)
                else:
                    ot_hmac.random_batch(capture_cfg.num_segments)
            else:
                ot_hmac.single(msg[0], key[0], mask[0])

            # Capture traces.
            waves = scope.capture_and_transfer_waves(target)
            assert waves.shape[0] == capture_cfg.num_segments

            # Compare received ciphertext with generated.
            check_ciphertext(ot_hmac, tag_expected)

            # Store trace and crypto material into database.
            for i in range(capture_cfg.num_segments):
                # Sanity check retrieved data (wave).
                assert len(waves[i, :]) >= 1
                # Store trace into database.
                project.append_trace(wave = waves[i, :],
                                     plaintext = bytearray(msg[i]),
                                     ciphertext = bytearray(mask[i]),
                                     key = bytearray(key[i]))

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

    # Determine the capture mode and configure the current capture.
    mode = "hmac_fvsr"
    if "random" in cfg["test"]["which_test"]:
        mode = "hmac_random"

    # Setup the target, scope and project.
    target, scope, project = setup(cfg, args.project)

    # Create capture config object.
    capture_cfg = CaptureConfig(capture_mode = mode,
                                batch_mode = scope.scope_cfg.batch_mode,
                                num_traces = cfg["capture"]["num_traces"],
                                num_segments = scope.scope_cfg.num_segments,
                                output_len = cfg["target"]["output_len_bytes"],
                                key_fixed = cfg["test"]["key_fixed"],
                                mask_fixed = cfg["test"]["mask_fixed"],
                                key_len_bytes = cfg["test"]["key_len_bytes"],
                                msg_len_bytes = cfg["test"]["msg_len_bytes"],
                                protocol = cfg["target"]["protocol"],
                                port = cfg["target"].get("port"))
    logger.info(f"Setting up capture {capture_cfg.capture_mode} batch={capture_cfg.batch_mode}...")

    # Open communication with target.
    ot_hmac, ot_prng = establish_communication(target, capture_cfg)

    # Configure cipher.
    configure_cipher(cfg, capture_cfg, ot_hmac, ot_prng)

    # Capture traces.
    capture(scope, ot_hmac, capture_cfg, project, target)

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
