#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import binascii
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
from Crypto.Cipher import AES
from lib.ot_communication import OTAES, OTPRNG, OTUART
from project_library.project import ProjectConfig, SCAProject
from scopes.scope import (Scope, ScopeConfig, convert_num_cycles,
                          convert_offset_cycles, determine_sampling_rate)
from tqdm import tqdm

import util.helpers as helpers
from target.cw_fpga import CWFPGA
from util import check_version
from util import data_generator as dg
from util import plot

"""AES SCA capture script.

Captures power traces during AES operations.

The data format of the crypto material (ciphertext, plaintext, and key) inside
the script is stored in plain integer arrays.

Typical usage:
>>> ./capture_aes.py -c configs/aes_sca_cw310.yaml -p projects/aes_sca_capture
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
    text_fixed: bytearray
    key_fixed: bytearray
    key_len_bytes: int
    text_len_bytes: int
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

    # Init target.
    logger.info(f"Initializing target {cfg['target']['target_type']} ...")
    target = CWFPGA(
        bitstream = cfg["target"]["fpga_bitstream"],
        force_programming = cfg["target"]["force_program_bitstream"],
        firmware = cfg["target"]["fw_bin"],
        pll_frequency = cfg["target"]["pll_frequency"],
        baudrate = cfg["target"]["baudrate"],
        output_len = cfg["target"]["output_len_bytes"],
        protocol = cfg["target"]["protocol"]
    )

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
        acqu_channel = cfg[scope_type].get("channel"),
        ip = cfg[scope_type].get("waverunner_ip"),
        num_samples = cfg[scope_type]["num_samples"],
        offset_samples = cfg[scope_type]["offset_samples"],
        sampling_rate = cfg[scope_type].get("sampling_rate"),
        num_segments = cfg[scope_type]["num_segments"],
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


def configure_cipher(cfg, target, capture_cfg) -> OTAES:
    """ Configure the AES cipher.

    Establish communication with the AES cipher and configure the seed.

    Args:
        cfg: The project config.
        target: The OT target.
        capture_cfg: The capture config.

    Returns:
        The communication interface to the AES cipher.
    """
    # Establish UART for uJSON command interface. Returns None for simpleserial.
    ot_uart = OTUART(protocol=capture_cfg.protocol, port=capture_cfg.port)

    # Create communication interface to OT AES.
    ot_aes = OTAES(target=target.target, protocol=capture_cfg.protocol,
                   port=ot_uart.uart)

    # Create communication interface to OT PRNG.
    ot_prng = OTPRNG(target=target.target, protocol=capture_cfg.protocol,
                     port=ot_uart.uart)

    # If batch mode, configure PRNGs.
    if capture_cfg.batch_mode:
        # Seed host's PRNG.
        random.seed(cfg["test"]["batch_prng_seed"])

        # Seed the target's PRNGs for initial key masking, and additionally
        # turn off masking when '0'.
        ot_prng.seed_prng(cfg["test"]["lfsr_seed"].to_bytes(4, "little"))
        ot_aes.seed_lfsr(cfg["test"]["batch_prng_seed"].to_bytes(4, "little"))

    return ot_aes


def generate_ref_crypto(sample_fixed, mode, key, plaintext):
    """ Generate cipher material for the encryption.

    This function derives the next key as well as the plaintext for the next
    encryption.

    Args:
        sample_fixed: Use fixed key or new key.
        mode: The mode of the capture.
        key: The current key.
        plaintext: The current plaintext.

    Returns:
        plaintext: The next plaintext.
        key: The next key.
        ciphertext: The next ciphertext.
        sample_fixed: Is the next sample fixed or not?
    """
    if mode == "aes_fvsr_key":
        if sample_fixed:
            plaintext, ciphertext, key = dg.get_fixed()
        else:
            plaintext, ciphertext, key = dg.get_random()
        # The next sample is either fixed or random.
        sample_fixed = plaintext[0] & 0x1
    else:
        if mode == "aes_random":
            cipher = AES.new(bytes(key), AES.MODE_ECB)
            ciphertext_bytes = cipher.encrypt(bytes(plaintext))
            ciphertext = [x for x in ciphertext_bytes]

    return plaintext, key, ciphertext, sample_fixed


def check_ciphertext(ot_aes, expected_last_ciphertext, ciphertext_len):
    """ Compares the received with the generated ciphertext.

    Ciphertext is read from the device and compared against the pre-computed
    generated ciphertext. In batch mode, only the last ciphertext is compared.
    Asserts on mismatch.

    Args:
        ot_aes: The OpenTitan AES communication interface.
        expected_last_ciphertext: The pre-computed ciphertext.
        ciphertext_len: The length of the ciphertext in bytes.
    """
    actual_last_ciphertext = ot_aes.read_ciphertext(ciphertext_len)
    assert actual_last_ciphertext == expected_last_ciphertext[0:ciphertext_len], (
        f"Incorrect encryption result!\n"
        f"actual: {actual_last_ciphertext}\n"
        f"expected: {expected_last_ciphertext}"
    )


def capture(scope: Scope, ot_aes: OTAES, capture_cfg: CaptureConfig,
            project: SCAProject, cwtarget: CWFPGA):
    """ Capture power consumption during AES encryption.

    Supports four different capture types:
    * aes_random: Fixed key, random plaintext.
    * aes_random_batch: Fixed key, random plaintext in batch mode.
    * aes_fvsr: Fixed vs. random key.
    * aes_fvsr_batch: Fixed vs. random key batch.

    Args:
        scope: The scope class representing a scope (Husky or WaveRunner).
        ot_aes: The OpenTitan AES communication interface.
        capture_cfg: The configuration of the capture.
        project: The SCA project.
        cwtarget: The CW FPGA target.
    """
    # Initial plaintext.
    text_fixed = capture_cfg.text_fixed
    text = text_fixed
    # Load fixed key.
    key_fixed = capture_cfg.key_fixed
    key = key_fixed
    logger.info(f"Initializing OT AES with key {binascii.b2a_hex(bytes(key))} ...")
    if capture_cfg.capture_mode == "aes_fvsr_key":
        dg.set_start()
    else:
        ot_aes.key_set(key)

    # Generate plaintexts and keys for first batch.
    if capture_cfg.batch_mode:
        if capture_cfg.capture_mode == "aes_fvsr_key":
            ot_aes.start_fvsr_batch_generate()
            ot_aes.write_fvsr_batch_generate(capture_cfg.num_segments.to_bytes(4, "little"))
        elif capture_cfg.capture_mode == "aes_random":
            ot_aes.batch_plaintext_set(text)

    # FVSR setup.
    sample_fixed = 1

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

            # Trigger encryption.
            if capture_cfg.batch_mode:
                # Batch mode.
                if capture_cfg.capture_mode == "aes_random":
                    # Fixed key, random plaintexts.
                    ot_aes.batch_alternative_encrypt(
                        capture_cfg.num_segments.to_bytes(4, "little"))
                else:
                    # Fixed vs random key test.
                    ot_aes.fvsr_key_batch_encrypt(
                        capture_cfg.num_segments.to_bytes(4, "little"))
            else:
                # Non batch mode.
                if capture_cfg.capture_mode == "aes_fvsr_key":
                    # Generate reference crypto material for aes_fvsr_key in non-batch mode
                    text, key, ciphertext, sample_fixed = generate_ref_crypto(
                        sample_fixed = sample_fixed,
                        mode = capture_cfg.capture_mode,
                        key = key,
                        plaintext = text
                    )
                    ot_aes.key_set(key)
                ot_aes.single_encrypt(text)

            # Capture traces.
            waves = scope.capture_and_transfer_waves(cwtarget.target)
            assert waves.shape[0] == capture_cfg.num_segments

            # Generate reference crypto material for all modes other than aes_fvsr_key
            # non-batch mode.
            # Store traces
            for i in range(capture_cfg.num_segments):
                if capture_cfg.batch_mode or capture_cfg.capture_mode == "aes_random":
                    text, key, ciphertext, sample_fixed = generate_ref_crypto(
                        sample_fixed = sample_fixed,
                        mode = capture_cfg.capture_mode,
                        key = key,
                        plaintext = text
                    )
                # Sanity check retrieved data (wave).
                assert len(waves[i, :]) >= 1
                # Store trace into database.
                project.append_trace(wave = waves[i, :],
                                     plaintext = bytearray(text),
                                     ciphertext = bytearray(ciphertext),
                                     key = bytearray(key))

                if capture_cfg.capture_mode == "aes_random":
                    # Use ciphertext as next text, first text is the initial
                    # one. Convert byte list into int list.
                    text = [x for x in ciphertext]

            # Compare received ciphertext with generated.
            compare_len = capture_cfg.output_len
            if capture_cfg.batch_mode and capture_cfg.capture_mode == "aes_fvsr_key":
                compare_len = 4
            check_ciphertext(ot_aes, ciphertext, compare_len)

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
    mode = "aes_fvsr_key"
    batch = False
    if "aes_random" in cfg["test"]["which_test"]:
        mode = "aes_random"
    if "batch" in cfg["test"]["which_test"]:
        batch = True
    else:
        # For non-batch mode, make sure that num_segments = 1.
        cfg[cfg["capture"]["scope_select"]]["num_segments"] = 1
        logger.info("num_segments needs to be 1 in non-batch mode. Setting num_segments=1.")

    # Setup the target, scope and project.
    target, scope, project = setup(cfg, args.project)

    # Create capture config object.
    capture_cfg = CaptureConfig(capture_mode = mode,
                                batch_mode = batch,
                                num_traces = cfg["capture"]["num_traces"],
                                num_segments = cfg[cfg["capture"]["scope_select"]]["num_segments"],
                                output_len = cfg["target"]["output_len_bytes"],
                                text_fixed = cfg["test"]["text_fixed"],
                                key_fixed = cfg["test"]["key_fixed"],
                                key_len_bytes = cfg["test"]["key_len_bytes"],
                                text_len_bytes = cfg["test"]["text_len_bytes"],
                                protocol = cfg["target"]["protocol"],
                                port = cfg["target"].get("port"))
    logger.info(f"Setting up capture {capture_cfg.capture_mode} batch={capture_cfg.batch_mode}...")

    # Configure cipher.
    ot_aes = configure_cipher(cfg, target, capture_cfg)

    # Capture traces.
    capture(scope, ot_aes, capture_cfg, project, target)

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
    metadata["fpga_bitstream_path"] = cfg["target"]["fpga_bitstream"]
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
