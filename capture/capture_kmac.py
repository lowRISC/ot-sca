#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Note: The word ciphertext refers to the tag in kmac
#       To be compatible to the other capture scripts, the variable is
#       called ciphertext

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
from Crypto.Hash import KMAC128
from project_library.project import ProjectConfig, SCAProject
from scopes.scope import (Scope, ScopeConfig, convert_num_cycles,
                          convert_offset_cycles, determine_sampling_rate)
from tqdm import tqdm

import util.helpers as helpers
from target.communication.sca_kmac_commands import OTKMAC
from target.communication.sca_prng_commands import OTPRNG
from target.targets import Target, TargetConfig
from util import check_version
from util import data_generator as dg
from util import plot

"""KMAC SCA capture script.

Captures power traces during KMAC operations.

The data format of the crypto material (ciphertext, plaintext, and key) inside
the script is stored in plain integer arrays.

Typical usage:
>>> ./capture_kmac.py -c configs/kmac_sca_cw310.yaml -p projects/kmac_sca_capture
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

    # Create target config & setup target.
    logger.info(f"Initializing target {cfg['target']['target_type']} ...")
    target_cfg = TargetConfig(
        target_type = cfg["target"]["target_type"],
        fw_bin = cfg["target"]["fw_bin"],
        protocol = cfg["target"]["protocol"],
        pll_frequency = cfg["target"]["pll_frequency"],
        bitstream = cfg["target"].get("fpga_bitstream"),
        baudrate = cfg["target"].get("baudrate"),
        port = cfg["target"].get("port"),
        output_len = cfg["target"].get("output_len_bytes")
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


def configure_cipher(cfg, target, capture_cfg) -> OTKMAC:
    """ Configure the KMAC cipher.

    Establish communication with the KMAC cipher and configure the seed.

    Args:
        cfg: The project config.
        target: The OT target.
        capture_cfg: The capture config.

    Returns:
        The communication interface to the KMAC cipher.
    """
    # Create communication interface to OT KMAC.
    ot_kmac = OTKMAC(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to OT PRNG.
    ot_prng = OTPRNG(target=target, protocol=capture_cfg.protocol)

    # Configure PRNGs.
    # Seed the software LFSR used for initial key masking.
    ot_kmac.write_lfsr_seed(cfg["test"]["lfsr_seed"].to_bytes(4, "little"))

    # Seed the PRNG used for generating keys and plaintexts in batch mode.
    if capture_cfg.batch_mode:
        # Seed host's PRNG.
        random.seed(cfg["test"]["batch_prng_seed"])

        # Seed the target's PRNG.
        ot_prng.seed_prng(cfg["test"]["batch_prng_seed"].to_bytes(4, "little"))

    return ot_kmac


def generate_ref_crypto(sample_fixed, mode, batch, key, key_fixed, plaintext,
                        plaintext_fixed, key_length):
    """ Generate cipher material for the encryption.

    This function derives the next key as well as the plaintext for the next
    encryption.

    Args:
        sample_fixed: Use fixed key or new key.
        mode: The mode of the capture.
        batch: Batch or non-batch mode.
        key: The current key.
        key_fixed: The fixed key for FVSR.
        plaintext: The current plaintext.
        plaintext_fixed: The fixed plaintext for FVSR.
        key_length: Th length of the key.

    Returns:
        plaintext: The next plaintext.
        key: The next key.
        ciphertext: The next ciphertext.
        sample_fixed: Is the next sample fixed or not?
    """
    if mode == "kmac_fvsr_key" and not batch:
        # returns a pt, ct, key tripple
        # does only need the sample_fixed argument
        if sample_fixed:
            # Expected ciphertext.
            plaintext, ciphertext, key = dg.get_kmac_fixed()
        else:
            plaintext, ciphertext, key = dg.get_kmac_random()
        # The next sample is either fixed or random.
        sample_fixed = plaintext[0] & 0x1
    else:
        if mode == "kmac_random":
            # returns pt, ct, key, needs key and pt as arguments
            mac = KMAC128.new(key=bytes(key), mac_len=32)
            mac.update(bytes(plaintext))
            ciphertext_bytes = mac.digest()
            ciphertext = [x for x in ciphertext_bytes]
        else:
            # returns random pt, ct, key, needs no arguments
            if sample_fixed:
                # Use fixed_key as this key.
                key = key_fixed
            else:
                # Generate this key from the PRNG.
                key = []
                for i in range(0, key_length):
                    key.append(random.randint(0, 255))
            # Always generate this plaintext from PRNG (including very first one).
            plaintext = []
            for i in range(0, 16):
                plaintext.append(random.randint(0, 255))
            # Compute ciphertext for this key and plaintext.
            mac = KMAC128.new(key=bytes(key), mac_len=32)
            mac.update(bytes(plaintext))
            ciphertext_bytes = mac.digest()
            ciphertext = [x for x in ciphertext_bytes]
            # Determine if next iteration uses fixed_key.
            sample_fixed = plaintext[0] & 0x1
    return plaintext, key, ciphertext, sample_fixed


def check_ciphertext(ot_kmac, expected_last_ciphertext, ciphertext_len):
    """ Compares the received with the generated ciphertext.

    Ciphertext is read from the device and compared against the pre-computed
    generated ciphertext. In batch mode, only the last ciphertext is compared.
    Asserts on mismatch.

    Args:
        ot_kmac: The OpenTitan KMAC communication interface.
        expected_last_ciphertext: The pre-computed ciphertext.
        ciphertext_len: The length of the ciphertext in bytes.
    """
    actual_last_ciphertext = ot_kmac.read_ciphertext(ciphertext_len)
    assert actual_last_ciphertext == expected_last_ciphertext[0:ciphertext_len], (
        f"Incorrect encryption result!\n"
        f"actual:   {actual_last_ciphertext}\n"
        f"expected: {expected_last_ciphertext}"
    )


def capture(scope: Scope, ot_kmac: OTKMAC, capture_cfg: CaptureConfig,
            project: SCAProject, target: Target):
    """ Capture power consumption during KMAC Tag computation.

    Supports four different capture types:
    * kmac_random: Fixed key, random plaintext.
    * kmac_fvsr: Fixed vs. random key.
    * kmac_fvsr_batch: Fixed vs. random key batch.

    Args:
        scope: The scope class representing a scope (Husky or WaveRunner).
        ot_kmac: The OpenTitan KMAC communication interface.
        capture_cfg: The configuration of the capture.
        project: The SCA project.
        target: The OpenTitan target.
    """
    # Initial plaintext.
    text_fixed = capture_cfg.text_fixed
    text = text_fixed
    # Load fixed key.
    key_fixed = capture_cfg.key_fixed
    key = key_fixed

    # FVSR setup.
    # in the kmac_serial.c: `static bool run_fixed = false;`
    # we should adjust this throughout all scripts.
    sample_fixed = 0

    logger.info(f"Initializing OT KMAC with key {binascii.b2a_hex(bytes(key))} ...")
    if capture_cfg.capture_mode == "kmac_fvsr_key":
        ot_kmac.fvsr_key_set(key)
    else:
        ot_kmac.write_key(key)

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
                # Batch mode. Is always kmac_fvsr_key
                ot_kmac.absorb_batch(
                    capture_cfg.num_segments.to_bytes(4, "little"))
            else:
                # Non batch mode. either random or fvsr
                if capture_cfg.capture_mode == "kmac_fvsr_key":
                    text, key, ciphertext, sample_fixed = generate_ref_crypto(
                        sample_fixed = sample_fixed,
                        mode = capture_cfg.capture_mode,
                        batch = capture_cfg.batch_mode,
                        key = key,
                        key_fixed = key_fixed,
                        plaintext = text,
                        plaintext_fixed = text_fixed,
                        key_length = capture_cfg.key_len_bytes
                    )
                    ot_kmac.write_key(key)
                ot_kmac.absorb(text)
            # Capture traces.
            waves = scope.capture_and_transfer_waves(target)
            assert waves.shape[0] == capture_cfg.num_segments

            expected_ciphertext = None
            # Generate reference crypto material and store trace.
            for i in range(capture_cfg.num_segments):
                if capture_cfg.batch_mode or capture_cfg.capture_mode == "kmac_random":
                    text, key, ciphertext, sample_fixed = generate_ref_crypto(
                        sample_fixed = sample_fixed,
                        mode = capture_cfg.capture_mode,
                        batch = capture_cfg.batch_mode,
                        key = key,
                        key_fixed = key_fixed,
                        plaintext = text,
                        plaintext_fixed = text_fixed,
                        key_length = capture_cfg.key_len_bytes
                    )
                # Sanity check retrieved data (wave).
                assert len(waves[i, :]) >= 1
                # Store trace into database.
                project.append_trace(wave = waves[i, :],
                                     plaintext = bytearray(text),
                                     ciphertext = bytearray(ciphertext),
                                     key = bytearray(key))

                if capture_cfg.capture_mode == "kmac_random":
                    plaintext = bytearray(16)
                    for i in range(0, 16):
                        plaintext[i] = random.randint(0, 255)

                if capture_cfg.batch_mode:
                    exp_cipher_bytes = (ciphertext if expected_ciphertext is
                                        None else (a ^ b for (a, b) in
                                                   zip(ciphertext,
                                                       expected_ciphertext)))
                    expected_ciphertext = [x for x in exp_cipher_bytes]
                else:
                    expected_ciphertext = ciphertext

            # Compare received ciphertext with generated.
            compare_len = capture_cfg.output_len
            check_ciphertext(ot_kmac, expected_ciphertext, compare_len)

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
    mode = "kmac_fvsr_key"
    if "kmac_random" in cfg["test"]["which_test"]:
        mode = "kmac_random"

    # Setup the target, scope and project.
    target, scope, project = setup(cfg, args.project)

    # Create capture config object.
    capture_cfg = CaptureConfig(capture_mode = mode,
                                batch_mode = scope.scope_cfg.batch_mode,
                                num_traces = cfg["capture"]["num_traces"],
                                num_segments = scope.scope_cfg.num_segments,
                                output_len = cfg["target"]["output_len_bytes"],
                                text_fixed = cfg["test"]["text_fixed"],
                                key_fixed = cfg["test"]["key_fixed"],
                                key_len_bytes = cfg["test"]["key_len_bytes"],
                                text_len_bytes = cfg["test"]["text_len_bytes"],
                                protocol = cfg["target"]["protocol"],
                                port = cfg["target"].get("port"))
    logger.info(f"Setting up capture {capture_cfg.capture_mode} batch={capture_cfg.batch_mode}...")

    # Configure cipher.
    ot_kmac = configure_cipher(cfg, target, capture_cfg)

    # Capture traces.
    capture(scope, ot_kmac, capture_cfg, project, target)

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
