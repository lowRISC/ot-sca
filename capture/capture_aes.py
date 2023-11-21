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

import lib.helpers as helpers
import numpy as np
import yaml
from Crypto.Cipher import AES
from lib.ot_communication import OTAES
from project_library.project import ProjectConfig, SCAProject
from scopes.cycle_converter import convert_num_cycles, convert_offset_cycles
from scopes.scope import Scope, ScopeConfig, determine_sampling_rate
from tqdm import tqdm

sys.path.append("../")
from target.cw_fpga import CWFPGA  # noqa: E402
from util import plot  # noqa: E402

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
    )

    # Init scope.
    scope_type = cfg["capture"]["scope_select"]

    # Determine sampling rate, if necessary.
    cfg[scope_type]["sampling_rate"] = determine_sampling_rate(cfg, scope_type)
    # Convert number of cycles into number of samples, if necessary.
    cfg[scope_type]["num_samples"] = convert_num_cycles(cfg, scope_type)
    # Convert offset in cycles into offset in samples, if necessary.
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
    # Create communication interface to OT AES.
    ot_aes = OTAES(target.target)

    # If batch mode, configure PRNGs.
    if capture_cfg.batch_mode:
        # Seed host's PRNG.
        random.seed(cfg["test"]["batch_prng_seed"])

        # Seed the target's PRNGs for initial key masking, and additionally
        # turn off masking when '0'.
        ot_aes.write_lfsr_seed(cfg["test"]["lfsr_seed"].to_bytes(4, "little"))
        ot_aes.write_batch_prng_seed(cfg["test"]["batch_prng_seed"].to_bytes(4, "little"))

    return ot_aes


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
    if mode == "aes_fsvr_key" and not batch:
        if sample_fixed:
            # Expected ciphertext.
            cipher = AES.new(bytes(key_fixed), AES.MODE_ECB)
            ciphertext = bytearray(cipher.encrypt(bytes(plaintext_fixed)))
            # Next key is random.
            key = bytearray(key_length)
            for i in range(0, key_length):
                key[i] = random.randint(0, 255)
            # Next plaintext is random.
            plaintext = bytearray(16)
            for i in range(0, 16):
                plaintext[i] = random.randint(0, 255)
            sample_fixed = 0
        else:
            cipher = AES.new(bytes(key), AES.MODE_ECB)
            ciphertext = bytearray(cipher.encrypt(bytes(plaintext)))
            # Use fixed_key as the next key.
            key = np.asarray(key_fixed)
            # Use fixed_plaintext as the next plaintext.
            plaintext = np.asarray(plaintext_fixed)
            sample_fixed = 1
    else:
        if mode == "aes_random":
            cipher = AES.new(bytes(key), AES.MODE_ECB)
            ciphertext = bytearray(cipher.encrypt(bytes(plaintext)))
        else:
            if sample_fixed:
                # Use fixed_key as this key.
                key = np.asarray(key_fixed)
            else:
                # Generate this key from the PRNG.
                key = bytearray(key_length)
                for i in range(0, key_length):
                    key[i] = random.randint(0, 255)
            # Always generate this plaintext from PRNG (including very first one).
            plaintext = bytearray(16)
            for i in range(0, 16):
                plaintext[i] = random.randint(0, 255)
            # Compute ciphertext for this key and plaintext.
            # TODO: Instantiating the AES could be a bottleneck.
            cipher = AES.new(bytes(key), AES.MODE_ECB)
            ciphertext = bytearray(cipher.encrypt(bytes(plaintext)))
            # Determine if next iteration uses fixed_key.
            sample_fixed = plaintext[0] & 0x1

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
    text_fixed = bytearray(capture_cfg.text_fixed)
    text = text_fixed
    # Load fixed key.
    key_fixed = bytearray(capture_cfg.key_fixed)
    key = key_fixed
    logger.info(f"Initializing OT AES with key {binascii.b2a_hex(bytes(key))} ...")
    if capture_cfg.capture_mode == "aes_fsvr_key":
        ot_aes.fvsr_key_set(key)
    else:
        ot_aes.write_key(key)

    # Generate plaintexts and keys for first batch.
    if capture_cfg.batch_mode:
        if capture_cfg.capture_mode == "aes_fsvr_key":
            ot_aes.write_fvsr_batch_generate(capture_cfg.num_segments.to_bytes(4, "little"))
        elif capture_cfg.capture_mode == "aes_random":
            ot_aes.write_init_text(text)

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
                    ot_aes.encrypt_batch(
                        capture_cfg.num_segments.to_bytes(4, "little"))
                else:
                    # Fixed vs random key test.
                    ot_aes.encrypt_fvsr_key_batch(
                        capture_cfg.num_segments.to_bytes(4, "little"))
            else:
                # Non batch mode.
                if capture_cfg.capture_mode == "aes_fsvr_key":
                    ot_aes.write_key(key)
                ot_aes.encrypt(text)

            # Capture traces.
            waves = scope.capture_and_transfer_waves(cwtarget.target)
            assert waves.shape[0] == capture_cfg.num_segments

            # Generate reference crypto material and store trace.
            for i in range(capture_cfg.num_segments):
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
                                     plaintext = text,
                                     ciphertext = ciphertext,
                                     key = key)

                if capture_cfg.capture_mode == "aes_random":
                    # Use ciphertext as next text, first text is the initial
                    # one.
                    text = ciphertext

            # Compare received ciphertext with generated.
            compare_len = capture_cfg.output_len
            if capture_cfg.batch_mode and capture_cfg.capture_mode == "aes_fsvr_key":
                compare_len = 4
            check_ciphertext(ot_aes, ciphertext, compare_len)

            # Memory allocation optimization for CW trace library.
            num_segments_storage = project.optimize_capture(num_segments_storage)

            # Update the loop variable and the progress bar.
            remaining_num_traces -= capture_cfg.num_segments
            pbar.update(capture_cfg.num_segments)


def print_plot(project: SCAProject, config: dict) -> None:
    """ Print plot of traces.

    Printing the plot helps to adjust the scope gain and check for clipping.

    Args:
        project: The project containing the traces.
        config: The capture configuration.
    """
    if config["capture"]["show_plot"]:
        plot.save_plot_to_file(project.get_waves(0, config["capture"]["plot_traces"]),
                               set_indices = None,
                               num_traces = config["capture"]["plot_traces"],
                               outfile = config["capture"]["trace_image_filename"],
                               add_mean_stddev=True)
        print(f'Created plot with {config["capture"]["plot_traces"]} traces: '
              f'{Path(config["capture"]["trace_image_filename"]).resolve()}')


def main(argv=None):
    # Configure the logger.
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    # Parse the provided arguments.
    args = helpers.parse_arguments(argv)

    # Load configuration from file.
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Determine the capture mode and configure the current capture.
    mode = "aes_fsvr_key"
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
                                text_len_bytes = cfg["test"]["text_len_bytes"])
    logger.info(f"Setting up capture {capture_cfg.capture_mode} batch={capture_cfg.batch_mode}...")

    # Configure cipher.
    ot_aes = configure_cipher(cfg, target, capture_cfg)

    # Capture traces.
    capture(scope, ot_aes, capture_cfg, project, target)

    # Print plot.
    print_plot(project, cfg)

    # Save metadata.
    metadata = {}
    metadata["datetime"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    metadata["cfg"] = cfg
    metadata["num_samples"] = scope.scope_cfg.num_samples
    metadata["offset_samples"] = scope.scope_cfg.offset_samples
    metadata["scope_gain"] = scope.scope_cfg.scope_gain
    metadata["cfg_file"] = str(args.cfg)
    metadata["fpga_bitstream"] = cfg["target"]["fpga_bitstream"]
    # TODO: Store binary into database instead of binary path.
    # (Issue lowrisc/ot-sca#214)
    metadata["fw_bin"] = cfg["target"]["fw_bin"]
    # TODO: Allow user to enter notes via CLI.
    # (Issue lowrisc/ot-sca#213)
    metadata["notes"] = ""
    project.write_metadata(metadata)

    # Save and close project.
    project.save()


if __name__ == "__main__":
    main()
