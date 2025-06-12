#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Note: The word ciphertext refers to the tag in sha3
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
from Crypto.Hash import SHA3_256
from project_library.project import ProjectConfig, SCAProject
from scopes.scope import (Scope, ScopeConfig, convert_num_cycles,
                          convert_offset_cycles, determine_sampling_rate)
from tqdm import tqdm

import util.helpers as helpers
from target.communication.sca_prng_commands import OTPRNG
from target.communication.sca_sha3_commands import OTSHA3
from target.communication.sca_trigger_commands import OTTRIGGER
from target.targets import Target, TargetConfig
from util import check_version
from util import data_generator as dg
from util import plot

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
        ot_sha3: The communication interface to the SHA3 SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
        ot_trig: The communication interface to the SCA trigger.
    """
    # Create communication interface to OT SHA3.
    ot_sha3 = OTSHA3(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to SCA trigger.
    ot_trig = OTTRIGGER(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to OT PRNG.
    ot_prng = OTPRNG(target=target, protocol=capture_cfg.protocol)

    return ot_sha3, ot_prng, ot_trig


def configure_cipher(cfg, capture_cfg, ot_sha3, ot_prng):
    """ Configure the SHA3 cipher.

    Establish communication with the SHA3 cipher and configure the seed and mask.

    Args:
        cfg: The project config.
        capture_cfg: The capture config.
        ot_sha3: The communication interface to the SHA3 SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
    Returns:
        device_id: The ID of the target device.
    """
    # Check if we want to run KMAC SCA for FPGA or discrete. On the FPGA, we
    # can use functionality helping us to capture cleaner traces.
    fpga_mode_bit = 0
    if "cw" in cfg["target"]["target_type"]:
        fpga_mode_bit = 1
    # Initialize KMAC on the target.
    device_id = ot_sha3.init(fpga_mode_bit,
                             cfg["test"]["enable_icache"],
                             cfg["test"]["enable_dummy_instr"],
                             cfg["test"]["jittery_clock_enable"],
                             cfg["test"]["sram_readback_enable"])

    if cfg["test"]["masks_off"] is True:
        logger.info("Configure device to use constant, fast entropy!")
        ot_sha3.set_mask_off()
    else:
        ot_sha3.set_mask_on()

    # Configure PRNGs.
    # Seed the software LFSR.
    ot_sha3.write_lfsr_seed(cfg["test"]["lfsr_seed"].to_bytes(4, "little"))

    # Seed the PRNG used for generating plaintexts in batch mode.
    if capture_cfg.batch_mode:
        # Seed host's PRNG.
        random.seed(cfg["test"]["batch_prng_seed"])

        # Seed the target's PRNG.
        ot_prng.seed_prng(cfg["test"]["batch_prng_seed"].to_bytes(4, "little"))
    return device_id


def generate_ref_crypto(sample_fixed, mode, batch, plaintext,
                        plaintext_fixed, text_len_bytes):
    """ Generate cipher material for the encryption.

    This function derives the next key as well as the plaintext for the next
    encryption.

    Args:
        sample_fixed: Use fixed key or new key.
        mode: The mode of the capture.
        batch: Batch or non-batch mode.
        plaintext: The current plaintext.
        plaintext_fixed: The fixed plaintext for FVSR.
        text_len_bytes: Th length of the plaintext.

    Returns:
        plaintext: The next plaintext.
        ciphertext: The next ciphertext.
        sample_fixed: Is the next sample fixed or not?
    """
    if mode == "sha3_fvsr_data" and not batch:
        # returns a pt, ct, key (not used) tripple
        # does only need the sample_fixed argument
        if sample_fixed:
            # Expected ciphertext.
            plaintext, ciphertext, key = dg.get_sha3_fixed()
        else:
            plaintext, ciphertext, key = dg.get_sha3_random()
        # The next sample is either fixed or random.
        sample_fixed = plaintext[0] & 0x1
    else:
        if mode == "sha3_random":
            # returns pt, ct, needs pt as arguments
            sha3 = SHA3_256.new(bytes(plaintext))
            ciphertext_bytes = sha3.digest()
            ciphertext = [x for x in ciphertext_bytes]
        else:  # mode = sha3_fvsr_data_batch
            # returns random pt, ct, needs no arguments
            if sample_fixed:
                plaintext = plaintext_fixed
            else:
                random_plaintext = []
                for i in range(0, text_len_bytes):
                    random_plaintext.append(random.randint(0, 255))
                plaintext = random_plaintext

            # needed to be in sync with ot lfsr and for sample_fixed generation
            dummy_plaintext = []
            for i in range(0, 16):
                dummy_plaintext.append(random.randint(0, 255))
            # Compute ciphertext for this plaintext.
            sha3 = SHA3_256.new(bytes(plaintext))
            ciphertext_bytes = sha3.digest()
            ciphertext = [x for x in ciphertext_bytes]
            # Determine if next iteration uses fixed_key.
            sample_fixed = dummy_plaintext[0] & 0x1
    return plaintext, ciphertext, sample_fixed


def check_ciphertext(received_ciphertext, expected_last_ciphertext, ciphertext_len):
    """ Compares the received with the generated ciphertext.

    Received ciphertext is compared against the pre-computed generated
    ciphertext. In batch mode, only the last ciphertext is compared.
    Asserts on mismatch.

    Args:
        received_ciphertext: The received ciphertext.
        expected_last_ciphertext: The pre-computed ciphertext.
        ciphertext_len: The length of the ciphertext in bytes.
    """
    assert received_ciphertext == expected_last_ciphertext[0:ciphertext_len], (
        f"Incorrect encryption result!\n"
        f"actual:   {received_ciphertext}\n"
        f"expected: {expected_last_ciphertext}"
    )


def init_target(cfg: dict, capture_cfg: CaptureConfig, target: Target, text_fixed):
    """ Initializes the target.

    Establish a communication interface with the target and configure the cipher.

    Args:
        cfg: The project config.
        capture_cfg: The capture config.
        target: The OT target.
        text_fixed: The fixed text for FVSR.
    Returns:
        ot_sha3: The communication interface handler.
        device_id: The ID of the target device.
    """
    # Open communication with target.
    ot_sha3, ot_prng, ot_trig = establish_communication(target, capture_cfg)

    # Configure cipher.
    device_id = configure_cipher(cfg, capture_cfg, ot_sha3, ot_prng)

    # Configure trigger source.
    # 0 for HW, 1 for SW.
    trigger_source = 1
    if "hw" in cfg["target"].get("trigger"):
        trigger_source = 0
    ot_trig.select_trigger(trigger_source)

    # Configure the fixed text for FVSR in the batch mode.
    if capture_cfg.batch_mode:
        ot_sha3.fvsr_fixed_msg_set(text_fixed)

    return ot_sha3, device_id


def capture(scope: Scope, cfg: dict, capture_cfg: CaptureConfig,
            project: SCAProject, target: Target):
    """ Capture power consumption during SHA3 digest computation.

    Supports four different capture types:
    * sha3_random: random plaintext.
    * sha3_fvsr: Fixed vs. random data.
    * sha3_fvsr_batch: Fixed vs. random data batch.

    Args:
        scope: The scope class representing a scope (Husky or WaveRunner).
        cfg: The config of the project.
        capture_cfg: The configuration of the capture.
        project: The SCA project.
        target: The OpenTitan target.
    Returns:
        device_id: The ID of the target device.
    """
    # Initial plaintext.
    text_fixed = capture_cfg.text_fixed
    text = text_fixed

    # FVSR setup.
    # in the sha3_serial.c: `static bool run_fixed = false;`
    # we should adjust this throughout all scripts.
    sample_fixed = 0

    # Optimization for CW trace library.
    num_segments_storage = 1

    # Initialize target.
    ot_sha3, device_id = init_target(cfg, capture_cfg, target, text_fixed)

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
                # Batch mode. Is always sha3_fvsr_data
                ot_sha3.absorb_batch(capture_cfg.num_segments)
            else:
                # Non batch mode. either random or fvsr
                if capture_cfg.capture_mode == "sha3_fvsr_data":
                    text, ciphertext, sample_fixed = generate_ref_crypto(
                        sample_fixed = sample_fixed,
                        mode = capture_cfg.capture_mode,
                        batch = capture_cfg.batch_mode,
                        plaintext = text,
                        plaintext_fixed = text_fixed,
                        text_len_bytes = capture_cfg.text_len_bytes
                    )
                ot_sha3.absorb(text)
            # Capture traces.
            waves = scope.capture_and_transfer_waves(target)
            assert waves.shape[0] == capture_cfg.num_segments

            expected_ciphertext = None
            text_array = []
            ciphertext_array = []
            # Generate reference crypto material and store trace.
            for i in range(capture_cfg.num_segments):
                if capture_cfg.batch_mode or capture_cfg.capture_mode == "sha3_random":
                    text, ciphertext, sample_fixed = generate_ref_crypto(
                        sample_fixed = sample_fixed,
                        mode = capture_cfg.capture_mode,
                        batch = capture_cfg.batch_mode,
                        plaintext = text,
                        plaintext_fixed = text_fixed,
                        text_len_bytes = capture_cfg.text_len_bytes
                    )
                # Sanity check retrieved data (wave).
                assert len(waves[i, :]) >= 1

                # Append text and ciphertext to result array.
                text_array.append(text)
                ciphertext_array.append(ciphertext)

                if capture_cfg.capture_mode == "sha3_random":
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

            # Receive ciphertext and compare against expected one. If
            # successful, store into database.
            compare_len = capture_cfg.output_len
            rcv_ctx, rcv_resp = ot_sha3.read_ciphertext(compare_len)
            if rcv_resp:
                # Check response and store into database
                check_ciphertext(rcv_ctx, expected_ciphertext, compare_len)
                for i in range(capture_cfg.num_segments):
                    project.append_trace(wave = waves[i, :],
                                         plaintext = bytearray(text_array[i]),
                                         ciphertext = bytearray(ciphertext_array[i]),
                                         key = None)
                # Update the loop variable and the progress bar.
                remaining_num_traces -= capture_cfg.num_segments
                pbar.update(capture_cfg.num_segments)
            else:
                # No response, reset device and start over.
                logger.info("No response received, resetting device!")
                target.reset_target()
                ot_sha3, device_id = init_target(cfg, capture_cfg, target, text_fixed)

            # Memory allocation optimization for CW trace library.
            num_segments_storage = project.optimize_capture(num_segments_storage)
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
        print(f'Created plot with {config["capture"]["plot_traces"]} traces: '
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
    mode = "sha3_fvsr_data"
    if "sha3_random" in cfg["test"]["which_test"]:
        mode = "sha3_random"

    # Setup the target, scope and project.
    target, scope, project = setup(cfg, args.project)

    # Create capture config object.
    capture_cfg = CaptureConfig(capture_mode = mode,
                                batch_mode = scope.scope_cfg.batch_mode,
                                num_traces = cfg["capture"]["num_traces"],
                                num_segments = scope.scope_cfg.num_segments,
                                output_len = cfg["target"]["output_len_bytes"],
                                text_fixed = cfg["test"]["text_fixed"],
                                text_len_bytes = cfg["test"]["text_len_bytes"],
                                protocol = cfg["target"]["protocol"],
                                port = cfg["target"].get("port"))
    logger.info(f"Setting up capture {capture_cfg.capture_mode} batch={capture_cfg.batch_mode}...")

    # Capture traces.
    device_id = capture(scope, cfg, capture_cfg, project, target)

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
