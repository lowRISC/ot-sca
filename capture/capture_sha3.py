#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Note: The word ciphertext refers to the tag in sha3
#       To be compatible to the other capture scripts, the variable is
#       called ciphertext

import json
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
from util import check_version, plot

# Byte lengths of the text and tag.
text_length = 16
tag_length = 32

logger = logging.getLogger()


def abort_handler_during_loop(this_project, sig, frame):
    """Abort capture and store traces.

    Args:
        this_project: Project instance.
    """
    if this_project is not None:
        logger.info("\nHandling keyboard interrupt")
        this_project.close(save=True)
    sys.exit(0)


@dataclass
class CaptureConfig:
    """Configuration class for the current capture."""

    capture_mode: str
    num_traces: int
    num_segments: int
    text_fixed: bytearray
    port: Optional[str] = "None"


def setup(cfg: dict, project: Path):
    """Setup target, scope, and project.

    Args:
        cfg: The configuration for the current experiment.
        project: The path for the project file.

    Returns:
        The target, scope, and project.
    """
    # Calculate pll_frequency of the target.
    # target_freq = pll_frequency * target_clk_mult
    # target_clk_mult is a hardcoded constant in the FPGA bitstream.
    cfg["target"]["pll_frequency"] = (cfg["target"]["target_freq"] /
                                      cfg["target"]["target_clk_mult"])

    # Init scope.
    scope_type = cfg["capture"]["scope_select"]

    # Check the ChipWhisperer version.
    if scope_type == "husky":
        check_version.check_cw("5.7.0")

    # Create target config & setup target.
    logger.info(f"Initializing target {cfg['target']['target_type']} ...")
    target_cfg = TargetConfig(
        target_type=cfg["target"]["target_type"],
        fw_bin=cfg["target"]["fw_bin"],
        pll_frequency=cfg["target"]["pll_frequency"],
        bitstream=cfg["target"].get("fpga_bitstream"),
        force_program_bitstream=cfg["target"].get("force_program_bitstream"),
        baudrate=cfg["target"].get("baudrate"),
        port=cfg["target"].get("port"),
        usb_serial=cfg["target"].get("usb_serial"),
        interface=cfg["target"].get("interface"),
        husky_serial=cfg["husky"].get("usb_serial"),
        opentitantool=cfg["target"]["opentitantool"],
    )
    target = Target(target_cfg)

    if scope_type != "none":
        # Determine sampling rate, if necessary.
        cfg[scope_type]["sampling_rate"] = determine_sampling_rate(
            cfg, scope_type)
        # Convert number of cycles into number of samples, if necessary.
        cfg[scope_type]["num_samples"] = convert_num_cycles(cfg, scope_type)
        # Convert offset in cycles into offset in samples, if necessary.
        cfg[scope_type]["offset_samples"] = convert_offset_cycles(
            cfg, scope_type)

        logger.info(f"Initializing scope {scope_type} with a sampling rate of \
            {cfg[scope_type]['sampling_rate']}...")  # noqa: E501

        # Determine if we are in batch mode or not.
        batch = True
        if "single" in cfg["test"]["which_test"]:
            batch = False

        # Create scope config & setup scope.
        scope_cfg = ScopeConfig(
            scope_type=scope_type,
            batch_mode=batch,
            bit=cfg[scope_type].get("bit"),
            acqu_channel=cfg[scope_type].get("channel"),
            ip=cfg[scope_type].get("waverunner_ip"),
            num_samples=cfg[scope_type]["num_samples"],
            offset_samples=cfg[scope_type]["offset_samples"],
            sampling_rate=cfg[scope_type].get("sampling_rate"),
            num_segments=cfg["capture"].get("num_segments"),
            sparsing=cfg[scope_type].get("sparsing"),
            scope_gain=cfg[scope_type].get("scope_gain"),
            pll_frequency=cfg["target"]["pll_frequency"],
            scope_sn=cfg[scope_type].get("usb_serial"),
        )
        scope = Scope(scope_cfg)

        # Init project.
        project_cfg = ProjectConfig(
            type=cfg["capture"]["trace_db"],
            path=project,
            wave_dtype=np.uint16,
            overwrite=True,
            trace_threshold=cfg["capture"].get("trace_threshold"),
        )
        project = SCAProject(project_cfg)
        project.create_project()
    else:
        scope = None
        project = None

    return target, scope, project


def establish_communication(target):
    """Establish communication with the target device.

    Args:
        target: The OT target.
        capture_cfg: The capture config.

    Returns:
        ot_sha3: The communication interface to the SHA3 SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
        ot_trig: The communication interface to the SCA trigger.
    """
    # Create communication interface to OT SHA3.
    ot_sha3 = OTSHA3(target=target)

    # Create communication interface to SCA trigger.
    ot_trig = OTTRIGGER(target=target)

    # Create communication interface to OT PRNG.
    ot_prng = OTPRNG(target=target)

    return ot_sha3, ot_prng, ot_trig


def configure_cipher(cfg, ot_sha3, ot_prng):
    """Configure the SHA3 cipher.

    Establish communication with the SHA3 cipher and configure the seed and mask.

    Args:
        cfg: The project config.
        ot_sha3: The communication interface to the SHA3 SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
    Returns:
        device_id: The ID of the target device.
        owner_page: The owner info page
        boot_log: The boot log
        boot_measurments: The boot measurements
        version: The testOS version
    """
    # Check if we want to run KMAC SCA for FPGA or discrete. On the FPGA, we
    # can use functionality helping us to capture cleaner traces.
    fpga_mode_bit = 0
    if "cw" in cfg["target"]["target_type"]:
        fpga_mode_bit = 1
    # Initialize KMAC on the target.
    device_id, owner_page, boot_log, boot_measurements, version = ot_sha3.init(
        fpga_mode_bit, cfg["test"]["core_config"],
        cfg["test"]["sensor_config"])

    if cfg["test"]["masks_off"] is True:
        logger.info("Configure device to use constant, fast entropy!")
        ot_sha3.set_mask_off()
    else:
        ot_sha3.set_mask_on()

    # Configure PRNGs.
    # Seed the software LFSR.
    ot_sha3.write_lfsr_seed(cfg["test"]["lfsr_seed"].to_bytes(4, "little"))

    # Seed the PRNG used for generating plaintexts in batch mode.
    # Seed host's PRNG.
    random.seed(cfg["test"]["batch_prng_seed"])

    # Seed the target's PRNG.
    ot_prng.seed_prng(cfg["test"]["batch_prng_seed"].to_bytes(4, "little"))
    return device_id, owner_page, boot_log, boot_measurements, version


def generate_ref_crypto(sample_fixed, mode, text_fixed):
    """Generate cipher material for the encryption.

    This function derives the next key as well as the plaintext for the next
    encryption.

    Args:
        sample_fixed: Use fixed key or new key.
        mode: The mode of the capture.
        text_fixed: The fixed plaintext.

    Returns:
        batch_text: The used text.
        batch_output: The returned output.
        sample_fixed: Is the next sample fixed or not?
    """
    if mode == "single_absorb":
        new_sample_fixed = 1
        batch_text = text_fixed
    elif mode == "batch_absorb":
        if sample_fixed == 1:
            batch_text = text_fixed
        else:
            batch_text = [random.randint(0, 255) for _ in range(16)]
        dummy = [random.randint(0, 255) for _ in range(16)]
        new_sample_fixed = dummy[0] & 0x1
    else:
        logger.info("Error: Mode not recognized.")
        return None, None, None

    sha3 = SHA3_256.new(bytes(batch_text))
    output_bytes = sha3.digest()
    batch_output = [x for x in output_bytes]

    return batch_text, batch_output, new_sample_fixed


def check_digest(received_output, expected_output):
    """Compares the received with the generated ciphertext.

    Received output is compared against the pre-computed generated
    output. In batch mode, only the last output is compared.
    Asserts on mismatch.

    Args:
        received_output: The received output.
        expected_output: The pre-computed output.
    """
    assert received_output == expected_output, (
        f"Incorrect encryption result!\n"
        f"actual:   {received_output}\n"
        f"expected: {expected_output}")


def init_target(cfg: dict, capture_cfg: CaptureConfig, target: Target,
                text_fixed):
    """Initializes the target.

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
    ot_sha3, ot_prng, ot_trig = establish_communication(target)

    # Configure cipher.
    device_id, owner_page, boot_log, boot_measurements, version = configure_cipher(
        cfg, ot_sha3, ot_prng)

    # Configure trigger source.
    # 0 for HW, 1 for SW.
    trigger_source = 1
    if "hw" in cfg["target"].get("trigger"):
        trigger_source = 0
    ot_trig.select_trigger(trigger_source)

    # Configure the fixed text for FVSR in the batch mode.
    if "batch" in capture_cfg.capture_mode:
        ot_sha3.fvsr_fixed_msg_set(text_fixed)

    return ot_sha3, device_id, owner_page, boot_log, boot_measurements, version


def capture(
    scope: Scope,
    cfg: dict,
    capture_cfg: CaptureConfig,
    project: SCAProject,
    target: Target,
):
    """Capture power consumption during SHA3 digest computation.

    Supports four different capture types:
    * batch_absorb: Fixed vs. random data.
    * single_absorb: Fixed data.

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
    ot_sha3, device_id, owner_page, boot_log, boot_measurements, version = init_target(
        cfg, capture_cfg, target, text_fixed)

    # Register ctrl-c handler to store traces on abort.
    signal.signal(signal.SIGINT, partial(abort_handler_during_loop, project))
    # Main capture with progress bar.
    remaining_num_traces = capture_cfg.num_traces
    with tqdm(total=remaining_num_traces,
              desc="Capturing",
              ncols=80,
              unit=" traces") as pbar:
        while remaining_num_traces > 0:
            # Arm the scope.
            if scope is not None:
                scope.arm()
            # Trigger encryption.
            if capture_cfg.capture_mode == "batch_absorb":
                # Batch mode. Is always sha3_fvsr_data
                ot_sha3.absorb_batch(capture_cfg.num_segments)
            elif capture_cfg.capture_mode == "single_absorb":
                ot_sha3.absorb(text)
            else:
                logger.info("Error: Mode not recognized.")
                return

            # Capture and store traces
            if scope is not None:
                waves = scope.capture_and_transfer_waves(target)
                assert waves.shape[0] == capture_cfg.num_segments

            # Generate reference crypto material and store trace.
            xor_output = [0 for _ in range(32)]
            for i in range(capture_cfg.num_segments):
                text, output, sample_fixed = generate_ref_crypto(
                    sample_fixed=sample_fixed,
                    mode=capture_cfg.capture_mode,
                    text_fixed=text_fixed,
                )
                xor_output = [xor_output[i] ^ output[i] for i in range(32)]

                if scope is not None:
                    # Sanity check retrieved data (wave).
                    assert len(waves[i, :]) >= 1

                    project.append_trace(
                        wave=waves[i, :],
                        plaintext=bytearray(text),
                        ciphertext=bytearray(output),
                        key=None,
                    )

            response_full = target.read_response()
            reponse_json = json.loads(response_full)
            response = reponse_json["batch_digest"]
            check_digest(response, xor_output)

            # Memory allocation optimization for CW trace library.
            if scope is not None:
                num_segments_storage = project.optimize_capture(
                    num_segments_storage)

            # Update the loop variable and the progress bar.
            remaining_num_traces -= capture_cfg.num_segments
            pbar.update(capture_cfg.num_segments)

    return device_id, owner_page, boot_log, boot_measurements, version


def print_plot(project: SCAProject, config: dict, file: Path) -> None:
    """Print plot of traces.

    Printing the plot helps to adjust the scope gain and check for clipping.

    Args:
        project: The project containing the traces.
        config: The capture configuration.
        file: The output file path.
    """
    if config["capture"]["show_plot"] and config["capture"][
            "scope_select"] != "none":
        plot.save_plot_to_file(
            project.get_waves(0, config["capture"]["plot_traces"]),
            set_indices=None,
            num_traces=config["capture"]["plot_traces"],
            outfile=file,
            add_mean_stddev=True,
        )
        print(f'Created plot with {config["capture"]["plot_traces"]} traces: '
              f'{Path(str(file) + ".html").resolve()}')


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

    # Setup the target, scope and project.
    target, scope, project = setup(cfg, args.project)

    if "single" in cfg["test"]["which_test"]:
        cfg["capture"]["num_segments"] = 1

    # Create capture config object.
    capture_cfg = CaptureConfig(
        capture_mode=cfg["test"]["which_test"],
        num_traces=cfg["capture"]["num_traces"],
        num_segments=cfg["capture"]["num_segments"],
        text_fixed=cfg["test"]["text_fixed"],
        port=cfg["target"].get("port"),
    )
    logger.info(f"Setting up capture {capture_cfg.capture_mode} ...")

    # Capture traces.
    device_id, owner_page, boot_log, boot_measurements, version = capture(
        scope, cfg, capture_cfg, project, target)

    # Print plot.
    print_plot(project, cfg, args.project)

    # Save metadata.
    if cfg["capture"]["scope_select"] != "none":
        metadata = {}
        metadata["device_id"] = device_id
        metadata["owner_page"] = owner_page
        metadata["boot_log"] = boot_log
        metadata["boot_measurements"] = boot_measurements
        metadata["version"] = version
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
            metadata["fpga_bitstream_crc"] = helpers.file_crc(
                cfg["target"]["fpga_bitstream"])
        if args.save_bitstream:
            metadata["fpga_bitstream"] = helpers.get_binary_blob(
                cfg["target"]["fpga_bitstream"])
        # Store binary information.
        metadata["fw_bin_path"] = cfg["target"]["fw_bin"]
        metadata["fw_bin_crc"] = helpers.file_crc(cfg["target"]["fw_bin"])
        if args.save_binary:
            metadata["fw_bin"] = helpers.get_binary_blob(
                cfg["target"]["fw_bin"])
        # Store user provided notes.
        metadata["notes"] = args.notes
        # Store the Git hash.
        metadata["git_hash"] = helpers.get_git_hash()
        # Write metadata into project database.
        project.write_metadata(metadata)

        # Finalize the capture.
        project.finalize_capture(capture_cfg.num_traces)
        # Save and close project.
        project.save()


if __name__ == "__main__":
    main()
