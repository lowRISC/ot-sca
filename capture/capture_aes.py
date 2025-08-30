#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

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
from Crypto.Cipher import AES
from project_library.project import ProjectConfig, SCAProject
from scopes.scope import (Scope, ScopeConfig, convert_num_cycles,
                          convert_offset_cycles, determine_sampling_rate)
from tqdm import tqdm

import util.helpers as helpers
from target.communication.sca_aes_commands import OTAES
from target.communication.sca_prng_commands import OTPRNG
from target.communication.sca_trigger_commands import OTTRIGGER
from target.targets import Target, TargetConfig
from util import check_version, plot

# Both the number of bytes in the plaintext and key are fixed.
plaintext_len = 16
key_len = 16

"""AES SCA capture script.

Captures power traces during AES operations.

The data format of the crypto material (ciphertext, plaintext, and key) inside
the script is stored in plain integer arrays.

Typical usage:
>>> ./capture_aes.py -c configs/aes_sca_cw310.yaml -p projects/aes_sca_capture
"""


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
    key_fixed: bytearray
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
    cfg["target"]["pll_frequency"] = (
        cfg["target"]["target_freq"] / cfg["target"]["target_clk_mult"]
    )

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
        husky_serial = cfg["husky"].get("usb_serial"),
        opentitantool=cfg["target"]["opentitantool"],
    )
    target = Target(target_cfg)

    # Init scope.
    scope_type = cfg["capture"]["scope_select"]

    # Check the ChipWhisperer version.
    if scope_type == "husky":
        check_version.check_cw("5.7.0")

    if scope_type != "none":
        # Will determine sampling rate (for Husky only), if not given in cfg.
        cfg[scope_type]["sampling_rate"] = determine_sampling_rate(cfg, scope_type)
        # Will convert number of cycles into number of samples if they are not given in cfg.
        cfg[scope_type]["num_samples"] = convert_num_cycles(cfg, scope_type)
        # Will convert offset in cycles into offset in samples, if they are not given in cfg.
        cfg[scope_type]["offset_samples"] = convert_offset_cycles(cfg, scope_type)

        logger.info(
            f"Initializing scope {scope_type} with a sampling rate of \
            {cfg[scope_type]['sampling_rate']}..."
        )  # noqa: E501

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
        ot_aes: The communication interface to the AES SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
        ot_trig: The communication interface to the SCA trigger.
    """
    # Create communication interface to OT AES.
    ot_aes = OTAES(target=target)

    # Create communication interface to OT PRNG.
    ot_prng = OTPRNG(target=target)

    # Create communication interface to SCA trigger.
    ot_trig = OTTRIGGER(target=target)

    return ot_aes, ot_prng, ot_trig


def configure_cipher(cfg, ot_aes, ot_prng):
    """Configure the AES cipher.

    Establish communication with the AES cipher and configure the seed.

    Args:
        cfg: The project config.
        ot_aes: The communication interface to the AES SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
    Returns:
        device_id: The ID of the target device.
        owner_page: The owner info page
        boot_log: The boot log
        boot_measurments: The boot measurements
        version: The testOS version
    """
    # Check if we want to run AES SCA for FPGA or discrete. On the FPGA, we
    # can use functionality helping us to capture cleaner traces.
    fpga_mode_bit = 0
    if "cw" in cfg["target"]["target_type"]:
        fpga_mode_bit = 1
    # Initialize AES on the target.
    device_id, owner_page, boot_log, boot_measurements, version = ot_aes.init(
        fpga_mode_bit, cfg["test"]["core_config"], cfg["test"]["sensor_config"]
    )
    # Configure PRNGs.
    # Seed the software LFSR used for initial key masking and additionally
    # turning off the masking when '0'.
    ot_aes.seed_lfsr(cfg["test"]["lfsr_seed"].to_bytes(4, "little"))

    # Seed the PRNG used for generating keys and plaintexts, only useful for batch mode.
    # Seed host's PRNG.
    random.seed(cfg["test"]["batch_prng_seed"])
    # Seed the target's PRNG.
    ot_prng.seed_prng(cfg["test"]["batch_prng_seed"].to_bytes(4, "little"))

    return device_id, owner_page, boot_log, boot_measurements, version


def generate_ref_crypto(
    sample_fixed, mode, fixed_key, fixed_plaintext, last_ciphertext
):
    """Generate cipher material for the encryption.

    Args:
        sample_fixed: Use the fixed or random bucket.
        mode: The mode of the capture.
        fixed_key: The fixed key.
        fixed_plaintext: The fixed plaintext.
        last_ciphertext: The previous ciphertext.

    Returns:
        batch_plaintext: The plaintext used AES.
        batch_key: The key used for the AES.
        batch_ciphertext: The ciphertext used for the AES.
        new_sample_fixed: The sample_fixed for the next experiment.
    """

    if mode == "aes_fvsr_key":
        if sample_fixed == 1:
            batch_key = fixed_key
        else:
            batch_key = [random.randint(0, 255) for _ in range(key_len)]
        batch_plaintext = [random.randint(0, 255) for _ in range(plaintext_len)]
        new_sample_fixed = random.randint(0, 255) & 0x1
    elif mode == "aes_fvsr_data":
        if sample_fixed == 1:
            batch_plaintext = fixed_plaintext
        else:
            batch_plaintext = [random.randint(0, 255) for _ in range(plaintext_len)]
        new_sample_fixed = random.randint(0, 255) & 0x1
        batch_key = fixed_key
    elif mode == "aes_random":
        batch_plaintext = [random.randint(0, 255) for _ in range(plaintext_len)]
        batch_key = fixed_key
        new_sample_fixed = 1
    elif mode == "daisy_chain":
        batch_plaintext = last_ciphertext
        batch_key = fixed_key
        new_sample_fixed = 1
    elif mode == "single":
        batch_plaintext = fixed_plaintext
        batch_key = fixed_key
        new_sample_fixed = 1
    else:
        logger.info("Error: Mode not recognized.")
        return None, None, None, None

    cipher = AES.new(bytes(batch_key), AES.MODE_ECB)
    ciphertext_bytes = cipher.encrypt(bytes(batch_plaintext))
    batch_ciphertext = [x for x in ciphertext_bytes]

    return batch_plaintext, batch_key, batch_ciphertext, new_sample_fixed


def check_ciphertext(target, expected_last_ciphertext):
    """Compares the received with the generated ciphertext.

    Ciphertext is read from the device and compared against the pre-computed
    generated ciphertext. In batch mode, only the last ciphertext is compared.
    Asserts on mismatch.

    Args:
        target: The OpenTitan communication interface.
        expected_last_ciphertext: The pre-computed ciphertext.
    """
    actual_last_ciphertext_full = target.read_response()
    actual_last_ciphertext_json = json.loads(actual_last_ciphertext_full)
    actual_last_ciphertext = actual_last_ciphertext_json["ciphertext"]
    assert actual_last_ciphertext == expected_last_ciphertext, (
        f"Incorrect encryption result!\n"
        f"actual: {actual_last_ciphertext}\n"
        f"expected: {expected_last_ciphertext}"
    )


def capture(
    scope: Scope,
    ot_aes: OTAES,
    capture_cfg: CaptureConfig,
    project: SCAProject,
    target: Target,
):
    """Capture power consumption during AES encryption.

    Supports six different capture types:
    * single: Fixed key, fixed plaintext.
    * daisy_chain: Fixed key, ciphertext is the next plaintext (batched).
    * aes_random: Fixed key, random plaintext (batched).
    * aes_fvsr_key: Fixed vs. random key (batched).
    * aes_fvsr_data: Fixed vs. random data (batched).

    Args:
        scope: The scope class representing a scope (Husky or WaveRunner).
        ot_aes: The OpenTitan AES communication interface.
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
    # Starting off at plaintext to capture daisy_chain behaviour
    ciphertext = text

    # Optimization for CW trace library.
    num_segments_storage = 1

    # Register ctrl-c handler to store traces on abort.
    signal.signal(signal.SIGINT, partial(abort_handler_during_loop, project))
    # Main capture with progress bar.
    remaining_num_traces = capture_cfg.num_traces
    with tqdm(
        total=remaining_num_traces, desc="Capturing", ncols=80, unit=" traces"
    ) as pbar:
        while remaining_num_traces > 0:
            # Arm the scope.
            if scope is not None:
                scope.arm()

            if capture_cfg.capture_mode == "single":
                ot_aes.single_encrypt(key_fixed, text_fixed)
            elif capture_cfg.capture_mode == "daisy_chain":
                text = ciphertext
                ot_aes.batch_daisy_chain(capture_cfg.num_segments, key_fixed, text)
            elif capture_cfg.capture_mode == "aes_random":
                ot_aes.batch_random(capture_cfg.num_segments, key_fixed)
            elif capture_cfg.capture_mode == "aes_fvsr_key":
                ot_aes.batch_fvsr_key(capture_cfg.num_segments, key_fixed)
            elif capture_cfg.capture_mode == "aes_fvsr_data":
                ot_aes.batch_fvsr_data(capture_cfg.num_segments, key_fixed, text_fixed)
            else:
                logger.info("Error: Mode not recognized.")
                return

            # Capture and store traces
            if scope is not None:
                waves = scope.capture_and_transfer_waves(target)
                assert waves.shape[0] == capture_cfg.num_segments

            # Generate reference crypto material
            sample_fixed = 1
            for i in range(capture_cfg.num_segments):
                text, key, ciphertext, sample_fixed = generate_ref_crypto(
                    sample_fixed=sample_fixed,
                    mode=capture_cfg.capture_mode,
                    fixed_key=key_fixed,
                    fixed_plaintext=text_fixed,
                    last_ciphertext=ciphertext,
                )

                if scope is not None:
                    # Sanity check retrieved data (wave).
                    assert len(waves[i, :]) >= 1
                    # Store trace into database.
                    project.append_trace(
                        wave=waves[i, :],
                        plaintext=bytearray(text),
                        ciphertext=bytearray(ciphertext),
                        key=bytearray(key),
                    )

            # Compare received ciphertext with generated.
            check_ciphertext(target, ciphertext)

            if scope is not None:
                # Memory allocation optimization for CW trace library.
                num_segments_storage = project.optimize_capture(num_segments_storage)

            # Update the loop variable and the progress bar.
            remaining_num_traces -= capture_cfg.num_segments
            pbar.update(capture_cfg.num_segments)


def print_plot(project: SCAProject, config: dict, file: Path) -> None:
    """Print plot of traces.

    Printing the plot helps to adjust the scope gain and check for clipping.

    Args:
        project: The project containing the traces.
        config: The capture configuration.
        file: The output file path.
    """
    if config["capture"]["show_plot"] and config["capture"]["scope_select"] != "none":
        plot.save_plot_to_file(
            project.get_waves(0, config["capture"]["plot_traces"]),
            set_indices=None,
            num_traces=config["capture"]["plot_traces"],
            outfile=file,
            add_mean_stddev=True,
        )
        logger.info(
            f'Created plot with {config["capture"]["plot_traces"]} traces: '
            f'{Path(str(file) + ".html").resolve()}'
        )


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

    if cfg["test"]["which_test"] == "single":
        cfg["capture"]["num_segments"] = 1

    # Create capture config object.
    capture_cfg = CaptureConfig(
        capture_mode=cfg["test"]["which_test"],
        num_traces=cfg["capture"]["num_traces"],
        num_segments=cfg["capture"]["num_segments"],
        text_fixed=cfg["test"]["text_fixed"],
        key_fixed=cfg["test"]["key_fixed"],
        port=cfg["target"].get("port"),
    )
    logger.info(f"Setting up capture {capture_cfg.capture_mode} ...")

    # Open communication with target.
    ot_aes, ot_prng, ot_trig = establish_communication(target)

    # Configure cipher.
    device_id, owner_page, boot_log, boot_measurements, version = configure_cipher(
        cfg, ot_aes, ot_prng
    )

    # Configure trigger source.
    # 0 for HW, 1 for SW.
    trigger_source = 1
    if "hw" in cfg["target"].get("trigger"):
        trigger_source = 0
    ot_trig.select_trigger(trigger_source)

    # Capture traces.
    capture(scope, ot_aes, capture_cfg, project, target)

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
                cfg["target"]["fpga_bitstream"]
            )
        if args.save_bitstream:
            metadata["fpga_bitstream"] = helpers.get_binary_blob(
                cfg["target"]["fpga_bitstream"]
            )
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

        # Finalize the capture.
        project.finalize_capture(capture_cfg.num_traces)
        # Save and close project.
        project.save()


if __name__ == "__main__":
    main()
