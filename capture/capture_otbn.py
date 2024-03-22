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
from target.communication.sca_otbn_commands import OTOTBN
from target.communication.sca_prng_commands import OTPRNG
from target.communication.sca_trigger_commands import OTTRIGGER
from target.targets import Target, TargetConfig
from util import check_version
from util import data_generator as dg
from util import plot

"""OTBN vertical SCA capture script.

Captures power traces during OTBN operations.

The data format of the crypto material (ciphertext, plaintext, and key) inside
the script is stored in plain integer arrays.

Typical usage:
>>> ./capture_otbn.py -c configs/otbn_vertical_keygen_sca_cw310.yaml \
        -p projects/otbn_vertical_sca_cw310_keygen
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
    key_len_bytes: int
    text_len_bytes: int
    C: Optional[list]
    seed_fixed: Optional[list]
    expected_fixed_key: Optional[bytearray]
    k_fixed: Optional[bytearray]
    expected_fixed_output: Optional[int]
    protocol: str
    port: Optional[str] = "None"


@dataclass
class CurveConfig:
    """ Configuration class for curve dependant parameters.
    """
    curve_order_n: int
    key_bytes: int
    seed_bytes: int
    modinv_share_bytes: int
    modinv_mask_bytes: int


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

    # Determine if we are in batch mode or not.
    batch = cfg["test"]["batch_mode"]

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
        curve_cfg: The capture config

    Returns:
        ot_otbn: The communication interface to the OTBN app.
        ot_prng: The communication interface to the PRNG SCA application.
        ot_trig: The communication interface to the SCA trigger.
    """
    # Create communication interface to OTBN.
    ot_otbn = OTOTBN(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to OT PRNG.
    ot_prng = OTPRNG(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to SCA trigger.
    ot_trig = OTTRIGGER(target=target, protocol=capture_cfg.protocol)

    return ot_otbn, ot_prng, ot_trig


def configure_cipher(cfg: dict, capture_cfg: CaptureConfig, ot_otbn) -> OTOTBN:
    """ Configure the OTBN app.

    Establish communication with the OTBN keygen app and configure the seed.

    Args:
        cfg: The configuration for the current experiment.
        curve_cfg: The curve config.
        capture_cfg: The configuration of the capture.
        ot_otbn: The communication interface to the OTBN app.

    Returns:
        curve_cfg: The curve configuration values.
    """
    # Initialize OTBN on the target.
    ot_otbn.init()

    # Initialize some curve-dependent parameters.
    if cfg["test"]["curve"] == 'p256':
        # Create curve config object
        curve_cfg = CurveConfig(
            curve_order_n=
            0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551,
            key_bytes=256 // 8,
            seed_bytes=320 // 8,
            modinv_share_bytes=320 // 8,
            modinv_mask_bytes=128 // 8)
    else:
        # TODO: add support for P384
        raise NotImplementedError(
            f'Curve {cfg["test"]["curve"]} is not supported')

    if capture_cfg.capture_mode == "keygen":
        # Generate fixed constants for all traces of the keygen operation.
        if cfg["test"]["test_type"] == 'KEY':
            # In fixed-vs-random KEY mode we use two fixed constants:
            #    1. C - a 320 bit constant redundancy
            #    2. fixed_number - a 256 bit number used to derive the fixed key
            #                      for the fixed set of measurements. Note that in
            #                      this set, the fixed key is equal to
            #                      (C + fixed_number) mod curve_order_n
            capture_cfg.C = random.randbytes(curve_cfg.seed_bytes)
            fixed_number = random.randbytes(curve_cfg.key_bytes)
            seed_fixed_int = int.from_bytes(capture_cfg.C, byteorder='little') + int.from_bytes(fixed_number, byteorder='little')
            capture_cfg.seed_fixed = seed_fixed_int.to_bytes(curve_cfg.seed_bytes, byteorder='little')
        else:
            # In fixed-vs-random SEED mode we use only one fixed constant:
            #    1. seed_fixed - A 320 bit constant used to derive the fixed key
            #                    for the fixed set of measurements. Note that in
            #                    this set, the fixed key is equal to:
            #                    seed_fixed mod curve_order_n
            capture_cfg.seed_fixed = random.randbytes(curve_cfg.seed_bytes)
    else:
        # TODO: add support for Modinv
        raise NotImplementedError(
            f'Curve {cfg["test"]["app"]} is not supported')

    return curve_cfg


def generate_ref_crypto_keygen(cfg: dict, curve_cfg: CurveConfig,
                               capture_cfg: CaptureConfig, sample_fixed: int):
    """ Generate cipher material for keygen application.

    Args:
        cfg: The configuration for the current experiment.
        curve_cfg: The curve config.
        capture_cfg: The configuration of the capture.

    Returns:
        seed_used: The next seed.
        mask: The next mask.
        d0_expected: The expected d0 share.
        sample_fixed: Fixed or random run.
    """
    masks = []
    seeds = []
    d0_expected = []
    mod = curve_cfg.curve_order_n << ((curve_cfg.seed_bytes - curve_cfg.key_bytes) * 8)
    for i in range(capture_cfg.num_segments):
        if cfg["test"]["masks_on"]:
            # Generate a new random mask for each trace.
            masks.append(random.randbytes(curve_cfg.seed_bytes))
        else:
            # Use a constant 0 mask for each trace.
            masks.append(bytearray(0))

        if cfg["test"]["test_type"] == 'KEY':
            # In fixed-vs-random KEY mode, the fixed set of measurements is
            # generated using the fixed 320 bit seed. The random set of
            # measurements is generated in two steps:
            #    1. Choose a random 256 bit number r
            #    2. Compute the seed as (C + r) where C is the fixed 320 bit
            #       constant. Note that in this case the used key is equal to
            #       (C + r) mod curve_order_n
            if sample_fixed:
                seeds.append(capture_cfg.seed_fixed)
            else:
                random_number = random.randbytes(curve_cfg.key_bytes)
                seed_used_int = int.from_bytes(capture_cfg.C, byteorder='little') + int.from_bytes(random_number, byteorder='little')
                seeds.append(seed_used_int.to_bytes(curve_cfg.seed_bytes, byteorder='little'))
        else:
            # In fixed-vs-random SEED mode, the fixed set of measurements is
            # generated using the fixed 320 bit seed. The random set of
            # measurements is generated using a random 320 bit seed. In both
            # cases, the used key is equal to:
            #    seed mod curve_order_n
            if sample_fixed:
                seeds.append(capture_cfg.seed_fixed)
            else:
                seeds.append(random.randbytes(curve_cfg.seed_bytes))
        # Calculate expected d0 share.
        mask_int = int.from_bytes(masks[i], "little")
        seed = int.from_bytes(seeds[i], "little") ^ mask_int
        d0 = ((seed ^ mask_int) - mask_int) % mod
        d0_expected.append(d0.to_bytes(curve_cfg.seed_bytes, byteorder='little'))
        
        # The next sample is either fixed or random.
        sample_fixed = random.getrandbits(32) & 0x1

    return seeds, masks, d0_expected, sample_fixed


def check_d0_keygen(ot_otbn: OTOTBN, d0_expected: list, curve_cfg: CurveConfig):
    """ Compares the received d0 with the generated d0 share.

    Args:
        ot_otbn: The OpenTitan OTBN vertical communication interface.
        d0_expected: The pre-computed key.
        curve_cfg: The curve config.
    """
    # Read the output and compare.
    d0_received = ot_otbn.read_batch_digest()

    # Calculate XOR of all expected d0.
    d0_exp_batch_digest = None
    for d0 in d0_expected:
        d0_int = int.from_bytes(d0, "little")
        d0_exp_batch_digest = (d0_int if d0_exp_batch_digest is None else d0_int ^ d0_exp_batch_digest)
    d0_expected = bytearray(d0_exp_batch_digest.to_bytes(curve_cfg.seed_bytes, byteorder='little'))
    d0_expected = [x for x in d0_expected]

    assert d0_expected == d0_received, (f"Incorrect d0 result!\n"
                                        f"actual: {d0_received}\n"
                                        f"expected: {d0_expected}")


def capture_keygen(cfg: dict, scope: Scope, ot_otbn: OTOTBN,
                   capture_cfg: CaptureConfig, curve_cfg: CurveConfig,
                   project: SCAProject, target: Target, ot_prng: OTPRNG):
    """ Capture power consumption during selected OTBN operation.

    Args:
        cfg: The configuration for the current experiment.
        scope: The scope class representing a scope (Husky or WaveRunner).
        ot_otbn: The OpenTitan OTBN vertical communication interface.
        capture_cfg: The configuration of the capture.
        curve_cfg: The curve config.
        project: The SCA project.
        target: The OpenTitan target.
        ot_prng: The interface to the OpenTitan PRNG.
    """
    # Seed host's PRNG.
    random.seed(cfg["test"]["batch_prng_seed"])

    # Seed the target's PRNGs
    ot_prng.seed_prng(cfg["test"]["batch_prng_seed"].to_bytes(4, "little"))

    # First sample is always fixed.
    sample_fixed = 1

    # Optimization for CW trace library.
    num_segments_storage = 1

    # Enable or disable masking.
    ot_otbn.config_keygen_masking(cfg["test"]["masks_on"])
    capture_cfg.batch_mode = True

    # Register ctrl-c handler to store traces on abort.
    signal.signal(signal.SIGINT, partial(abort_handler_during_loop, project))
    # Main capture with progress bar.
    remaining_num_traces = capture_cfg.num_traces
    with tqdm(total=remaining_num_traces, desc="Capturing", ncols=80, unit=" traces") as pbar:
        while remaining_num_traces > 0:
            # Arm the scope.
            scope.arm()
            
            # Generate reference crypto material.
            seeds, masks, expected_d0, sample_fixed = generate_ref_crypto_keygen(cfg,
                                                                                 curve_cfg,
                                                                                 capture_cfg,
                                                                                 sample_fixed)
            # Trigger encryption.
            if capture_cfg.batch_mode:
                # Send the seed to ibex.
                # Ibex receives the seed and the mask and computes the two shares as:
                #     Share0 = seed XOR mask
                #     Share1 = mask
                # These shares are then forwarded to OTBN.
                if cfg["test"]["test_type"] == 'KEY':
                    ot_otbn.write_keygen_key_constant_redundancy(capture_cfg.C)
                ot_otbn.write_keygen_seed(capture_cfg.seed_fixed)
                ot_otbn.start_keygen_batch(cfg["test"]["test_type"], capture_cfg.num_segments)
            else:
                # TODO: add support for batch mode
                raise NotImplementedError('Non-batch mode not yet supported.')
                
            # Capture traces.
            waves = scope.capture_and_transfer_waves(target)
            assert waves.shape[0] == capture_cfg.num_segments

            # Compare received d0 with generated d0.
            check_d0_keygen(ot_otbn, expected_d0, curve_cfg)

            # Store trace into database.
            for i in range(capture_cfg.num_segments):
                assert len(waves[i, :]) >= 1
                project.append_trace(wave = waves[i, :],
                                     plaintext=masks[i],
                                     ciphertext=expected_d0[i],
                                     key=seeds[i])

            # Memory allocation optimization for CW trace library.
            num_segments_storage = project.optimize_capture(
                num_segments_storage)

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
    mode = cfg["test"]["app"]

    # Setup the target, scope and project.
    target, scope, project = setup(cfg, args.project)

    # Create capture config object.
    capture_cfg = CaptureConfig(
        capture_mode=mode,
        batch_mode=scope.scope_cfg.batch_mode,
        num_traces=cfg["capture"]["num_traces"],
        num_segments=scope.scope_cfg.num_segments,
        output_len=cfg["target"]["output_len_bytes"],
        key_len_bytes=cfg["test"]["key_len_bytes"],
        text_len_bytes=cfg["test"]["text_len_bytes"],
        protocol=cfg["target"]["protocol"],
        port = cfg["target"].get("port"),
        C=bytearray(),
        seed_fixed=bytearray(),
        expected_fixed_key=bytearray(),
        k_fixed=bytearray(),
        expected_fixed_output=0)
    logger.info(
        f"Setting up capture {capture_cfg.capture_mode} batch={capture_cfg.batch_mode}..."
    )

    # Open communication with target.
    ot_otbn, ot_prng, ot_trig = establish_communication(target, capture_cfg)

    # Configure cipher.
    curve_cfg = configure_cipher(cfg, capture_cfg, ot_otbn)

    # Select SW trigger on the device.
    ot_trig.select_trigger(1)

    # Capture traces.
    if mode == "keygen":
        capture_keygen(cfg, scope, ot_otbn, capture_cfg, curve_cfg,
                       project, target, ot_prng)
    else:
        # TODO: add support for modinv app
        raise NotImplementedError('Cofigured OTBN app not yet supported.')

    # Print plot.
    print_plot(project, cfg, args.project)

    # Save metadata.
    metadata = {}
    metadata["datetime"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    metadata["cfg"] = cfg
    metadata["num_samples"] = scope.scope_cfg.num_samples
    metadata["offset_samples"] = scope.scope_cfg.offset_samples
    metadata["scope_gain"] = scope.scope_cfg.scope_gain
    #if cfg["capture"]["scope_select"] == "husky":
    #    metadata[
    #        "sampling_rate"] = scope.scope.scope.clock.adc_freq / scope.scope.scope.adc.decimate
    #    metadata["samples_trigger_high"] = scope.scope.scope.adc.trig_count
    #else:
    #    metadata["sampling_rate"] = scope.scope_cfg.sampling_rate
    metadata["num_traces"] = capture_cfg.num_traces
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
    #metadata["git_hash"] = helpers.get_git_hash()
    # Write metadata into project database.
    project.write_metadata(metadata)

    # Finale the capture.
    project.finalize_capture(capture_cfg.num_traces)
    # Save and close project.
    project.save()


if __name__ == "__main__":
    main()
