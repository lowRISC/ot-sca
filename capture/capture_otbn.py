#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import binascii
import logging
import random
import signal
import sys
import time
import typer
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from lib.ot_communication import OTOTBNVERT, OTUART
from project_library.project import ProjectConfig, SCAProject
from scopes.cycle_converter import convert_num_cycles, convert_offset_cycles
from scopes.scope import Scope, ScopeConfig, determine_sampling_rate
from tqdm import tqdm

sys.path.append("../")
import util.helpers as helpers
from target.cw_fpga import CWFPGA  # noqa: E402
from util import check_version  # noqa: E402
from util import plot  # noqa: E402
from util import data_generator as dg  # noqa: E402

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
    C: bytearray
    seed_fixed: bytearray
    expected_fixed_key: bytearray
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
    cfg["target"]["pll_frequency"] = cfg["target"]["target_freq"] / cfg[
        "target"]["target_clk_mult"]

    # Init target.
    logger.info(f"Initializing target {cfg['target']['target_type']} ...")
    target = CWFPGA(bitstream=cfg["target"]["fpga_bitstream"],
                    force_programming=cfg["target"]["force_program_bitstream"],
                    firmware=cfg["target"]["fw_bin"],
                    pll_frequency=cfg["target"]["pll_frequency"],
                    baudrate=cfg["target"]["baudrate"],
                    output_len=cfg["target"]["output_len_bytes"],
                    protocol=cfg["target"]["protocol"])

    # Init scope.
    scope_type = cfg["capture"]["scope_select"]

    # Determine sampling rate, if necessary.
    cfg[scope_type]["sampling_rate"] = determine_sampling_rate(cfg, scope_type)
    # Convert number of cycles into number of samples, if necessary.
    cfg[scope_type]["num_samples"] = convert_num_cycles(cfg, scope_type)
    # Convert offset in cycles into offset in samples, if necessary.
    cfg[scope_type]["offset_samples"] = convert_offset_cycles(cfg, scope_type)

    logger.info(
        f"Initializing scope {scope_type} with a sampling rate of {cfg[scope_type]['sampling_rate']}..."
    )  # noqa: E501

    # Create scope config & setup scope.
    scope_cfg = ScopeConfig(
        scope_type=scope_type,
        acqu_channel=cfg[scope_type].get("channel"),
        ip=cfg[scope_type].get("waverunner_ip"),
        num_samples=cfg[scope_type]["num_samples"],
        offset_samples=cfg[scope_type]["offset_samples"],
        sampling_rate=cfg[scope_type].get("sampling_rate"),
        num_segments=cfg[scope_type]["num_segments"],
        sparsing=cfg[scope_type].get("sparsing"),
        scope_gain=cfg[scope_type].get("scope_gain"),
        pll_frequency=cfg["target"]["pll_frequency"],
    )
    scope = Scope(scope_cfg)

    # OTBN's public-key operations might not fit into the sample buffer of the scope
    # These two parameters allows users to conrol the sampling frequency
    # `adc_mul` affects the sample frequency (clock_freq = adc_mul * pll_freq)
    # `decimate` is the ADC downsampling factor that allows us to sample at
    #  every `decimate` cycles.
    if scope_type == "husky":
        if "adc_mul" in cfg["husky"]:
            scope.scope.scope.clock.adc_mul = cfg["husky"]["adc_mul"]
        if "decimate" in cfg["husky"]:
            scope.scope.scope.adc.decimate = cfg["husky"]["decimate"]
        # Print final scope parameter
        print(
            f'Scope setup with final sampling rate of {scope.scope.scope.clock.adc_freq} S/s'
        )

    # Init project.
    project_cfg = ProjectConfig(
        type=cfg["capture"]["trace_db"],
        path=project,
        wave_dtype=np.uint16,
        overwrite=True,
        trace_threshold=cfg["capture"].get("trace_threshold"))
    project = SCAProject(project_cfg)
    project.create_project()

    return target, scope, project


def configure_cipher_keygen(cfg: dict, target,
                            capture_cfg: CaptureConfig) -> OTOTBNVERT:
    """ Configure the OTBN app.

    Establish communication with the OTBN app and configure the seed.

    Args:
        cfg: The configuration for the current experiment.
        target: The OT target.
        curve_cfg: The curve config.
        capture_cfg: The configuration of the capture.

    Returns:
        The communication interface to the OTBN app.
    """
    # Establish UART for uJSON command interface. Returns None for simpleserial.
    ot_uart = OTUART(protocol=capture_cfg.protocol, port=capture_cfg.port)

    # Create communication interface to OTBN.
    ot_otbn_vert = OTOTBNVERT(target=target.target, protocol=capture_cfg.protocol,
                   port=ot_uart.uart)

    # Seed host's PRNG.
    random.seed(cfg["test"]["batch_prng_seed"])

    # Seed the target's PRNGs
    ot_otbn_vert.write_batch_prng_seed(cfg["test"]["batch_prng_seed"].to_bytes(
        4, "little"))

    # select the otbn app on the device (0 -> keygen, 1 -> modinv)
    ot_otbn_vert.choose_otbn_app(cfg["test"]["app"])

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
        # Create curve config object
        curve_cfg = CurveConfig(curve_order_n=0,
                                key_bytes=0,
                                seed_bytes=0,
                                modinv_share_bytes=0,
                                modinv_mask_bytes=0)
        # TODO: add support for P384
        raise NotImplementedError(
            f'Curve {cfg["test"]["curve"]} is not supported')

    if capture_cfg.batch_mode:
        # TODO: add support for batch mode
        raise NotImplementedError(f'Batch mode not yet supported.')
    else:
        # Generate fixed constants for all traces of the keygen operation.
        if cfg["test"]["test_type"] == 'KEY':
            # In fixed-vs-random KEY mode we use two fixed constants:
            #    1. C - a 320 bit constant redundancy
            #    2. fixed_number - a 256 bit number used to derive the fixed key
            #                      for the fixed set of measurements. Note that in
            #                      this set, the fixed key is equal to
            #                      (C + fixed_number) mod curve_order_n
            r1, r2, r3 = dg.get_random()
            capture_cfg.C = (bytearray(r1) + bytearray(r2) +
                             bytearray(r3))[:curve_cfg.seed_bytes]
            r1, r2, r3 = dg.get_random()
            fixed_number = (bytearray(r1) + bytearray(r2) +
                            bytearray(r3))[:curve_cfg.key_bytes]
            seed_fixed_int = int.from_bytes(capture_cfg.C, byteorder='little') + \
                int.from_bytes(fixed_number, byteorder='little')
            capture_cfg.seed_fixed = seed_fixed_int.to_bytes(
                curve_cfg.seed_bytes, byteorder='little')
        else:
            # In fixed-vs-random SEED mode we use only one fixed constant:
            #    1. seed_fixed - A 320 bit constant used to derive the fixed key
            #                    for the fixed set of measurements. Note that in
            #                    this set, the fixed key is equal to:
            #                    seed_fixed mod curve_order_n
            r1, r2, r3 = dg.get_random()
            capture_cfg.seed_fixed = (bytearray(r1) + bytearray(r2) +
                                      bytearray(r3))[:curve_cfg.seed_bytes]

        # Expected key is `seed mod n`, where n is the order of the curve and
        # `seed` is interpreted as little-endian.
        capture_cfg.expected_fixed_key = int.from_bytes(
            capture_cfg.seed_fixed,
            byteorder='little') % curve_cfg.curve_order_n

    return ot_otbn_vert, curve_cfg


def generate_ref_crypto_keygen(cfg: dict, sample_fixed, curve_cfg: CurveConfig,
                               capture_cfg: CaptureConfig):
    """ Generate cipher material for the encryption.

    This function derives the next key as well as the plaintext for the next
    encryption.

    Args:
        cfg: The configuration for the current experiment.
        sample_fixed: Use fixed key or new key.
        curve_cfg: The curve config.
        capture_cfg: The configuration of the capture.

    Returns:
        seed_used: The next seed.
        mask: The next mask.
        expected_key: The next expected key.
        sample_fixed: Is the next sample fixed or not?
    """

    if capture_cfg.batch_mode:
        # TODO: add support for batch mode
        raise NotImplementedError(f'Batch mode not yet supported.')
    else:
        if cfg["test"]["masks_off"] == 'True':
            # Use a constant mask for each trace
            mask = bytearray(curve_cfg.seed_bytes)  # all zeros
        else:
            # Generate a new random mask for each trace.
            r1, r2, r3 = dg.get_random()
            mask = (bytearray(r1) + bytearray(r2) +
                    bytearray(r3))[:curve_cfg.seed_bytes]
        # Generate fixed constants for all traces of the keygen operation.
        if cfg["test"]["test_type"] == 'KEY':
            # In fixed-vs-random KEY mode, the fixed set of measurements is
            # generated using the fixed 320 bit seed. The random set of
            # measurements is generated in two steps:
            #    1. Choose a random 256 bit number r
            #    2. Compute the seed as (C + r) where C is the fixed 320 bit
            #       constant. Note that in this case the used key is equal to
            #       (C + r) mod curve_order_n
            if sample_fixed:
                seed_used = capture_cfg.seed_fixed
                expected_key = capture_cfg.expected_fixed_key
            else:
                r1, r2, r3 = dg.get_random()
                random_number = (bytearray(r1) + bytearray(r2) +
                                 bytearray(r3))[:curve_cfg.key_bytes]
                seed_used_int = int.from_bytes(capture_cfg.C, byteorder='little') + \
                    int.from_bytes(random_number, byteorder='little')
                seed_used = seed_used_int.to_bytes(curve_cfg.seed_bytes,
                                                   byteorder='little')
                expected_key = int.from_bytes(seed_used, byteorder='little') % \
                    curve_cfg.curve_order_n
        else:
            # In fixed-vs-random SEED mode, the fixed set of measurements is
            # generated using the fixed 320 bit seed. The random set of
            # measurements is generated using a random 320 bit seed. In both
            # cases, the used key is equal to:
            #    seed mod curve_order_n
            if sample_fixed:
                seed_used = capture_cfg.seed_fixed
                expected_key = capture_cfg.expected_fixed_key
            else:
                r1, r2, r3 = dg.get_random()
                seed_used = (bytearray(r1) + bytearray(r2) +
                             bytearray(r3))[:curve_cfg.seed_bytes]
                expected_key = int.from_bytes(seed_used, byteorder='little') % \
                    curve_cfg.curve_order_n

        # The next sample is either fixed or random.
        r1, r2, r3 = dg.get_random()
        sample_fixed = r1[0] & 0x1

    return seed_used, mask, expected_key, sample_fixed


def check_ciphertext_keygen(ot_otbn_vert: OTOTBNVERT, expected_key,
                            curve_cfg: CurveConfig):
    """ Compares the received with the generated key.

    Key shares are read from the device and compared against the pre-computed
    generated key. In batch mode, only the last key is compared.
    Asserts on mismatch.

    Args:
        ot_otbn_vert: The OpenTitan OTBN vertical communication interface.
        expected_key: The pre-computed key.
        curve_cfg: The curve config.
    """
    # Read the output, unmask the key, and check if it matches
    # expectations.
    share0 = ot_otbn_vert.read_output(curve_cfg.seed_bytes)
    share1 = ot_otbn_vert.read_output(curve_cfg.seed_bytes)
    if share0 is None:
        raise RuntimeError('Random share0 is none')
    if share1 is None:
        raise RuntimeError('Random share1 is none')

    d0 = int.from_bytes(share0, byteorder='little')
    d1 = int.from_bytes(share1, byteorder='little')
    actual_key = (d0 + d1) % curve_cfg.curve_order_n

    assert actual_key == expected_key, (f"Incorrect encryption result!\n"
                                        f"actual: {actual_key}\n"
                                        f"expected: {expected_key}")

    return share0, share1


def capture_keygen(cfg: dict, scope: Scope, ot_otbn_vert: OTOTBNVERT,
                   capture_cfg: CaptureConfig, curve_cfg: CurveConfig,
                   project: SCAProject, cwtarget: CWFPGA):
    """ Capture power consumption during selected OTBN operation.

    Args:
        cfg: The configuration for the current experiment.
        scope: The scope class representing a scope (Husky or WaveRunner).
        ot_otbn_vert: The OpenTitan OTBN vertical communication interface.
        capture_cfg: The configuration of the capture.
        curve_cfg: The curve config.
        project: The SCA project.
        cwtarget: The CW FPGA target.
    """
    # Initial seed.
    seed_used = capture_cfg.seed_fixed
    # Initial mask.
    mask = bytearray(curve_cfg.seed_bytes)  # all zeros
    # Initial expected key.
    expected_key = capture_cfg.expected_fixed_key

    # FVSR setup.
    sample_fixed = 1

    # Optimization for CW trace library.
    num_segments_storage = 1

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
            scope.arm()

            seed_used, mask, expected_key, sample_fixed = generate_ref_crypto_keygen(
                cfg, sample_fixed, curve_cfg, capture_cfg)

            # Trigger encryption.
            if capture_cfg.batch_mode:
                # TODO: add support for batch mode
                raise NotImplementedError(f'Batch mode not yet supported.')
            else:
                # Send the seed to ibex.
                # Ibex receives the seed and the mask and computes the two shares as:
                #     Share0 = seed XOR mask
                #     Share1 = mask
                # These shares are then forwarded to OTBN.
                ot_otbn_vert.write_keygen_seed(seed_used)
                ot_otbn_vert.start_keygen(mask)

                # Capture traces.
                waves = scope.capture_and_transfer_waves(cwtarget.target)
                assert waves.shape[0] == capture_cfg.num_segments

                # Compare received key with generated key.
                share0, share1 = check_ciphertext_keygen(
                    ot_otbn_vert, expected_key, curve_cfg)

                # Store trace into database.
                project.append_trace(wave=waves[0, :],
                                     plaintext=mask,
                                     ciphertext=share0 + share1,
                                     key=seed_used)

            # Memory allocation optimization for CW trace library.
            num_segments_storage = project.optimize_capture(
                num_segments_storage)

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
        plot.save_plot_to_file(
            project.get_waves(0, config["capture"]["plot_traces"]),
            set_indices=None,
            num_traces=config["capture"]["plot_traces"],
            outfile=config["capture"]["trace_image_filename"],
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

    # Check the ChipWhisperer version.
    check_version.check_cw("5.7.0")

    # Load configuration from file.
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Determine the capture mode and configure the current capture.
    mode = cfg["test"]["app"]
    batch = cfg["test"]["batch_mode"]
    if batch == False:
        # For non-batch mode, make sure that num_segments = 1.
        cfg[cfg["capture"]["scope_select"]]["num_segments"] = 1
        logger.info(
            "num_segments needs to be 1 in non-batch mode. Setting num_segments=1."
        )

    # Setup the target, scope and project.
    target, scope, project = setup(cfg, args.project)

    # Create capture config object.
    capture_cfg = CaptureConfig(
        capture_mode=mode,
        batch_mode=batch,
        num_traces=cfg["capture"]["num_traces"],
        num_segments=cfg[cfg["capture"]["scope_select"]]["num_segments"],
        output_len=cfg["target"]["output_len_bytes"],
        text_fixed=bytearray(),
        key_fixed=bytearray(),
        key_len_bytes=cfg["test"]["key_len_bytes"],
        text_len_bytes=cfg["test"]["text_len_bytes"],
        C=bytearray(),
        seed_fixed=bytearray(),
        expected_fixed_key=bytearray(),
        protocol=cfg["target"]["protocol"])
    logger.info(
        f"Setting up capture {capture_cfg.capture_mode} batch={capture_cfg.batch_mode}..."
    )

    # Configure cipher.
    ot_aes, curve_cfg = configure_cipher_keygen(cfg, target, capture_cfg)

    # Capture traces.
    if mode == "keygen":
        capture_keygen(cfg, scope, ot_aes, capture_cfg, curve_cfg, project,
                       target)
    else:
        # TODO: add support for modinv app
        raise NotImplementedError(f'Modinv app not yet supported.')

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
