#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import binascii
import random
import signal
import sys
import time
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import chipwhisperer as cw
import numpy as np
import scared
import typer
import yaml
from chipwhisperer.common.traces import Trace
from Crypto.Cipher import AES
from Crypto.Hash import KMAC128, SHA3_256
from cw_segmented import CwSegmented
from tqdm import tqdm
from waverunner import WaveRunner

from util import device, plot


class ScopeType(str, Enum):
    cw = "cw"
    waverunner = "waverunner"


app = typer.Typer(add_completion=False)
# To be able to define subcommands for the "capture" command.
app_capture = typer.Typer()
app.add_typer(app_capture, name="capture", help="Capture traces for SCA")
# Shared options for capture commands
opt_force_program_bitstream = typer.Option(
    None, help=("Force program FPGA with the bitstream."))
opt_num_traces = typer.Option(None, help="Number of traces to capture.")
opt_plot_traces = typer.Option(None, help="Number of traces to plot.")
opt_scope_type = typer.Option(ScopeType.cw, help=("Scope type"))
opt_ciphertexts_store = typer.Option(
    False, help=("Store all ciphertexts for batch capture."))


def create_waverunner(ot, capture_cfg):
    """Create a WaveRunner object to be used for batch capture."""
    return WaveRunner(capture_cfg["waverunner_ip"])


def create_cw_segmented(ot, capture_cfg, device_cfg):
    """Create CwSegmented object to be used for batch capture."""
    return CwSegmented(
        num_samples=ot.num_samples,
        offset_samples=ot.offset_samples,
        scope_gain=capture_cfg["scope_gain"],
        scope=ot.scope,
        clkgen_freq=ot.clkgen_freq,
        adc_mul=ot.adc_mul,
    )


SCOPE_FACTORY = {
    ScopeType.cw: create_cw_segmented,
    ScopeType.waverunner: create_waverunner,
}


def abort_handler(project, sig, frame):
    """Handler for ctrl-c keyboard interrupts:
    Saves capture project before exiting, in case abort is intended.
    Needs to be registered in every capture function before capture loop.
    To register handler use:
    signal.signal(signal.SIGINT, partial(abort_handler, project))
    where 'project' is the variable for the capture project"""
    if project is not None:
        print("\nCaught keyboard interrupt -> saving project (traces)...")
        project.close(save=True)
    sys.exit(0)


def save_metadata(project, device_cfg, capture_cfg, trigger_cycles,
                  sample_rate):
    # Save metadata to project file
    if sample_rate is not None:
        project.settingsDict["sample_rate"] = sample_rate
    if device_cfg is not None:
        for entry in device_cfg:
            project.settingsDict[entry] = device_cfg[entry]
    if capture_cfg is not None:
        for entry in capture_cfg:
            project.settingsDict[entry] = capture_cfg[entry]
    # store last number of cycles where the trigger signal was high to metadata
    if trigger_cycles is not None:
        project.settingsDict["samples_trigger_high"] = trigger_cycles
    project.settingsDict["datetime"] = datetime.now().strftime(
        "%m/%d/%Y, %H:%M:%S")


# Note: initialize_capture and plot_results are also used by other scripts.
def initialize_capture(device_cfg, capture_cfg):
    """Initialize capture."""
    ot = device.OpenTitan(
        device_cfg["fpga_bitstream"],
        device_cfg["force_program_bitstream"],
        device_cfg["fw_bin"],
        device_cfg["pll_frequency"],
        device_cfg["target_clk_mult"],
        device_cfg["baudrate"],
        capture_cfg["scope_gain"],
        capture_cfg["num_cycles"],
        capture_cfg["offset_cycles"],
        capture_cfg["output_len_bytes"],
    )
    print(f"Scope setup with sampling rate {ot.scope.clock.adc_freq} S/s")
    # Ping target
    print("Reading from FPGA using simpleserial protocol.")
    version = None
    ping_cnt = 0
    while not version:
        if ping_cnt == 3:
            raise RuntimeError(
                f"No response from the target (attempts: {ping_cnt}).")
        ot.target.write("v" + "\n")
        ping_cnt += 1
        time.sleep(0.5)
        version = ot.target.read().strip()
    print(f"Target simpleserial version: {version} (attempts: {ping_cnt}).")
    return ot


def check_range(waves, bits_per_sample):
    """The ADC output is in the interval [0, 2**bits_per_sample-1]. Check that the recorded
    traces are within [1, 2**bits_per_sample-2] to ensure the ADC doesn't saturate."""
    adc_range = np.array([0, 2**bits_per_sample])
    if not (np.all(np.greater(waves[:], adc_range[0])) and
            np.all(np.less(waves[:], adc_range[1] - 1))):
        print("\nWARNING: Some samples are outside the range [" +
              str(adc_range[0] + 1) + ", " + str(adc_range[1] - 2) + "].")
        print("The ADC has a max range of [" + str(adc_range[0]) + ", " +
              str(adc_range[1] - 1) + "] and might saturate.")
        print("It is recommended to reduce the scope gain (see device.py).")


def plot_results(plot_cfg, project_name):
    """Plots traces from `project_name` using `plot_cfg` settings."""
    project = cw.open_project(project_name)

    if len(project.waves) == 0:
        print("Project contains no traces. Did the capture fail?")
        return

    plot.save_plot_to_file(project.waves, None, plot_cfg["num_traces"],
                           plot_cfg["trace_image_filename"])
    print(f'Created plot with {plot_cfg["num_traces"]} traces: '
          f'{Path(plot_cfg["trace_image_filename"]).resolve()}')


@app.command()
def init(ctx: typer.Context):
    """Initalize target for SCA."""
    initialize_capture(ctx.obj.cfg["device"], ctx.obj.cfg["capture"])


def capture_init(ctx, force_program_bitstream, num_traces, plot_traces):
    """Initializes the user data stored in the context and programs the target."""
    cfg = ctx.obj.cfg
    if force_program_bitstream is not None:
        cfg["device"]["force_program_bitstream"] = force_program_bitstream

    if num_traces:
        cfg["capture"]["num_traces"] = num_traces

    if plot_traces:
        cfg["plot_capture"]["show"] = True
        cfg["plot_capture"]["num_traces"] = plot_traces

    # Key and plaintext generator
    ctx.obj.ktp = cw.ktp.Basic()
    # This is a workaroung for https://github.com/lowRISC/ot-sca/issues/116
    if "use_fixed_key_iter" in cfg["capture"]:  # for backwarts compatibility
        ctx.obj.ktp.fixed_key = cfg["capture"]["use_fixed_key_iter"]
    # ktp.key_len is only evaluated if ktp.fixed_key is set to False
    ctx.obj.ktp.key_len = cfg["capture"]["key_len_bytes"]
    ctx.obj.ktp.text_len = cfg["capture"]["plain_text_len_bytes"]

    ctx.obj.ot = initialize_capture(cfg["device"], cfg["capture"])


def capture_loop(trace_gen, ot, capture_cfg, device_cfg):
    """Main capture loop.

    Args:
      trace_gen: A trace generator.
      capture_cfg: Capture configuration.
    """
    project = cw.create_project(capture_cfg["project_name"], overwrite=True)

    # register ctrl-c handler to not lose already recorded traces if measurement is aborted
    signal.signal(signal.SIGINT, partial(abort_handler, project))

    for _ in tqdm(range(capture_cfg["num_traces"]), desc="Capturing",
                  ncols=80):
        traces = next(trace_gen)
        check_range(traces.wave, ot.scope.adc.bits_per_sample)
        project.traces.append(traces, dtype=np.uint16)

    sample_rate = int(round(ot.scope.clock.adc_freq, -6))
    save_metadata(project, device_cfg, capture_cfg, None, sample_rate)
    project.save()


def capture_end(cfg):
    if cfg["plot_capture"]["show"]:
        plot_results(cfg["plot_capture"], cfg["capture"]["project_name"])
    if "project_export" in cfg["capture"] and cfg["capture"]["project_export"]:
        project = cw.open_project(cfg["capture"]["project_name"])
        project.export(cfg["capture"]["project_export_filename"])
        project.close(save=False)


def capture_aes_static(ot):
    """A generator for capturing AES traces for fixed key and test.

    Args:
      ot: Initialized OpenTitan target.
    """
    key = bytearray([
        0x81,
        0x1E,
        0x37,
        0x31,
        0xB0,
        0x12,
        0x0A,
        0x78,
        0x42,
        0x78,
        0x1E,
        0x22,
        0xB2,
        0x5C,
        0xDD,
        0xF9,
    ])
    text = bytearray([
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
    ])

    tqdm.write(f"Fixed key: {binascii.b2a_hex(bytes(key))}")

    while True:
        cipher = AES.new(bytes(key), AES.MODE_ECB)
        ret = cw.capture_trace(ot.scope,
                               ot.target,
                               text,
                               key,
                               ack=False,
                               as_int=True)
        if not ret:
            raise RuntimeError("Capture failed.")
        expected = binascii.b2a_hex(cipher.encrypt(bytes(text)))
        got = binascii.b2a_hex(ret.textout)
        if got != expected:
            raise RuntimeError(f"Bad ciphertext: {got} != {expected}.")
        yield ret


@app_capture.command()
def aes_static(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):
    """Capture AES traces from a target that runs the `aes_serial` program."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_loop(
        capture_aes_static(ctx.obj.ot),
        ctx.obj.ot,
        ctx.obj.cfg["capture"],
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


def capture_aes_random(ot, ktp):
    """A generator for capturing AES traces.
    Fixed key, Random texts.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
    """
    key, _ = ktp.next()
    tqdm.write(f"Using key: {binascii.b2a_hex(bytes(key))}")
    cipher = AES.new(bytes(key), AES.MODE_ECB)
    # Select the trigger type:
    # 0 - precise, hardware-generated trigger - default
    # 1 - fully software-controlled trigger
    ot.target.simpleserial_write("t", bytearray([0x00]))
    while True:
        _, text = ktp.next()
        ret = cw.capture_trace(ot.scope,
                               ot.target,
                               text,
                               key,
                               ack=False,
                               as_int=True)
        if not ret:
            raise RuntimeError("Capture failed.")
        expected = binascii.b2a_hex(cipher.encrypt(bytes(text)))
        got = binascii.b2a_hex(ret.textout)
        if got != expected:
            raise RuntimeError(f"Bad ciphertext: {got} != {expected}.")
        yield ret


@app_capture.command()
def aes_random(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):
    """Capture AES traces from a target that runs the `aes_serial` program."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_loop(
        capture_aes_random(ctx.obj.ot, ctx.obj.ktp),
        ctx.obj.ot,
        ctx.obj.cfg["capture"],
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


def optimize_cw_capture(project, num_segments_storage):
    """Optimize cw capture by managing API."""
    # Make sure to allocate sufficient memory for the storage segment array during the
    # first resize operation. By default, the ChipWhisperer API starts every new segment
    # with 1 trace and then increases it on demand by 25 traces at a time. This results in
    # frequent array resizing and decreasing capture rate.
    # See addWave() in chipwhisperer/common/traces/_base.py.
    if project.traces.cur_seg.tracehint < project.traces.seg_len:
        project.traces.cur_seg.setTraceHint(project.traces.seg_len)
    # Only keep the latest two trace storage segments enabled. By default the ChipWhisperer
    # API keeps all segments enabled and after appending a new trace, the trace ranges are
    # updated for all segments. This leads to a decreasing capture rate after time.
    # See:
    # - _updateRanges() in chipwhisperer/common/api/TraceManager.py.
    # - https://github.com/newaetech/chipwhisperer/issues/344
    if num_segments_storage != len(project.segments):
        if num_segments_storage >= 2:
            project.traces.tm.setTraceSegmentStatus(num_segments_storage - 2,
                                                    False)
        num_segments_storage = len(project.segments)
    return num_segments_storage


def check_ciphertext(ot, expected_last_ciphertext, ciphertext_len):
    """Check the first word of the last ciphertext in a batch to make sure we are in sync."""
    actual_last_ciphertext = ot.target.simpleserial_read("r",
                                                         ciphertext_len,
                                                         ack=False)
    assert actual_last_ciphertext == expected_last_ciphertext[
        0:ciphertext_len], (f"Incorrect encryption result!\n"
                            f"actual: {actual_last_ciphertext}\n"
                            f"expected: {expected_last_ciphertext}")


def capture_aes_random_batch(ot, ktp, capture_cfg, scope_type, device_cfg):
    """A generator for capturing AES traces in batch mode.
    Fixed key, Random texts.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
      capture_cfg: Capture configuration.
      scope_type: cw or waverunner as a scope for batch capture.
    """
    # Seed host's PRNG.
    # TODO: Replace this with a dedicated PRNG to avoid other packages breaking our code.
    random.seed(capture_cfg["batch_prng_seed"])
    # Set the target's key
    key = ktp.next_key()
    tqdm.write(f"Using key: {binascii.b2a_hex(bytes(key))}")
    ot.target.simpleserial_write("k", key)
    # Seed the target's PRNG
    ot.target.simpleserial_write(
        "s", capture_cfg["batch_prng_seed"].to_bytes(4, "little"))

    # Create the ChipWhisperer project.
    project = cw.create_project(capture_cfg["project_name"], overwrite=True)

    # Capture traces.
    rem_num_traces = capture_cfg["num_traces"]
    num_segments_storage = 1
    # cw and waverunner scopes are supported fot batch capture.
    scope = SCOPE_FACTORY[scope_type](ot, capture_cfg, device_cfg)

    # register ctrl-c handler to not lose already recorded traces if measurement is aborted
    signal.signal(signal.SIGINT, partial(abort_handler, project))

    with tqdm(total=rem_num_traces, desc="Capturing", ncols=80,
              unit=" traces") as pbar:
        while rem_num_traces > 0:
            # Determine the number of traces for this batch and arm the oscilloscope.
            scope.num_segments = min(rem_num_traces, scope.num_segments_max)
            scope.arm()

            # Start batch encryption.
            ot.target.simpleserial_write(
                "b", scope.num_segments_actual.to_bytes(4, "little"))

            # Transfer traces
            waves = scope.capture_and_transfer_waves()
            assert waves.shape[0] == scope.num_segments
            # Check that the ADC didn't saturate when recording this batch.
            check_range(waves, ot.scope.adc.bits_per_sample)

            # Generate plaintexts and ciphertexts to compare with the batch encryption results.
            plaintexts = [
                ktp.next()[1] for _ in range(scope.num_segments_actual)
            ]
            ciphertexts = [
                bytearray(c) for c in scared.aes.base.encrypt(
                    np.asarray(plaintexts), np.asarray(key))
            ]

            check_ciphertext(ot, ciphertexts[-1], 4)

            num_segments_storage = optimize_cw_capture(project,
                                                       num_segments_storage)

            # Add traces of this batch to the project.
            for wave, plaintext, ciphertext in zip(waves, plaintexts,
                                                   ciphertexts):
                project.traces.append(
                    cw.common.traces.Trace(wave, plaintext, ciphertext, key),
                    dtype=np.uint16,
                )

            # Update the loop variable and the progress bar.
            rem_num_traces -= scope.num_segments
            pbar.update(scope.num_segments)

    # Before saving the project, re-enable all trace storage segments.
    for s in range(len(project.segments)):
        project.traces.tm.setTraceSegmentStatus(s, True)
    assert len(project.traces) == capture_cfg["num_traces"]

    # Save metadata to project file
    sample_rate = int(round(scope._scope.clock.adc_freq, -6))
    save_metadata(project, device_cfg, capture_cfg, None, sample_rate)

    # Save the project to disk.
    project.save()


@app_capture.command()
def aes_random_batch(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
    scope_type: ScopeType = opt_scope_type,
):
    """Capture AES traces in batch mode. Fixed key random texts."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_aes_random_batch(
        ctx.obj.ot,
        ctx.obj.ktp,
        ctx.obj.cfg["capture"],
        scope_type,
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


def capture_aes_fvsr_key(ot):
    """A generator for capturing AES traces for fixed vs random key test.
    The data collection method is based on the derived test requirements (DTR) for TVLA:
    https://www.rambus.com/wp-content/uploads/2015/08/TVLA-DTR-with-AES.pdf
    The measurements are taken by using either fixed or randomly selected key.
    In order to simplify the analysis, the first sample has to use fixed key.
    The initial key and plaintext values as well as the derivation methods are as specified in the
    DTR.

    Args:
      ot: Initialized OpenTitan target.
    """
    key_generation = bytearray([
        0x12,
        0x34,
        0x56,
        0x78,
        0x9A,
        0xBC,
        0xDE,
        0xF1,
        0x23,
        0x45,
        0x67,
        0x89,
        0xAB,
        0xCD,
        0xE0,
        0xF0,
    ])
    cipher_gen = AES.new(bytes(key_generation), AES.MODE_ECB)
    text_fixed = bytearray([
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
    ])
    text_random = bytearray([
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
    ])
    key_fixed = bytearray([
        0x81,
        0x1E,
        0x37,
        0x31,
        0xB0,
        0x12,
        0x0A,
        0x78,
        0x42,
        0x78,
        0x1E,
        0x22,
        0xB2,
        0x5C,
        0xDD,
        0xF9,
    ])
    key_random = bytearray([
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
    ])

    tqdm.write(f"Fixed key: {binascii.b2a_hex(bytes(key_fixed))}")

    sample_fixed = 1
    while True:
        if sample_fixed:
            text_fixed = bytearray(cipher_gen.encrypt(text_fixed))
            key, text = key_fixed, text_fixed
        else:
            text_random = bytearray(cipher_gen.encrypt(text_random))
            key_random = bytearray(cipher_gen.encrypt(key_random))
            key, text = key_random, text_random
        sample_fixed = random.randint(0, 1)

        cipher = AES.new(bytes(key), AES.MODE_ECB)
        ret = cw.capture_trace(ot.scope,
                               ot.target,
                               text,
                               key,
                               ack=False,
                               as_int=True)
        if not ret:
            raise RuntimeError("Capture failed.")
        expected = binascii.b2a_hex(cipher.encrypt(bytes(text)))
        got = binascii.b2a_hex(ret.textout)
        if got != expected:
            raise RuntimeError(f"Bad ciphertext: {got} != {expected}.")
        yield ret


@app_capture.command()
def aes_fvsr_key(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):
    """Capture AES traces from a target that runs the `aes_serial` program."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_loop(
        capture_aes_fvsr_key(ctx.obj.ot),
        ctx.obj.ot,
        ctx.obj.cfg["capture"],
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


def capture_aes_fvsr_key_batch(ot, ktp, capture_cfg, scope_type,
                               gen_ciphertexts, device_cfg):
    """A generator for capturing AES traces for fixed vs random key test in batch mode.
    The data collection method is based on the derived test requirements (DTR) for TVLA:
    https://www.rambus.com/wp-content/uploads/2015/08/TVLA-DTR-with-AES.pdf
    The measurements are taken by using either fixed or randomly selected key in batches.
    In order to simplify the analysis, the first encryption has to use fixed key.
    To generate random keys and texts internal PRNG's are used instead of AES functions as specified
    in the DTR.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
      capture_cfg: Capture configuration.
      scope_type: cw or waverunner as a scope for batch capture.
    """
    # Seed host's PRNG.
    # TODO: Replace this with a dedicated PRNG to avoid other packages breaking our code.
    random.seed(capture_cfg["batch_prng_seed"])
    # Seed the target's PRNGs
    ot.target.simpleserial_write(
        "l", capture_cfg["lfsr_seed"].to_bytes(4, "little"))
    ot.target.simpleserial_write(
        "s", capture_cfg["batch_prng_seed"].to_bytes(4, "little"))

    # Set and transfer the fixed key.
    # Without the sleep statement, the CW305 seems to fail to configure the batch PRNG
    # seed and/or the fixed key and then gets completely out of sync.
    time.sleep(0.5)
    key_fixed = bytearray([
        0x81,
        0x1E,
        0x37,
        0x31,
        0xB0,
        0x12,
        0x0A,
        0x78,
        0x42,
        0x78,
        0x1E,
        0x22,
        0xB2,
        0x5C,
        0xDD,
        0xF9,
    ])
    tqdm.write(f"Fixed key: {binascii.b2a_hex(bytes(key_fixed))}")
    ot.target.simpleserial_write("f", key_fixed)

    sample_fixed = 1
    is_first_batch = True

    # Create the ChipWhisperer project.
    project = cw.create_project(capture_cfg["project_name"], overwrite=True)
    # Capture traces.
    rem_num_traces = capture_cfg["num_traces"]
    num_segments_storage = 1
    # cw and waverunner scopes are supported for batch capture.
    scope = SCOPE_FACTORY[scope_type](ot, capture_cfg, device_cfg)

    # register ctrl-c handler to not lose already recorded traces if measurement is aborted
    signal.signal(signal.SIGINT, partial(abort_handler, project))

    with tqdm(total=rem_num_traces, desc="Capturing", ncols=80,
              unit=" traces") as pbar:
        while rem_num_traces > 0:
            # Determine the number of traces for this batch and arm the oscilloscope.
            scope.num_segments = min(rem_num_traces, scope.num_segments_max)
            scope.arm()

            # Start batch encryption. In order to increase capture rate, after the first batch
            # encryption, the device will start automatically to generate random keys and plaintexts
            # when this script is getting waves from the scope.
            if is_first_batch:
                ot.target.simpleserial_write(
                    "g", scope.num_segments_actual.to_bytes(4, "little"))
                is_first_batch = False
            ot.target.simpleserial_write(
                "e", scope.num_segments_actual.to_bytes(4, "little"))

            # Transfer traces.
            waves = scope.capture_and_transfer_waves()
            assert waves.shape[0] == scope.num_segments
            # Check that the ADC didn't saturate when recording this batch.
            check_range(waves, ot.scope.adc.bits_per_sample)

            # Generate keys, plaintexts and ciphertexts
            keys = []
            plaintexts = []
            ciphertexts = []
            for ii in range(scope.num_segments_actual):
                if sample_fixed:
                    key = np.asarray(key_fixed)
                else:
                    key = np.asarray(ktp.next()[1])
                plaintext = np.asarray(ktp.next()[1])
                keys.append(key)
                plaintexts.append(plaintext)
                if gen_ciphertexts:
                    ciphertext = np.asarray(
                        scared.aes.base.encrypt(plaintext, key))
                    ciphertexts.append(ciphertext)
                sample_fixed = plaintext[0] & 0x1
            if gen_ciphertexts:
                expected_last_ciphertext = ciphertexts[-1]
            else:
                expected_last_ciphertext = np.asarray(
                    scared.aes.base.encrypt(plaintext, key))

            check_ciphertext(ot, expected_last_ciphertext, 4)

            num_segments_storage = optimize_cw_capture(project,
                                                       num_segments_storage)

            # Add traces of this batch to the project. By default we don't store the ciphertexts as
            # generating them on the host as well as transferring them over from the target
            # substantially reduces capture performance. It should therefore only be enabled if
            # absolutely needed.
            if gen_ciphertexts:
                for wave, plaintext, ciphertext, key in zip(
                        waves, plaintexts, ciphertexts, keys):
                    project.traces.append(
                        cw.common.traces.Trace(wave, plaintext, ciphertext,
                                               key),
                        dtype=np.uint16,
                    )
            else:
                for wave, plaintext, key in zip(waves, plaintexts, keys):
                    project.traces.append(
                        cw.common.traces.Trace(wave, plaintext, None, key),
                        dtype=np.uint16,
                    )

            # Update the loop variable and the progress bar.
            rem_num_traces -= scope.num_segments
            pbar.update(scope.num_segments)

    # Before saving the project, re-enable all trace storage segments.
    for s in range(len(project.segments)):
        project.traces.tm.setTraceSegmentStatus(s, True)
    assert len(project.traces) == capture_cfg["num_traces"]

    # Save metadata to project file
    sample_rate = int(round(scope._scope.clock.adc_freq, -6))
    save_metadata(project, device_cfg, capture_cfg, None, sample_rate)

    # Save the project to disk.
    project.save()


@app_capture.command()
def aes_fvsr_key_batch(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
    scope_type: ScopeType = opt_scope_type,
    gen_ciphertexts: bool = opt_ciphertexts_store,
):
    """Capture AES traces in batch mode. Fixed vs random keys, random texts."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_aes_fvsr_key_batch(
        ctx.obj.ot,
        ctx.obj.ktp,
        ctx.obj.cfg["capture"],
        scope_type,
        gen_ciphertexts,
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


@app_capture.command()
def aes_mix_column(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):
    """Capture AES traces. Fixed key, Random texts. 4 sets of traces. Mix Column HD CPA Attack.
    Attack implemented by ChipWhisperer:
    Repo: https://github.com/newaetech/chipwhisperer-jupyter/blob/master/experiments/MixColumn%20Attack.ipynb # noqa: E501
    Reference: https://eprint.iacr.org/2019/343.pdf
    See mix_columns_cpa_attack.py for attack portion.
    """
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)

    ctx.obj.ktp = cw.ktp.VarVec()
    ctx.obj.ktp.key_len = ctx.obj.cfg["capture"]["key_len_bytes"]
    ctx.obj.ktp.text_len = ctx.obj.cfg["capture"]["plain_text_len_bytes"]
    project_name = ctx.obj.cfg["capture"]["project_name"]
    # For each iteration, run a capture where only the bytes specified in
    # `text_range` are set to random values. All other bytes are set to a
    # fixed value.
    for var_vec in range(4):
        ctx.obj.cfg["capture"]["project_name"] = f"{project_name}_{var_vec}"
        ctx.obj.ktp.var_vec = var_vec
        capture_loop(
            capture_aes_random(ctx.obj.ot, ctx.obj.ktp),
            ctx.obj.ot,
            ctx.obj.cfg["capture"],
            ctx.obj.cfg["device"],
        )

    capture_end(ctx.obj.cfg)


def capture_sha3_random(ot, ktp, capture_cfg):
    """A generator for capturing sha3 traces.
    Fixed key, Random texts.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
    """

    # if masks_off is true:
    # configure target to disable masking and reuse constant fast entropy
    if capture_cfg["masks_off"] is True:
        print("Warning: Configure device to use constant, fast entropy!")
        ot.target.simpleserial_write("m", bytearray([0x01]))
    else:
        ot.target.simpleserial_write("m", bytearray([0x00]))

    ack_ret = ot.target.simpleserial_wait_ack(5000)
    if ack_ret is None:
        raise Exception(
            "Batch mode acknowledge error: Device and host not in sync")

    tqdm.write("No key used, as we are doing sha3 hashing")
    while True:
        _, text = ktp.next()
        ret = cw.capture_trace(ot.scope,
                               ot.target,
                               text,
                               key=None,
                               ack=False,
                               as_int=True)
        if not ret:
            raise RuntimeError("Capture failed.")
        sha3 = SHA3_256.new(text)
        expected = binascii.b2a_hex(sha3.digest())
        got = binascii.b2a_hex(ret.textout)
        if got != expected:
            raise RuntimeError(f"Bad digest: {got} != {expected}.")
        yield ret


def capture_sha3_fvsr_data_batch(ot, ktp, capture_cfg, scope_type, device_cfg):
    """A generator for fast capturing sha3 traces.
    The data collection method is based on the derived test requirements (DTR) for TVLA:
    https://www.rambus.com/wp-content/uploads/2015/08/TVLA-DTR-with-AES.pdf
    The measurements are taken by using either fixed or randomly selected message.
    In order to simplify the analysis, the first sample has to use fixed message.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
      capture_cfg: Capture configuration.
      scope_type: cw or waverunner as a scope for batch capture.
    """

    # if masks_off is true:
    # configure target to disable masking and reuse constant fast entropy
    if capture_cfg["masks_off"] is True:
        print("Warning: Configure device to use constant, fast entropy!")
        ot.target.simpleserial_write("m", bytearray([0x01]))
    else:
        ot.target.simpleserial_write("m", bytearray([0x00]))

    ack_ret = ot.target.simpleserial_wait_ack(5000)
    if ack_ret is None:
        raise Exception(
            "Batch mode acknowledge error: Device and host not in sync")

    # Value defined under Section 5.3 in the derived test requirements (DTR) for TVLA.
    plaintext_fixed = bytearray([
        0xDA,
        0x39,
        0xA3,
        0xEE,
        0x5E,
        0x6B,
        0x4B,
        0x0D,
        0x32,
        0x55,
        0xBF,
        0xEF,
        0x95,
        0x60,
        0x18,
        0x90,
    ])

    # Note that - at least on FPGA - the DTR value above may lead to "fake" leakage as for the
    # fixed trace set, the number of bits set in the first (37) and second 64-bit word (31), as
    # well as in the Hamming distance between the two (30) is different from the statistical
    # mean (32). As a result, the loading of the fixed message into the SHA3 core on average
    # discharges the power rails slightly less than loading a random message. Until the SHA3 core
    # starts processing, the power rails will recharge but they might not be able to reach the same
    # levels for the fixed and random trace set, potentially leading to a small vertical offset
    # between the two trace sets. This offset is detectable by TVLA and covers actual leakage
    # happening during the SHA3 processing. The effect is most easliy visible between loading the
    # plaintext and appending the padding, i.e., when the target is completely idle and waiting for
    # the 40 clock cycle timer delay between the RUN and PROCESS command to expire.
    #
    # Crafted plaintext value with 4 bits set per byte, and where the Hamming distance between the
    # first and second 64-bit word is exatly 4 bits per byte. This can optionally be used for
    # debugging such "fake" leakage issues.
    # plaintext_fixed = bytearray([0xA5, 0xC3, 0x5A, 0x3C, 0x96, 0x0F, 0x69, 0xF0,
    #                              0xC3, 0xA5, 0x3C, 0x5A, 0x0F, 0x96, 0xF0, 0x69])

    ot.target.simpleserial_write("f", plaintext_fixed)

    plaintext = plaintext_fixed
    random.seed(capture_cfg["batch_prng_seed"])
    ot.target.simpleserial_write(
        "l", capture_cfg["lfsr_seed"].to_bytes(4, "little"))
    ot.target.simpleserial_write(
        "s", capture_cfg["batch_prng_seed"].to_bytes(4, "little"))

    # Create the ChipWhisperer project.
    project_file = capture_cfg["project_name"]
    project = cw.create_project(project_file, overwrite=True)
    # Capture traces.
    rem_num_traces = capture_cfg["num_traces"]
    num_segments_storage = 1
    sample_fixed = False
    # cw and waverunner scopes are supported fot batch capture.
    scope = SCOPE_FACTORY[scope_type](ot, capture_cfg, device_cfg)

    # register ctrl-c handler to not lose already recorded traces if measurement is aborted
    signal.signal(signal.SIGINT, partial(abort_handler, project))

    with tqdm(total=rem_num_traces, desc="Capturing", ncols=80,
              unit=" traces") as pbar:
        while rem_num_traces > 0:
            # Determine the number of traces for this batch and arm the oscilloscope.
            scope.num_segments = min(rem_num_traces, scope.num_segments_max)

            scope.arm()
            # Start batch encryption.
            ot.target.simpleserial_write(
                "b", scope.num_segments_actual.to_bytes(4, "little"))
            # This wait ist crucial to be in sync with the device
            ack_ret = ot.target.simpleserial_wait_ack(5000)
            if ack_ret is None:
                raise Exception(
                    "Batch mode acknowledge error: Device and host not in sync"
                )

            plaintexts = []
            ciphertexts = []

            batch_digest = None
            for i in range(scope.num_segments_actual):

                if sample_fixed:
                    plaintext = plaintext_fixed
                else:
                    random_plaintext = ktp.next()[1]
                    plaintext = random_plaintext

                # needed to be in sync with ot lfsr and for sample_fixed generation
                dummy_plaintext = ktp.next()[1]

                sha3 = SHA3_256.new(plaintext)
                ciphertext = sha3.digest()

                batch_digest = (ciphertext if batch_digest is None else bytes(
                    a ^ b for (a, b) in zip(ciphertext, batch_digest)))
                plaintexts.append(plaintext)
                ciphertexts.append(binascii.b2a_hex(ciphertext))
                sample_fixed = dummy_plaintext[0] & 1

            # Transfer traces
            waves = scope.capture_and_transfer_waves()
            assert waves.shape[0] == scope.num_segments
            # Check that the ADC didn't saturate when recording this batch.
            check_range(waves, ot.scope.adc.bits_per_sample)

            # Check the batch digest to make sure we are in sync.
            check_ciphertext(ot, batch_digest, 32)

            num_segments_storage = optimize_cw_capture(project,
                                                       num_segments_storage)

            # Add traces of this batch to the project.
            for wave, plaintext, ciphertext in zip(waves, plaintexts,
                                                   ciphertexts):
                project.traces.append(
                    cw.common.traces.Trace(wave, plaintext,
                                           bytearray(ciphertext), None),
                    dtype=np.uint16,
                )
            # Update the loop variable and the progress bar.
            rem_num_traces -= scope.num_segments
            pbar.update(scope.num_segments)
    # Before saving the project, re-enable all trace storage segments.
    for s in range(len(project.segments)):
        project.traces.tm.setTraceSegmentStatus(s, True)
    assert len(project.traces) == capture_cfg["num_traces"]

    # Save metadata to project file
    sample_rate = int(round(scope._scope.clock.adc_freq, -6))
    save_metadata(project, device_cfg, capture_cfg, None, sample_rate)

    # Save the project to disk.
    project.save()


@app_capture.command()
def sha3_random(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):
    """Capture sha3 traces from a target that runs the `sha3_serial` program."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_loop(
        capture_sha3_random(ctx.obj.ot, ctx.obj.ktp, ctx.obj.cfg["capture"]),
        ctx.obj.ot,
        ctx.obj.cfg["capture"],
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


def capture_sha3_fvsr_data(ot, capture_cfg):
    """A generator for capturing sha3 traces.
    The data collection method is based on the derived test requirements (DTR) for TVLA:
    https://www.rambus.com/wp-content/uploads/2015/08/TVLA-DTR-with-AES.pdf
    The measurements are taken by using either fixed or randomly selected message.
    In order to simplify the analysis, the first sample has to use fixed message.

    Args:
      ot: Initialized OpenTitan target.
    """

    # we are using AES in ECB mode for generating random texts
    key_generation = bytearray([
        0x12,
        0x34,
        0x56,
        0x78,
        0x9A,
        0xBC,
        0xDE,
        0xF1,
        0x23,
        0x45,
        0x67,
        0x89,
        0xAB,
        0xCD,
        0xE0,
        0xF0,
    ])
    cipher = AES.new(bytes(key_generation), AES.MODE_ECB)
    text_fixed = bytearray([
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
    ])
    text_random = bytearray([
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
    ])

    sha3 = SHA3_256.new(text_fixed)
    digest_fixed = binascii.b2a_hex(sha3.digest())

    # if masks_off is true:
    # configure target to disable masking and reuse constant fast entropy
    if capture_cfg["masks_off"] is True:
        print("Warning: Configure device to use constant, fast entropy!")
        ot.target.simpleserial_write("m", bytearray([0x01]))
    else:
        ot.target.simpleserial_write("m", bytearray([0x00]))

    ack_ret = ot.target.simpleserial_wait_ack(5000)
    if ack_ret is None:
        raise Exception(
            "Batch mode acknowledge error: Device and host not in sync")

    tqdm.write("No key used, as we are doing sha3 hashing")
    ot.target.simpleserial_write(
        "l", capture_cfg["lfsr_seed"].to_bytes(4, "little"))

    # Start sampling with the fixed key.
    sample_fixed = 1
    while True:
        if sample_fixed:
            ret = cw.capture_trace(ot.scope,
                                   ot.target,
                                   text_fixed,
                                   key=None,
                                   ack=False,
                                   as_int=True)
            if not ret:
                raise RuntimeError("Capture failed.")
            expected = digest_fixed
            got = binascii.b2a_hex(ret.textout)
        else:
            text_random = bytearray(cipher.encrypt(text_random))
            ret = cw.capture_trace(ot.scope,
                                   ot.target,
                                   text_random,
                                   key=None,
                                   ack=False,
                                   as_int=True)
            if not ret:
                raise RuntimeError("Capture failed.")
            sha3 = SHA3_256.new(text_random)
            expected = binascii.b2a_hex(sha3.digest())
            got = binascii.b2a_hex(ret.textout)
        sample_fixed = random.randint(0, 1)
        if got != expected:
            raise RuntimeError(f"Bad digest: {got} != {expected}.")
        yield ret


@app_capture.command()
def sha3_fvsr_data(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):
    """Capture sha3 traces from a target that runs the `sha3_serial` program."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_loop(
        capture_sha3_fvsr_data(ctx.obj.ot, ctx.obj.cfg["capture"]),
        ctx.obj.ot,
        ctx.obj.cfg["capture"],
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


@app_capture.command()
def sha3_fvsr_data_batch(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
    scope_type: ScopeType = opt_scope_type,
):
    """Capture sha3 traces in batch mode. Fixed vs Random."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_sha3_fvsr_data_batch(
        ctx.obj.ot,
        ctx.obj.ktp,
        ctx.obj.cfg["capture"],
        scope_type,
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


def capture_kmac_random(ot, ktp):
    """A generator for capturing KMAC-128 traces.
    Fixed key, Random texts.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
    """
    key, _ = ktp.next()
    tqdm.write(f"Using key: {binascii.b2a_hex(bytes(key))}")
    while True:
        _, text = ktp.next()
        ret = cw.capture_trace(ot.scope,
                               ot.target,
                               text,
                               key,
                               ack=False,
                               as_int=True)
        if not ret:
            raise RuntimeError("Capture failed.")
        mac = KMAC128.new(key=key, mac_len=32)
        mac.update(text)
        expected = mac.hexdigest()
        expected = expected.encode("ascii")
        got = binascii.b2a_hex(ret.textout)
        if got != expected:
            raise RuntimeError(f"Bad digest: {got} != {expected}.")
        yield ret


def capture_kmac_fvsr_key_batch(ot, ktp, capture_cfg, scope_type, device_cfg):
    """A generator for fast capturing KMAC-128 traces.
    The data collection method is based on the derived test requirements (DTR) for TVLA:
    https://www.rambus.com/wp-content/uploads/2015/08/TVLA-DTR-with-AES.pdf
    The measurements are taken by using either fixed or randomly selected key.
    In order to simplify the analysis, the first sample has to use fixed key.
    The initial key and plaintext values as well as the derivation methods are as specified in the
    DTR.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
      capture_cfg: Capture configuration.
      scope_type: cw or waverunner as a scope for batch capture.
    """

    key_fixed = bytearray([
        0x81,
        0x1E,
        0x37,
        0x31,
        0xB0,
        0x12,
        0x0A,
        0x78,
        0x42,
        0x78,
        0x1E,
        0x22,
        0xB2,
        0x5C,
        0xDD,
        0xF9,
    ])
    ot.target.simpleserial_write("f", key_fixed)
    key = key_fixed
    random.seed(capture_cfg["batch_prng_seed"])
    ot.target.simpleserial_write(
        "l", capture_cfg["lfsr_seed"].to_bytes(4, "little"))
    ot.target.simpleserial_write(
        "s", capture_cfg["batch_prng_seed"].to_bytes(4, "little"))

    # Create the ChipWhisperer project.
    project_file = capture_cfg["project_name"]
    project = cw.create_project(project_file, overwrite=True)
    # Capture traces.
    rem_num_traces = capture_cfg["num_traces"]
    num_segments_storage = 1
    sample_fixed = False
    # cw and waverunner scopes are supported fot batch capture.
    scope = SCOPE_FACTORY[scope_type](ot, capture_cfg, device_cfg)

    # register ctrl-c handler to not lose already recorded traces if measurement is aborted
    signal.signal(signal.SIGINT, partial(abort_handler, project))

    with tqdm(total=rem_num_traces, desc="Capturing", ncols=80,
              unit=" traces") as pbar:
        while rem_num_traces > 0:
            # Determine the number of traces for this batch and arm the oscilloscope.
            scope.num_segments = min(rem_num_traces, scope.num_segments_max)

            scope.arm()
            # Start batch encryption.
            ot.target.simpleserial_write(
                "b", scope.num_segments_actual.to_bytes(4, "little"))

            plaintexts = []
            ciphertexts = []
            keys = []

            batch_digest = None
            for i in range(scope.num_segments_actual):

                if sample_fixed:
                    key = key_fixed
                else:
                    random_key = ktp.next()[1]
                    key = random_key

                plaintext = ktp.next()[1]

                mac = KMAC128.new(key=key, mac_len=32)
                mac.update(plaintext)
                ciphertext = bytearray.fromhex(mac.hexdigest())
                batch_digest = (ciphertext if batch_digest is None else bytes(
                    a ^ b for (a, b) in zip(ciphertext, batch_digest)))
                plaintexts.append(plaintext)
                ciphertexts.append(binascii.b2a_hex(ciphertext))
                keys.append(key)
                sample_fixed = plaintext[0] & 1

            # Transfer traces
            waves = scope.capture_and_transfer_waves()
            assert waves.shape[0] == scope.num_segments
            # Check that the ADC didn't saturate when recording this batch.
            check_range(waves, ot.scope.adc.bits_per_sample)

            # Check the batch digest to make sure we are in sync.
            check_ciphertext(ot, batch_digest, 32)

            num_segments_storage = optimize_cw_capture(project,
                                                       num_segments_storage)

            # Add traces of this batch to the project.
            for wave, plaintext, ciphertext, key in zip(
                    waves, plaintexts, ciphertexts, keys):
                project.traces.append(
                    cw.common.traces.Trace(wave, plaintext,
                                           bytearray(ciphertext), key),
                    dtype=np.uint16,
                )
            # Update the loop variable and the progress bar.
            rem_num_traces -= scope.num_segments
            pbar.update(scope.num_segments)
    # Before saving the project, re-enable all trace storage segments.
    for s in range(len(project.segments)):
        project.traces.tm.setTraceSegmentStatus(s, True)
    assert len(project.traces) == capture_cfg["num_traces"]

    # Save metadata to project file
    sample_rate = int(round(scope._scope.clock.adc_freq, -6))
    save_metadata(project, device_cfg, capture_cfg, None, sample_rate)

    # Save the project to disk.
    project.save()


@app_capture.command()
def kmac_random(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):
    """Capture KMAC-128 traces from a target that runs the `kmac_serial` program."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_loop(
        capture_kmac_random(ctx.obj.ot, ctx.obj.ktp),
        ctx.obj.ot,
        ctx.obj.cfg["capture"],
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


def capture_kmac_fvsr_key(ot, capture_cfg):
    """A generator for capturing KMAC-128 traces.
    The data collection method is based on the derived test requirements (DTR) for TVLA:
    https://www.rambus.com/wp-content/uploads/2015/08/TVLA-DTR-with-AES.pdf
    The measurements are taken by using either fixed or randomly selected key.
    In order to simplify the analysis, the first sample has to use fixed key.
    The initial key and plaintext values as well as the derivation methods are as specified in the
    DTR.

    Args:
      ot: Initialized OpenTitan target.
    """

    key_generation = bytearray([
        0x12,
        0x34,
        0x56,
        0x78,
        0x9A,
        0xBC,
        0xDE,
        0xF1,
        0x23,
        0x45,
        0x67,
        0x89,
        0xAB,
        0xCD,
        0xE0,
        0xF0,
    ])
    cipher = AES.new(bytes(key_generation), AES.MODE_ECB)
    text_fixed = bytearray([
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
    ])
    text_random = bytearray([
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
        0xCC,
    ])
    key_fixed = bytearray([
        0x81,
        0x1E,
        0x37,
        0x31,
        0xB0,
        0x12,
        0x0A,
        0x78,
        0x42,
        0x78,
        0x1E,
        0x22,
        0xB2,
        0x5C,
        0xDD,
        0xF9,
    ])
    key_random = bytearray([
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
        0x53,
    ])

    tqdm.write(f"Using fixed key: {binascii.b2a_hex(bytes(key_fixed))}")
    ot.target.simpleserial_write(
        "l", capture_cfg["lfsr_seed"].to_bytes(4, "little"))

    # Start sampling with the fixed key.
    sample_fixed = 1
    while True:
        if sample_fixed:
            text_fixed = bytearray(cipher.encrypt(text_fixed))
            ret = cw.capture_trace(ot.scope,
                                   ot.target,
                                   text_fixed,
                                   key_fixed,
                                   ack=False,
                                   as_int=True)
            if not ret:
                raise RuntimeError("Capture failed.")
            mac = KMAC128.new(key=key_fixed, mac_len=32)
            mac.update(text_fixed)
            expected = mac.hexdigest()
            expected = expected.encode("ascii")
            got = binascii.b2a_hex(ret.textout)
        else:
            text_random = bytearray(cipher.encrypt(text_random))
            key_random = bytearray(cipher.encrypt(key_random))
            ret = cw.capture_trace(ot.scope,
                                   ot.target,
                                   text_random,
                                   key_random,
                                   ack=False,
                                   as_int=True)
            if not ret:
                raise RuntimeError("Capture failed.")
            mac = KMAC128.new(key=key_random, mac_len=32)
            mac.update(text_random)
            expected = mac.hexdigest()
            expected = expected.encode("ascii")
            got = binascii.b2a_hex(ret.textout)
        sample_fixed = random.randint(0, 1)
        if got != expected:
            raise RuntimeError(f"Bad digest: {got} != {expected}.")
        yield ret


@app_capture.command()
def kmac_fvsr_key(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):
    """Capture KMAC-128 traces from a target that runs the `kmac_serial` program."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_loop(
        capture_kmac_fvsr_key(ctx.obj.ot, ctx.obj.cfg["capture"]),
        ctx.obj.ot,
        ctx.obj.cfg["capture"],
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


@app_capture.command()
def kmac_fvsr_key_batch(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
    scope_type: ScopeType = opt_scope_type,
):
    """Capture KMAC-128 traces in batch mode. Fixed vs Random."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_kmac_fvsr_key_batch(
        ctx.obj.ot,
        ctx.obj.ktp,
        ctx.obj.cfg["capture"],
        scope_type,
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


def capture_otbn_vertical(ot, ktp, fw_bin, pll_frequency, capture_cfg,
                          device_cfg):
    """Capture traces for ECDSA P256/P384 secret key generation
    and modular inverse computation.

    For keygen it uses a fixed seed and generates several random masks.
    For modinv it uses two fixed or random key shares as input for the device.
    For the corresponding driver, check:
    <opentitan_repo_root>/sw/device/sca/otbn_vertical/otbn_vertical_serial.c

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
      fw_bin: Firmware binary.
      pll_frequency: Output frequency of the FPGA PLL.
      capture_cfg: Capture configuration from the yaml file
    """

    # We need an intuitive, but non default behavior for the kpt.next() iterator.
    # For backwards compatibility this must be set in the capture config file.
    # This is a workaroung for https://github.com/lowRISC/ot-sca/issues/116
    if "use_fixed_key_iter" not in capture_cfg:
        raise RuntimeError("use_fixed_key_iter not set!")
    if capture_cfg["use_fixed_key_iter"] is not False:
        raise RuntimeError("use_fixed_key_iter must be set to false!")

    # OTBN operations are long. CW-Husky can store only 131070 samples
    # in the non-stream mode.
    fifo_size = 131070
    if ot.num_samples > fifo_size:
        raise RuntimeError("Current setup only supports up to 130k samples")

    # Be sure we don't use the stream mode
    ot.scope.adc.stream_mode = False
    ot.scope.adc.bits_per_sample = 12
    ot.scope.adc.samples = ot.num_samples

    # Create a cw project to keep the data and traces
    project = cw.create_project(capture_cfg["project_name"], overwrite=True)

    # Initialize some curve-dependent parameters.
    if capture_cfg["curve"] == "p256":
        curve_order_n = (
            0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551)
        key_bytes = 256 // 8
        seed_bytes = 320 // 8
        modinv_share_bytes = 320 // 8
        modinv_mask_bytes = 128 // 8
    else:
        # TODO: add support for P384
        raise NotImplementedError(
            f'Curve {capture_cfg["curve"]} is not supported')

    # register ctrl-c handler to not lose already recorded traces if measurement is aborted
    signal.signal(signal.SIGINT, partial(abort_handler, project))

    if capture_cfg["app"] == "keygen":
        # Check the lengths in the key/plaintext generator. In this case, "key"
        # means seed and "plaintext" means mask.
        if ktp.keyLen() != seed_bytes:
            raise ValueError(
                f"Unexpected seed length: {ktp.keyLen()}.\n"
                f"Hint: set key len={seed_bytes} in the configuration file.")
        if ktp.textLen() != seed_bytes:
            raise ValueError(
                f"Unexpected mask length: {ktp.textLen()}.\n"
                f"Hint: set plaintext len={seed_bytes} in the configuration file."
            )

        # select the otbn app on the device (0 -> keygen, 1 -> modinv)
        ot.target.simpleserial_write("a", bytearray([0x00]))
        time.sleep(0.3)

        # Seed the RNG
        random.seed(capture_cfg["batch_prng_seed"])

        # Generate fixed constants for all traces of the keygen operation.
        if capture_cfg["test_type"] == "KEY":
            # In fixed-vs-random KEY mode we use two fixed constants:
            #    1. C - a 320 bit constant redundancy
            #    2. fixed_number - a 256 bit number used to derive the fixed key
            #                      for the fixed set of measurements. Note that in
            #                      this set, the fixed key is equal to
            #                      (C + fixed_number) mod curve_order_n
            C = ktp.next_key()
            if len(C) != seed_bytes:
                raise ValueError(
                    f"Fixed seed length is {len(C)}, expected {seed_bytes}")
            ktp.key_len = key_bytes
            fixed_number = ktp.next_key()
            if len(fixed_number) != key_bytes:
                raise ValueError(
                    f"Fixed key length is {len(fixed_number)}, expected {key_bytes}"
                )
            ktp.key_len = seed_bytes

            seed_fixed_int = int.from_bytes(
                C, byteorder="little") + int.from_bytes(fixed_number,
                                                        byteorder="little")
            seed_fixed = seed_fixed_int.to_bytes(seed_bytes,
                                                 byteorder="little")
        else:
            # In fixed-vs-random SEED mode we use only one fixed constant:
            #    1. seed_fixed - A 320 bit constant used to derive the fixed key
            #                    for the fixed set of measurements. Note that in
            #                    this set, the fixed key is equal to:
            #                    seed_fixed mod curve_order_n
            seed_fixed = ktp.next_key()
            if len(seed_fixed) != seed_bytes:
                raise ValueError(
                    f"Fixed seed length is {len(seed_fixed)}, expected {seed_bytes}"
                )

        # Expected key is `seed mod n`, where n is the order of the curve and
        # `seed` is interpreted as little-endian.
        expected_fixed_key = (int.from_bytes(seed_fixed, byteorder="little") %
                              curve_order_n)

        sample_fixed = 1
        # Loop to collect each power trace
        for _ in tqdm(range(capture_cfg["num_traces"]),
                      desc="Capturing",
                      ncols=80):

            ot.scope.adc.offset = ot.offset_samples

            if capture_cfg["masks_off"] is True:
                # Use a constant mask for each trace
                mask = bytearray(
                    capture_cfg["plain_text_len_bytes"])  # all zeros
            else:
                # Generate a new random mask for each trace.
                mask = ktp.next_text()

            tqdm.write("Starting new trace....")
            tqdm.write(f"mask   = {mask.hex()}")

            if capture_cfg["test_type"] == "KEY":
                # In fixed-vs-random KEY mode, the fixed set of measurements is
                # generated using the fixed 320 bit seed. The random set of
                # measurements is generated in two steps:
                #    1. Choose a random 256 bit number r
                #    2. Compute the seed as (C + r) where C is the fixed 320 bit
                #       constant. Note that in this case the used key is equal to
                #       (C + r) mod curve_order_n
                if sample_fixed:
                    seed_used = seed_fixed
                    expected_key = expected_fixed_key
                else:
                    ktp.key_len = key_bytes
                    random_number = ktp.next_key()
                    ktp.key_len = seed_bytes
                    seed_used_int = int.from_bytes(
                        C, byteorder="little") + int.from_bytes(
                            random_number, byteorder="little")
                    seed_used = seed_used_int.to_bytes(seed_bytes,
                                                       byteorder="little")
                    expected_key = (
                        int.from_bytes(seed_used, byteorder="little") %
                        curve_order_n)
            else:
                # In fixed-vs-random SEED mode, the fixed set of measurements is
                # generated using the fixed 320 bit seed. The random set of
                # measurements is generated using a random 320 bit seed. In both
                # cases, the used key is equal to:
                #    seed mod curve_order_n
                if sample_fixed:
                    seed_used = seed_fixed
                    expected_key = expected_fixed_key
                else:
                    seed_used = ktp.next_key()
                    expected_key = (
                        int.from_bytes(seed_used, byteorder="little") %
                        curve_order_n)

            # Decide for next round if we use the fixed or a random seed.
            sample_fixed = random.randint(0, 1)

            # Send the seed to ibex.
            # Ibex receives the seed and the mask and computes the two shares as:
            #     Share0 = seed XOR mask
            #     Share1 = mask
            # These shares are then forwarded to OTBN.
            ot.target.simpleserial_write("x", seed_used)
            tqdm.write(f"seed   = {seed_used.hex()}")

            # Check for errors.
            err = ot.target.read()
            if err:
                raise RuntimeError(f"Error writing seed: {err}")

            # Arm the scope
            ot.scope.arm()

            # Send the mask and start the keygen operation.
            ot.target.simpleserial_write("k", mask)

            # Wait until operation is done.
            ret = ot.scope.capture(poll_done=True)
            if ret:
                raise RuntimeError("Timeout during capture")

            # Check the number of cycles where the trigger signal was high.
            cycles = ot.scope.adc.trig_count
            tqdm.write("Observed number of cycles: %d" % cycles)

            waves = ot.scope.get_last_trace(as_int=True)
            # Read the output, unmask the key, and check if it matches
            # expectations.
            share0 = ot.target.simpleserial_read("r", seed_bytes, ack=False)
            share1 = ot.target.simpleserial_read("r", seed_bytes, ack=False)
            if share0 is None:
                raise RuntimeError("Random share0 is none")
            if share1 is None:
                raise RuntimeError("Random share1 is none")

            d0 = int.from_bytes(share0, byteorder="little")
            d1 = int.from_bytes(share1, byteorder="little")
            actual_key = (d0 + d1) % curve_order_n

            tqdm.write(f"share0 = {share0.hex()}")
            tqdm.write(f"share1 = {share1.hex()}")

            if actual_key != expected_key:
                raise RuntimeError("Bad generated key:\n"
                                   f"Expected: {hex(expected_key)}\n"
                                   f"Actual:   {hex(actual_key)}")

            # Create a chipwhisperer trace object and save it to the project
            # Args/fields of Trace object: waves, textin, textout, key
            textout = share0 + share1  # concatenate bytearrays
            trace = Trace(waves, mask, textout, seed_used)
            check_range(waves, ot.scope.adc.bits_per_sample)
            project.traces.append(trace, dtype=np.uint16)
    elif capture_cfg["app"] == "modinv":
        # Check the lengths in the key/plaintext generator. In this case, "key"
        # is the input to the modinv app. We only use the key part of the ktp
        # to generate the key share inputs to the modinv app.
        if ktp.keyLen() != modinv_share_bytes:
            raise ValueError(
                f"Unexpected input (share) length: {ktp.keyLen()}.\n"
                f"Hint: set key len={modinv_share_bytes} in the configuration file."
            )

        # select the otbn app on the device (0 -> keygen, 1 -> modinv)
        ot.target.simpleserial_write("a", bytearray([0x01]))
        time.sleep(0.3)

        # set PRNG seed to ensure same sequence of randoms for split up measurements
        random.seed(capture_cfg["batch_prng_seed"])

        # set ecc256 fixed key and share inputs
        # (uncomment the desired fixed shares depending on whether
        # you want a random fixed key or a hardcoded fixed key)
        # k_fixed_barray = ktp.next_key()[:256]
        k_fixed_barray = bytearray(
            (0x2648D0D248B70944DFD84C2F85EA5793729112E7CAFA50ABDF7EF8B7594FA2A1
             ).to_bytes(key_bytes, "little"))
        k_fixed = int.from_bytes(k_fixed_barray, byteorder="little")

        print("Fixed input:")
        print("k:  " + hex(k_fixed) + "\n")

        # Expected fixed output is `(k)^(-1) mod n`, where n is the curve order n
        expected_fixed_output = pow(k_fixed, -1, curve_order_n)

        sample_fixed = 1
        # Loop to collect each power trace
        for _ in tqdm(range(capture_cfg["num_traces"]),
                      desc="Capturing",
                      ncols=80):

            ot.scope.adc.offset = ot.offset_samples

            if sample_fixed:
                # Compute the fixed input shares:
                # generate two random 320-bit shares
                input_k0_fixed = ktp.next_key()
                input_k1_fixed = ktp.next_key()
                k0_fixed = int.from_bytes(input_k0_fixed, byteorder="little")
                k1_fixed = int.from_bytes(input_k1_fixed, byteorder="little")
                # adapt share k1 so that k = (k0 + k1) mod n
                k_tmp = (k0_fixed + k1_fixed) % curve_order_n
                k_tmp_diff = (k_fixed - k_tmp) % curve_order_n
                k1_fixed += k_tmp_diff
                if k1_fixed >= pow(2, 320):
                    k1_fixed -= curve_order_n
                input_k1_fixed = bytearray(
                    (k1_fixed).to_bytes(modinv_share_bytes, "little"))
                # Use the fixed input.
                input_k0_used = input_k0_fixed
                input_k1_used = input_k1_fixed
                k_used = k_fixed
                expected_output = expected_fixed_output
            else:
                # Use a random input.
                input_k0_used = ktp.next_key()
                input_k1_used = ktp.next_key()
                # calculate the key from the shares
                k_used = (int.from_bytes(input_k0_used, byteorder="little") +
                          int.from_bytes(input_k1_used,
                                         byteorder="little")) % curve_order_n
                expected_output = pow(k_used, -1, curve_order_n)

            tqdm.write(
                f'k0 = {hex(int.from_bytes(input_k0_used, byteorder="little"))}'
            )
            tqdm.write(
                f'k1 = {hex(int.from_bytes(input_k1_used, byteorder="little"))}'
            )

            # Decide for next round if we use the fixed or a random seed.
            sample_fixed = random.randint(0, 1)

            # Arm the scope
            ot.scope.arm()

            # Start modinv device computation
            ot.target.simpleserial_write("q", input_k0_used + input_k1_used)

            # Wait until operation is done.
            ret = ot.scope.capture(poll_done=True)
            if ret:
                raise RuntimeError("Timeout during capture")

            # Check the number of cycles where the trigger signal was high.
            cycles = ot.scope.adc.trig_count
            tqdm.write("Observed number of cycles: %d" % cycles)

            waves = ot.scope.get_last_trace(as_int=True)
            # Read the output, unmask the key, and check if it matches
            # expectations.
            kalpha_inv = ot.target.simpleserial_read("r", key_bytes, ack=False)
            if kalpha_inv is None:
                raise RuntimeError("Modinv device output (k*alpha)^-1 is none")
            alpha = ot.target.simpleserial_read("r",
                                                modinv_mask_bytes,
                                                ack=False)
            if alpha is None:
                raise RuntimeError("Modinv device output alpha is none")

            # Actual result (kalpha_inv*alpha) mod n:
            actual_output = (int.from_bytes(kalpha_inv, byteorder="little") *
                             int.from_bytes(alpha, byteorder="little") %
                             curve_order_n)

            tqdm.write(f"k^-1  = {hex(actual_output)}\n")

            if actual_output != expected_output:
                raise RuntimeError("Bad computed modinv output:\n"
                                   f"Expected: {hex(expected_output)}\n"
                                   f"Actual:   {hex(actual_output)}")

            # Create a chipwhisperer trace object and save it to the project
            # Args/fields of Trace object: waves, textin, textout, key
            trace = cw.common.traces.Trace(
                waves,
                bytearray(k_used.to_bytes(key_bytes, "little")),
                bytearray(actual_output.to_bytes(key_bytes, "little")),
                bytearray(k_used.to_bytes(key_bytes, "little")),
            )
            check_range(waves, ot.scope.adc.bits_per_sample)
            project.traces.append(trace, dtype=np.uint16)
    else:
        print("Invalid app configured in config file.")

    # Save metadata to project file
    sample_rate = int(round(ot.scope.clock.adc_freq, -6))
    save_metadata(project, device_cfg, capture_cfg, cycles, sample_rate)

    project.save()


@app_capture.command()
def otbn_vertical(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):
    """Capture ECDSA secret key generation traces."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)

    # OTBN's public-key operations might not fit into the sample buffer of the scope
    # These two parameters allows users to conrol the sampling frequency
    #
    # `adc_mul` affects the clock frequency (clock_freq = adc_mul * pll_freq)
    #
    # `decimate` is the ADC downsampling factor that allows us to sample at
    #  every `decimate` cycles.
    if "adc_mul" in ctx.obj.cfg["capture"]:
        ctx.obj.ot.scope.clock.adc_mul = ctx.obj.cfg["capture"]["adc_mul"]
    if "decimate" in ctx.obj.cfg["capture"]:
        ctx.obj.ot.scope.adc.decimate = ctx.obj.cfg["capture"]["decimate"]

    # Print the params
    print(
        f'Target setup with clock frequency {ctx.obj.cfg["device"]["pll_frequency"] / 1000000} MHz'
    )
    print(
        f"Scope setup with sampling rate {ctx.obj.ot.scope.clock.adc_freq} S/s"
    )

    # Call the capture loop
    capture_otbn_vertical(
        ctx.obj.ot,
        ctx.obj.ktp,
        ctx.obj.cfg["device"]["fw_bin"],
        ctx.obj.cfg["device"]["pll_frequency"],
        ctx.obj.cfg["capture"],
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


def capture_otbn_vertical_batch(ot, ktp, capture_cfg, scope_type, device_cfg):
    """A generator for fast capturing otbn vertical (ecc256 keygen + modinv) traces.
    The data collection method is based on the derived test requirements (DTR) for TVLA:
    https://www.rambus.com/wp-content/uploads/2015/08/TVLA-DTR-with-AES.pdf
    The measurements are taken by using either fixed or randomly selected seed.
    In order to simplify the analysis, the first sample has to use fixed seed.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
      capture_cfg: Capture configuration.
      scope_type: cw or waverunner as a scope for batch capture.
    """

    # We need an intuitive, but non default behavor for the kpt.next() interrator.
    # For backwards compatibility this must be set in the capture config file.
    # This is a workaroung for https://github.com/lowRISC/ot-sca/issues/116
    if "use_fixed_key_iter" not in capture_cfg:
        raise RuntimeError("use_fixed_key_iter not se set!")
    if capture_cfg["use_fixed_key_iter"] is not False:
        raise RuntimeError("use_fixed_key_iter must be set to false!")

    # OTBN operations are long. CW-Husky can store only 131070 samples
    # in the non-stream mode.
    fifo_size = 131070
    if ot.num_samples > fifo_size:
        raise RuntimeError("Current setup only supports up to 130k samples")

    # Create a cw project to keep the data and traces
    project = cw.create_project(capture_cfg["project_name"], overwrite=True)

    # Initialize some curve-dependent parameters.
    if capture_cfg["curve"] == "p256":
        curve_order_n = (
            0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551)
        key_bytes = 256 // 8
        seed_bytes = 320 // 8
    else:
        # TODO: add support for P384
        raise NotImplementedError(
            f'Curve {capture_cfg["curve"]} is not supported')

    # Capture traces.
    rem_num_traces = capture_cfg["num_traces"]
    num_segments_storage = 1
    sample_fixed = True

    # cw and waverunner scopes are supported fot batch capture.
    scope = SCOPE_FACTORY[scope_type](ot, capture_cfg, device_cfg)

    # OTBN's public-key operations might not fit into the sample buffer of the scope
    # These two parameters allows users to conrol the sampling frequency
    # `adc_mul` affects the sample frequency (clock_freq = adc_mul * pll_freq)
    # `decimate` is the ADC downsampling factor that allows us to sample at
    #  every `decimate` cycles.
    if "adc_mul" in capture_cfg:
        scope._scope.clock.adc_mul = capture_cfg["adc_mul"]
    if "decimate" in capture_cfg:
        scope._scope.adc.decimate = capture_cfg["decimate"]

    # Print final scope parameter
    print(
        f"Scope setup with final sampling rate of {scope._scope.clock.adc_freq} S/s"
    )

    # register ctrl-c handler to not lose already recorded traces if measurement is aborted
    signal.signal(signal.SIGINT, partial(abort_handler, project))

    if capture_cfg["app"] == "keygen":
        # Check the lengths in the key/plaintext generator. In this case, "key"
        # means seed and "plaintext" means mask.
        if ktp.keyLen() != seed_bytes:
            raise ValueError(
                f"Unexpected seed length: {ktp.keyLen()}.\n"
                f"Hint: set key len={seed_bytes} in the configuration file.")
        if ktp.textLen() != seed_bytes:
            raise ValueError(
                f"Unexpected mask length: {ktp.textLen()}.\n"
                f"Hint: set plaintext len={seed_bytes} in the configuration file."
            )

        # select the otbn app on the device (0 -> keygen, 1 -> modinv)
        ot.target.simpleserial_write("a", bytearray([0x00]))
        time.sleep(0.3)

        # set PRNG seed prior to setting fixed seed to have the same input sequence every time
        # running this function with the same batch_prng_seed
        random.seed(capture_cfg["batch_prng_seed"])
        ot.target.simpleserial_write(
            "s", capture_cfg["batch_prng_seed"].to_bytes(4, "little"))
        time.sleep(0.3)

        # Generate fixed constants for all traces of the keygen operation.

        if capture_cfg["test_type"] == "KEY":
            # In fixed-vs-random KEY mode we use two fixed constants:
            #    1. C - a 320 bit constant redundancy
            #    2. fixed_number - a 256 bit number used to derive the fixed key
            #                      for the fixed set of measurements. Note that in
            #                      this set, the fixed key is equal to
            #                      (C + fixed_number) mod curve_order_n
            C = ktp.next_key()
            if len(C) != seed_bytes:
                raise ValueError(
                    f"Fixed seed length is {len(C)}, expected {seed_bytes}")
            ktp.key_len = key_bytes
            fixed_number = ktp.next_key()
            if len(fixed_number) != key_bytes:
                raise ValueError(
                    f"Fixed key length is {len(fixed_number)}, expected {key_bytes}"
                )
            ktp.key_len = seed_bytes
            C_int = int.from_bytes(C, byteorder="little")
            seed_fixed_int = C_int + int.from_bytes(fixed_number,
                                                    byteorder="little")
            seed_fixed = seed_fixed_int.to_bytes(seed_bytes,
                                                 byteorder="little")

            print("Constant redundancy:")
            print(binascii.b2a_hex(C))
            ot.target.simpleserial_write("c", C_int.to_bytes(40, "little"))
            time.sleep(0.3)
        else:
            # In fixed-vs-random SEED mode we use only one fixed constant:
            #    1. seed_fixed - A 320 bit constant used to derive the fixed key
            #                    for the fixed set of measurements. Note that in
            #                    this set, the fixed key is equal to:
            #                    seed_fixed mod curve_order_n
            seed_fixed = ktp.next_key()
            if len(seed_fixed) != seed_bytes:
                raise ValueError(
                    f"Fixed seed length is {len(seed_fixed)}, expected {seed_bytes}"
                )

        print("Fixed seed:")
        print(binascii.b2a_hex(seed_fixed))
        ot.target.simpleserial_write("x", seed_fixed)
        time.sleep(0.3)

        # enable/disable masking
        if capture_cfg["masks_off"] is True:
            ot.target.simpleserial_write("m", bytearray([0x00]))
        else:
            ot.target.simpleserial_write("m", bytearray([0x01]))
        time.sleep(0.3)

        # Re-seeding the PRNG in the KEY mode. In this mode, the PRNG produces additional 32 bytes
        # to set up the fixed_number.
        # This is a necessary step to sync with the PRNG on the capture side.
        if capture_cfg["test_type"] == "KEY":
            random.seed(capture_cfg["batch_prng_seed"])
            ot.target.simpleserial_write(
                "s", capture_cfg["batch_prng_seed"].to_bytes(4, "little"))
            time.sleep(0.3)

        with tqdm(total=rem_num_traces,
                  desc="Capturing",
                  ncols=80,
                  unit=" traces") as pbar:
            while rem_num_traces > 0:
                # Determine the number of traces for this batch and arm the oscilloscope.
                scope.num_segments = min(rem_num_traces,
                                         scope.num_segments_max)

                scope.arm()
                # Start batch keygen
                if capture_cfg["test_type"] == "KEY":
                    ot.target.simpleserial_write(
                        "e", scope.num_segments_actual.to_bytes(4, "little"))
                else:
                    ot.target.simpleserial_write(
                        "b", scope.num_segments_actual.to_bytes(4, "little"))

                # Transfer traces
                waves = scope.capture_and_transfer_waves()
                assert waves.shape[0] == scope.num_segments

                # Check the number of cycles where the trigger signal was high.
                cycles = ot.scope.adc.trig_count // scope.num_segments_actual
                if rem_num_traces <= scope.num_segments_max:
                    tqdm.write("No. of cycles with trigger high: %d" % cycles)

                seeds = []
                masks = []
                d0s = []
                d1s = []

                batch_digest = None
                for i in range(scope.num_segments_actual):

                    if capture_cfg["test_type"] == "KEY":
                        if sample_fixed:
                            seed_barray = seed_fixed
                            seed = int.from_bytes(seed_barray, "little")
                        else:
                            ktp.key_len = key_bytes
                            random_number = ktp.next_key()
                            ktp.key_len = seed_bytes
                            seed_barray_int = C_int + int.from_bytes(
                                random_number, byteorder="little")
                            seed_barray = seed_barray_int.to_bytes(
                                seed_bytes, byteorder="little")
                            seed = seed_barray_int
                    else:
                        if sample_fixed:
                            seed_barray = seed_fixed
                            seed = int.from_bytes(seed_barray, "little")
                        else:
                            seed_barray = ktp.next_key()
                            seed = int.from_bytes(seed_barray, "little")

                    if capture_cfg["masks_off"] is True:
                        mask_barray = bytearray(
                            capture_cfg["plain_text_len_bytes"])
                        mask = int.from_bytes(mask_barray, "little")
                    else:
                        mask_barray = ktp.next_text()
                        mask = int.from_bytes(mask_barray, "little")

                    masks.append(mask_barray)
                    seed = seed ^ mask

                    # needed to be in sync with ot PRNG and for sample_fixed generation
                    dummy = ktp.next_key()

                    # calculate key shares
                    mod = curve_order_n << ((seed_bytes - key_bytes) * 8)
                    d0 = ((seed ^ mask) - mask) % mod
                    d1 = mask % mod

                    # calculate batch digest
                    batch_digest = d0 if batch_digest is None else d0 ^ batch_digest

                    seeds.append(seed_barray)
                    d0s.append(bytearray(d0.to_bytes(seed_bytes, "little")))
                    d1s.append(bytearray(d1.to_bytes(seed_bytes, "little")))
                    sample_fixed = dummy[0] & 1

                # Check the batch digest to make sure we are in sync.
                check_ciphertext(
                    ot,
                    bytearray(batch_digest.to_bytes(seed_bytes, "little")),
                    seed_bytes,
                )

                num_segments_storage = optimize_cw_capture(
                    project, num_segments_storage)

                # Create a chipwhisperer trace object and save it to the project
                # Args/fields of Trace object: waves, textin, textout, key
                for wave, seed, mask, d0, d1 in zip(waves, seeds, masks, d0s,
                                                    d1s):
                    d = d0 + d1
                    trace = cw.common.traces.Trace(wave, d, mask, seed)
                    project.traces.append(trace, dtype=np.uint16)
                # Update the loop variable and the progress bar.
                rem_num_traces -= scope.num_segments
                pbar.update(scope.num_segments)
    elif capture_cfg["app"] == "modinv":
        print("Batch mode capture is not implemented for 'modinv' app.")
    else:
        print("Invalid app configured in config file.")

    # Before saving the project, re-enable all trace storage segments.
    for s in range(len(project.segments)):
        project.traces.tm.setTraceSegmentStatus(s, True)
    assert len(project.traces) == capture_cfg["num_traces"]

    # Save metadata to project file
    sample_rate = int(round(scope._scope.clock.adc_freq, -6))
    save_metadata(project, device_cfg, capture_cfg, cycles, sample_rate)

    # Save the project to disk.
    project.save()


@app_capture.command()
def otbn_vertical_batch(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
    scope_type: ScopeType = opt_scope_type,
):
    """Capture vertical otbn (ecc256 keygen) traces in batch mode. Fixed vs Random."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)
    capture_otbn_vertical_batch(
        ctx.obj.ot,
        ctx.obj.ktp,
        ctx.obj.cfg["capture"],
        scope_type,
        ctx.obj.cfg["device"],
    )
    capture_end(ctx.obj.cfg)


def capture_ecdsa_sections(ot, fw_bin, pll_frequency, num_sections, secret_k,
                           priv_key_d, msg):
    """A utility function to collect the full OTBN trace section by section

    ECDSA is a long operation (e.g, ECDSA-256 takes ~7M samples) that doesn't fit
    into the 130k-sample trace buffer of CW-Husky. This function allows us
    to collect the full ECDSA trace section by section.

    Args:
          ot: Initialized OpenTitan target.
          fw_bin: ECDSA binary.
          pll_frequency: Output frequency of the FPGA PLL.
                         To capture the long OTBN operations,
                         we need to use different frequency than 100MHz
          num_sections: number of traces sections to collect
                        ECDSA is executed num_sections times
                        At each execution a different offset value is used
          secret_k: ephemeral secret k
          priv_key_d: private key d
          msg: message to be signed

    """

    # Create a temporary buffer to keep the collected sections
    tmp_buffer = np.array([])
    for ii in range(num_sections):

        # For each section ii, set the adc_offset parameter accordingly
        ot.scope.adc.offset = ii * 131070

        # Optional commands to overwrite the default values declared in the C code.
        ot.target.simpleserial_write("d", priv_key_d)
        # Message to sign
        ot.target.simpleserial_write("n", msg)
        # Send the ephemeral secret k and trigger the signature geneartion
        ot.target.simpleserial_write("k", secret_k)

        # Arm the scope
        ot.scope.arm()

        # Start the ECDSA operation
        ot.target.simpleserial_write("p", bytearray([0x01]))

        # Wait until operation is done
        ret = ot.scope.capture(poll_done=True)
        # If getting inconsistent results (e.g. variable number of cycles),
        # adding a sufficient sleep below here appears to fix things
        time.sleep(1)
        if ret:
            raise RuntimeError("Timeout during capture")
        # Check the number of cycles, where the trigger signal was high
        cycles = ot.scope.adc.trig_count
        print("Observed number of cycles: %d" % cycles)

        # Append the section into the waves array
        tmp_buffer = np.append(tmp_buffer,
                               ot.scope.get_last_trace(as_int=True))
    return tmp_buffer


def capture_ecdsa_simple(ot, fw_bin, pll_frequency, capture_cfg):
    """An example capture loop to capture OTBN-ECDSA-256/384 traces.

    Does not use the streaming feature of CW-Husky. he streaming mode puts
    limits on the sampling frequency and bits_per_sample. see this link
    for more details:
    https://rtfm.newae.com/Capture/ChipWhisperer-Husky/#streaming-mode

    Allows a user to set the ephemeral secret scalar k, private key d,
    and message msg. For the corresponding driver, check
    <opentitan_repo_root>/sw/device/ecc_serial.c

    Args:
      ot: Initialized OpenTitan target.
      fw_bin: ECDSA binary.
      pll_frequency: Output frequency of the FPGA PLL.
                     To capture the long OTBN operations,
                     we may need to use different frequency than 100MHz
      capture_cfg: Capture configuration from the yaml file
    """

    # Be sure we don't use the stream mode
    if ot.scope._is_husky:
        ot.scope.adc.stream_mode = False
        ot.scope.adc.bits_per_sample = 12
        ot.scope.adc.samples = 131070
    else:
        raise RuntimeError("Only CW-Husky is supported now")

    # OTBN operations are long. CW-Husky can store only 131070 samples
    # in the non-stream mode. In case we want to collect more samples,
    # we can run the operation multiple times with different adc_offset
    # values.
    # The trace from each run is kept in a section, and all sections are
    # concatenated to create the final trace.
    num_sections = ot.num_samples // 131070
    print(f"num_sections = {num_sections}")

    # Create a cw project to keep the data and traces
    project = cw.create_project(capture_cfg["project_name"], overwrite=True)

    # register ctrl-c handler to not lose already recorded traces if measurement is aborted
    signal.signal(signal.SIGINT, partial(abort_handler, project))

    # Loop to collect each power trace
    for _ in tqdm(range(capture_cfg["num_traces"]), desc="Capturing",
                  ncols=80):

        # This part can be modified to create a new command.
        # For example, a random secret scalar can be set using the following
        #   from numpy.random import default_rng
        #   rng = default_rng()
        #   secret_k0 = bytearray(rng.bytes(32))
        #   secret_k1 = bytearray(rng.bytes(32))
        msg = "Hello OTBN.".encode()
        # ECDSA-384
        if capture_cfg["key_len_bytes"] == 48:
            # Set two shares of the private key d
            priv_key_d0 = bytearray([
                0x6B,
                0x9D,
                0x3D,
                0xAD,
                0x2E,
                0x1B,
                0x8C,
                0x1C,
                0x05,
                0xB1,
                0x98,
                0x75,
                0xB6,
                0x65,
                0x9F,
                0x4D,
                0xE2,
                0x3C,
                0x3B,
                0x66,
                0x7B,
                0xF2,
                0x97,
                0xBA,
                0x9A,
                0xA4,
                0x77,
                0x40,
                0x78,
                0x71,
                0x37,
                0xD8,
                0x96,
                0xD5,
                0x72,
                0x4E,
                0x4C,
                0x70,
                0xA8,
                0x25,
                0xF8,
                0x72,
                0xC9,
                0xEA,
                0x60,
                0xD2,
                0xED,
                0xF5,
            ])
            priv_key_d1 = bytearray([
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
            # Set two shares of the scalar secret_k
            secret_k0 = bytearray([
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
            secret_k1 = bytearray([
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
        # ECDSA-256
        elif capture_cfg["key_len_bytes"] == 32:
            # Set two shares of the private key d
            priv_key_d0 = bytearray([
                0xCD,
                0xB4,
                0x57,
                0xAF,
                0x1C,
                0x9F,
                0x4C,
                0x74,
                0x02,
                0x0C,
                0x7E,
                0x8B,
                0xE9,
                0x93,
                0x3E,
                0x28,
                0x0C,
                0xF0,
                0x18,
                0x0D,
                0xF4,
                0x6C,
                0x0B,
                0xDA,
                0x7A,
                0xBB,
                0xE6,
                0x8F,
                0xB7,
                0xA0,
                0x45,
                0x55,
            ])
            priv_key_d1 = bytearray([
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
            # Set two shares of the scalar secret_k
            secret_k0 = bytearray([
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
            secret_k1 = bytearray([
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
        else:
            raise RuntimeError("priv_key_d must be either 32B or 48B")

        # Combine the two shares of d and k
        priv_key_d = priv_key_d0 + priv_key_d1
        secret_k = secret_k0 + secret_k1

        # Create a clean array to keep the collected traces
        waves = np.array([])
        waves = capture_ecdsa_sections(ot, fw_bin, pll_frequency, num_sections,
                                       secret_k, priv_key_d, msg)

        # Read 32 bytes of signature_r and signature_s back from the device
        sig_r = ot.target.simpleserial_read("r",
                                            capture_cfg["output_len_bytes"],
                                            ack=False)
        print(f"sig_r = {''.join('{:02x}'.format(x) for x in sig_r)}")
        sig_s = ot.target.simpleserial_read("r",
                                            capture_cfg["output_len_bytes"],
                                            ack=False)
        print(f"sig_s = {''.join('{:02x}'.format(x) for x in sig_s)}")

        # Create a chipwhisperer trace object and save it to the project
        # Args/fields of Trace object: waves, textin, textout, key
        # TODO: Change the assignments based on the requirements of the SCA
        trace = Trace(waves, secret_k, sig_s, priv_key_d)
        check_range(waves, ot.scope.adc.bits_per_sample)
        project.traces.append(trace, dtype=np.uint16)
        # Delete the objects before the next iteration
        del waves
        del trace
    project.save()


@app_capture.command()
def ecdsa_simple(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):

    # OTBN-specific settings
    """Capture OTBN-ECDSA-256/384 traces from a target that runs the `ecc384_serial` program."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)

    # OTBN's public-key operations might not fit into the sample buffer of the scope
    # These two parameters allows users to conrol the sampling frequency
    #
    # `adc_mul` affects the clock frequency (clock_freq = adc_mul * pll_freq)
    #
    # `decimate` is the ADC downsampling factor that allows us to sample at
    #  every `decimate` cycles.
    if "adc_mul" in ctx.obj.cfg["capture"]:
        ctx.obj.ot.scope.clock.adc_mul = ctx.obj.cfg["capture"]["adc_mul"]
    if "decimate" in ctx.obj.cfg["capture"]:
        ctx.obj.ot.scope.adc.decimate = ctx.obj.cfg["capture"]["decimate"]
    # Print the params
    print(
        f'Target setup with clock frequency {ctx.obj.cfg["device"]["pll_frequency"] / 1000000} MHz'
    )
    print(
        f"Scope setup with sampling rate {ctx.obj.ot.scope.clock.adc_freq} S/s"
    )

    # Call the capture loop
    capture_ecdsa_simple(
        ctx.obj.ot,
        ctx.obj.cfg["device"]["fw_bin"],
        ctx.obj.cfg["device"]["pll_frequency"],
        ctx.obj.cfg["capture"],
    )
    capture_end(ctx.obj.cfg)


def capture_ecdsa_stream(ot, fw_bin, pll_frequency, capture_cfg):
    """An example capture loop to capture OTBN-ECDSA-256/384 traces.

    Utilizes the streaming feature of CW-Husky. The streaming mode puts
    limits on the sampling frequency and bits_per_sample. see this link
    for more details:
    https://rtfm.newae.com/Capture/ChipWhisperer-Husky/#streaming-mode

    Allows a user to set the ephemeral secret scalar k, private key d,
    and message msg. For the corresponding driver, check
    <opentitan_repo_root>/sw/device/ecc384_serial.c

    Args:
      ot: Initialized OpenTitan target.
      fw_bin: Key and plaintext generator.
      pll_frequency: Output frequency of the FPGA PLL.
                     To capture the long OTBN operations,
                     we may need to use different frequency than 100MHz
      capture_cfg: Capture configuration from the yaml file
    """

    project = cw.create_project(capture_cfg["project_name"], overwrite=True)

    # Enable the streaming mode
    if ot.scope._is_husky:
        ot.scope.adc.stream_mode = True
        # In the stream mode, there is a tradeoff between
        # the adc-resolution (8/10/12 bits) and sampling-frequency (< 25MHz)
        # https://rtfm.newae.com/Capture/ChipWhisperer-Husky/#streaming-mode
        ot.scope.adc.bits_per_sample = 12
    else:
        # We support only CW-Husky for now.
        raise RuntimeError("Only CW-Husky is supported now")

    # register ctrl-c handler to not lose already recorded traces if measurement is aborted
    signal.signal(signal.SIGINT, partial(abort_handler, project))

    # Loop to collect traces
    for _ in tqdm(range(capture_cfg["num_traces"]), desc="Capturing",
                  ncols=80):
        # This part can be modified to create a new command.
        # For example, a random secret scalar can be set using the following
        #   from numpy.random import default_rng
        #   rng = default_rng()
        #   secret_k0 = bytearray(rng.bytes(32))
        #   secret_k1 = bytearray(rng.bytes(32))
        msg = "Hello OTBN.".encode()
        # ECDSA-384
        if capture_cfg["key_len_bytes"] == 48:
            # Set two shares of the private key d
            priv_key_d0 = bytearray([
                0x6B,
                0x9D,
                0x3D,
                0xAD,
                0x2E,
                0x1B,
                0x8C,
                0x1C,
                0x05,
                0xB1,
                0x98,
                0x75,
                0xB6,
                0x65,
                0x9F,
                0x4D,
                0xE2,
                0x3C,
                0x3B,
                0x66,
                0x7B,
                0xF2,
                0x97,
                0xBA,
                0x9A,
                0xA4,
                0x77,
                0x40,
                0x78,
                0x71,
                0x37,
                0xD8,
                0x96,
                0xD5,
                0x72,
                0x4E,
                0x4C,
                0x70,
                0xA8,
                0x25,
                0xF8,
                0x72,
                0xC9,
                0xEA,
                0x60,
                0xD2,
                0xED,
                0xF5,
            ])
            priv_key_d1 = bytearray([
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
            # Set two shares of the scalar secret_k
            secret_k0 = bytearray([
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
            secret_k1 = bytearray([
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
        # ECDSA-256
        elif capture_cfg["key_len_bytes"] == 32:
            # Set two shares of the private key d
            priv_key_d0 = bytearray([
                0xCD,
                0xB4,
                0x57,
                0xAF,
                0x1C,
                0x9F,
                0x4C,
                0x74,
                0x02,
                0x0C,
                0x7E,
                0x8B,
                0xE9,
                0x93,
                0x3E,
                0x28,
                0x0C,
                0xF0,
                0x18,
                0x0D,
                0xF4,
                0x6C,
                0x0B,
                0xDA,
                0x7A,
                0xBB,
                0xE6,
                0x8F,
                0xB7,
                0xA0,
                0x45,
                0x55,
            ])
            priv_key_d1 = bytearray([
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
            # Set two shares of the scalar secret_k
            secret_k0 = bytearray([
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
            secret_k1 = bytearray([
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ])
        else:
            raise RuntimeError("priv_key_d must be either 32B or 48B")

        # Combine the two shares of d and k
        priv_key_d = priv_key_d0 + priv_key_d1
        secret_k = secret_k0 + secret_k1

        # Create an arrey to keep the traces
        waves = np.array([])

        # Optional commands to overwrite the default values declared in the C code.
        ot.target.simpleserial_write("d", priv_key_d)
        # Message to sign
        ot.target.simpleserial_write("n", msg)
        # Send the ephemeral secret k and trigger the signature geneartion
        ot.target.simpleserial_write("k", secret_k)

        time.sleep(0.2)

        # Arm the scope
        ot.scope.arm()

        # Start the ECDSA operation
        ot.target.simpleserial_write("p", bytearray([0x01]))

        # Wait until operation is done
        ret = ot.scope.capture(poll_done=True)
        # If getting inconsistent results (e.g. variable number of cycles),
        # adding a sufficient sleep below here appears to fix things
        time.sleep(1)
        if ret:
            raise RuntimeError("Timeout during capture")
        # Check the number of cycles, where the trigger signal was high
        cycles = ot.scope.adc.trig_count
        print("Observed number of cycles: %d" % cycles)

        # Append the section into the waves array
        waves = np.append(waves, ot.scope.get_last_trace(as_int=True))

        # Read signature_r and signature_s back from the device
        sig_r = ot.target.simpleserial_read("r",
                                            capture_cfg["output_len_bytes"],
                                            ack=False)
        print(f"sig_r = {''.join('{:02x}'.format(x) for x in sig_r)}")
        sig_s = ot.target.simpleserial_read("r",
                                            capture_cfg["output_len_bytes"],
                                            ack=False)
        print(f"sig_s = {''.join('{:02x}'.format(x) for x in sig_s)}")

        # Create a chipwhisperer trace object and save it to the project
        # Args/fields of Trace object: waves, textin, textout, key
        # TODO: Change the assignments based on the requirements of the SCA
        trace = Trace(waves, secret_k, sig_s, priv_key_d)
        check_range(waves, ot.scope.adc.bits_per_sample)
        project.traces.append(trace, dtype=np.uint16)
        # Delete the objects before the next iteration
        del waves
        del trace
    project.save()


@app_capture.command()
def ecdsa_stream(
    ctx: typer.Context,
    force_program_bitstream: bool = opt_force_program_bitstream,
    num_traces: int = opt_num_traces,
    plot_traces: int = opt_plot_traces,
):
    """Use cw-husky stream mode to capture OTBN-ECDSA-256/384 traces
    from a target that runs the `ecc384_serial` program."""
    capture_init(ctx, force_program_bitstream, num_traces, plot_traces)

    # OTBN's public-key operations might not fit into the sample buffer of the scope
    # These two parameters allows users to conrol the sampling frequency
    #
    # `adc_mul` affects the clock frequency (clock_freq = adc_mul * pll_freq)
    #
    # `decimate` is the ADC downsampling factor that allows us to sample at
    #  every `decimate` cycles.
    if "adc_mul" in ctx.obj.cfg["capture"]:
        ctx.obj.ot.scope.clock.adc_mul = ctx.obj.cfg["capture"]["adc_mul"]
    if "decimate" in ctx.obj.cfg["capture"]:
        ctx.obj.ot.scope.adc.decimate = ctx.obj.cfg["capture"]["decimate"]
    print(
        f'Target setup with clock frequency {ctx.obj.cfg["device"]["pll_frequency"] / 1000000} MHz'
    )
    print(
        f"Scope setup with sampling rate {ctx.obj.ot.scope.clock.adc_freq} S/s"
    )

    capture_ecdsa_stream(
        ctx.obj.ot,
        ctx.obj.cfg["device"]["fw_bin"],
        ctx.obj.cfg["device"]["pll_frequency"],
        ctx.obj.cfg["capture"],
    )
    capture_end(ctx.obj.cfg)


@app.command("plot")
def plot_cmd(ctx: typer.Context, num_traces: int = opt_plot_traces):
    """Plots previously captured traces."""

    if num_traces is not None:
        ctx.obj.cfg["plot_capture"]["num_traces"] = num_traces
    plot_results(ctx.obj.cfg["plot_capture"],
                 ctx.obj.cfg["capture"]["project_name"])


@app.callback()
def main(ctx: typer.Context, cfg_file: str = None):
    """Capture traces for side-channel analysis."""

    cfg_file = "capture_aes_cw310.yaml" if cfg_file is None else cfg_file
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Store config in the user data attribute (`obj`) of the context.
    ctx.obj = SimpleNamespace(cfg=cfg)


if __name__ == "__main__":
    app()
