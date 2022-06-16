#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import binascii
from Crypto.Cipher import AES
from enum import Enum
import numpy as np
import scared
import time
from tqdm import tqdm
import yaml
from types import SimpleNamespace
import typer
from pathlib import Path

import chipwhisperer as cw
import random
from waverunner import WaveRunner
from cw_segmented import CwSegmented

from util import device
from util import plot
from pyXKCP import pyxkcp


class ScopeType(str, Enum):
    cw = "cw"
    waverunner = "waverunner"


app = typer.Typer(add_completion=False)
# To be able to define subcommands for the "capture" command.
app_capture = typer.Typer()
app.add_typer(app_capture, name="capture", help="Capture traces for SCA")
# Shared options for "capture aes" and "capture sha3".
opt_num_traces = typer.Option(None, help="Number of traces to capture.")
opt_plot_traces = typer.Option(None, help="Number of traces to plot.")
opt_scope_type = typer.Option(ScopeType.cw, help=("Scope type"))
opt_ciphertexts_store = typer.Option(False, help=("Store all ciphertexts for batch capture."))


def create_waverunner(ot, capture_cfg):
    """Create a WaveRunner object to be used for batch capture."""
    return WaveRunner(capture_cfg["waverunner_ip"])


def create_cw_segmented(ot, capture_cfg):
    """Create CwSegmented object to be used for batch capture."""
    return CwSegmented(num_samples=capture_cfg["num_samples"],
                       offset=capture_cfg["offset"],
                       scope_gain=capture_cfg["scope_gain"],
                       scope=ot.scope)


SCOPE_FACTORY = {
    ScopeType.cw: create_cw_segmented,
    ScopeType.waverunner: create_waverunner,
}


# Note: initialize_capture and plot_results are also used by other scripts.
def initialize_capture(device_cfg, capture_cfg):
    """Initialize capture."""
    ot = device.OpenTitan(device_cfg["fpga_bitstream"],
                          device_cfg["fw_bin"],
                          device_cfg["pll_frequency"],
                          device_cfg["baudrate"],
                          capture_cfg["scope_gain"],
                          capture_cfg["num_samples"],
                          capture_cfg["offset"],
                          capture_cfg["output_len_bytes"])
    print(f'Scope setup with sampling rate {ot.scope.clock.adc_freq} S/s')
    # Ping target
    print('Reading from FPGA using simpleserial protocol.')
    version = None
    ping_cnt = 0
    while not version:
        if ping_cnt == 3:
            raise RuntimeError(
                f'No response from the target (attempts: {ping_cnt}).')
        ot.target.write('v' + '\n')
        ping_cnt += 1
        time.sleep(0.5)
        version = ot.target.read().strip()
    print(f'Target simpleserial version: {version} (attempts: {ping_cnt}).')
    return ot


def check_range(waves, bits_per_sample):
    """ The ADC output is in the interval [0, 2**bits_per_sample-1]. Check that the recorded
        traces are within [1, 2**bits_per_sample-2] to ensure the ADC doesn't saturate. """
    adc_range = np.array([0, 2**bits_per_sample])
    if not (np.all(np.greater(waves[:], adc_range[0]))
            and np.all(np.less(waves[:], adc_range[1] - 1))):
        print('\nWARNING: Some samples are outside the range [' +
              str(adc_range[0] + 1) + ', ' + str(adc_range[1] - 2) + '].')
        print('The ADC has a max range of [' +
              str(adc_range[0]) + ', ' + str(adc_range[1]-1) + '] and might saturate.')
        print('It is recommended to reduce the scope gain (see device.py).')


def plot_results(plot_cfg, project_name):
    """Plots traces from `project_name` using `plot_cfg` settings."""
    project = cw.open_project(project_name)

    if len(project.waves) == 0:
        print('Project contains no traces. Did the capture fail?')
        return

    plot.save_plot_to_file(project.waves, plot_cfg["num_traces"],
                           plot_cfg["trace_image_filename"])
    print(
        f'Created plot with {plot_cfg["num_traces"]} traces: '
        f'{Path(plot_cfg["trace_image_filename"]).resolve()}'
    )


@app.command()
def init(ctx: typer.Context):
    """Initalize target for SCA."""
    initialize_capture(ctx.obj.cfg["device"], ctx.obj.cfg["capture"])


def capture_init(ctx, num_traces, plot_traces):
    """Initializes the user data stored in the context and programs the target."""
    cfg = ctx.obj.cfg
    if num_traces:
        cfg["capture"]["num_traces"] = num_traces

    if plot_traces:
        cfg["plot_capture"]["show"] = True
        cfg["plot_capture"]["num_traces"] = plot_traces

    # Key and plaintext generator
    ctx.obj.ktp = cw.ktp.Basic()
    ctx.obj.ktp.key_len = cfg["capture"]["key_len_bytes"]
    ctx.obj.ktp.text_len = cfg["capture"]["plain_text_len_bytes"]

    ctx.obj.ot = initialize_capture(cfg["device"], cfg["capture"])


def capture_loop(trace_gen, ot, capture_cfg):
    """Main capture loop.

    Args:
      trace_gen: A trace generator.
      capture_cfg: Capture configuration.
    """
    project = cw.create_project(capture_cfg["project_name"], overwrite=True)
    for _ in tqdm(range(capture_cfg["num_traces"]), desc='Capturing', ncols=80):
        traces = next(trace_gen)
        check_range(traces.wave, ot.scope.adc.bits_per_sample)
        project.traces.append(traces, dtype=np.uint16)
    project.save()


def capture_end(cfg):
    if cfg["plot_capture"]["show"]:
        plot_results(cfg["plot_capture"], cfg["capture"]["project_name"])


def capture_aes_random(ot, ktp):
    """A generator for capturing AES traces.
    Fixed key, Random texts.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
    """
    key, _ = ktp.next()
    tqdm.write(f'Using key: {binascii.b2a_hex(bytes(key))}')
    cipher = AES.new(bytes(key), AES.MODE_ECB)
    while True:
        _, text = ktp.next()
        ret = cw.capture_trace(ot.scope, ot.target, text, key, ack=False, as_int=True)
        if not ret:
            raise RuntimeError('Capture failed.')
        expected = binascii.b2a_hex(cipher.encrypt(bytes(text)))
        got = binascii.b2a_hex(ret.textout)
        if got != expected:
            raise RuntimeError(f'Bad ciphertext: {got} != {expected}.')
        yield ret


@app_capture.command()
def aes_random(ctx: typer.Context,
               num_traces: int = opt_num_traces,
               plot_traces: int = opt_plot_traces):
    """Capture AES traces from a target that runs the `aes_serial` program."""
    capture_init(ctx, num_traces, plot_traces)
    capture_loop(capture_aes_random(ctx.obj.ot, ctx.obj.ktp), ctx.obj.ot, ctx.obj.cfg["capture"])
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
            project.traces.tm.setTraceSegmentStatus(num_segments_storage - 2, False)
        num_segments_storage = len(project.segments)
    return num_segments_storage


def check_ciphertext(ot, expected_last_ciphertext, only_first_word):
    """Check the first word of the last ciphertext in a batch to make sure we are in sync."""
    ciphertext_len = 16
    if only_first_word:
        ciphertext_len = 4
    actual_last_ciphertext = ot.target.simpleserial_read("r", ciphertext_len, ack=False)
    assert actual_last_ciphertext == expected_last_ciphertext[0:ciphertext_len], (
        f"Incorrect encryption result!\n"
        f"actual: {actual_last_ciphertext}\n"
        f"expected: {expected_last_ciphertext}"
    )


def capture_aes_random_batch(ot, ktp, capture_cfg, scope_type):
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
    tqdm.write(f'Using key: {binascii.b2a_hex(bytes(key))}')
    ot.target.simpleserial_write("k", key)
    # Seed the target's PRNG
    ot.target.simpleserial_write("s", capture_cfg["batch_prng_seed"].to_bytes(4, "little"))

    # Create the ChipWhisperer project.
    project = cw.create_project(capture_cfg["project_name"], overwrite=True)

    # Capture traces.
    rem_num_traces = capture_cfg["num_traces"]
    num_segments_storage = 1
    # cw and waverunner scopes are supported fot batch capture.
    scope = SCOPE_FACTORY[scope_type](ot, capture_cfg)
    with tqdm(total=rem_num_traces, desc="Capturing", ncols=80, unit=" traces") as pbar:
        while rem_num_traces > 0:
            # Determine the number of traces for this batch and arm the oscilloscope.
            scope.num_segments = min(rem_num_traces, scope.num_segments_max)
            scope.arm()

            # Start batch encryption.
            ot.target.simpleserial_write(
                "b", scope.num_segments_actual.to_bytes(4, "little")
            )

            # Transfer traces
            waves = scope.capture_and_transfer_waves()
            assert waves.shape[0] == scope.num_segments
            # Check that the ADC didn't saturate when recording this batch.
            check_range(waves, ot.scope.adc.bits_per_sample)

            # Generate plaintexts and ciphertexts to compare with the batch encryption results.
            plaintexts = [ktp.next()[1] for _ in range(scope.num_segments_actual)]
            ciphertexts = [
                bytearray(c)
                for c in scared.aes.base.encrypt(
                    np.asarray(plaintexts), np.asarray(key)
                )
            ]

            check_ciphertext(ot, ciphertexts[-1], True)

            num_segments_storage = optimize_cw_capture(project, num_segments_storage)

            # Add traces of this batch to the project.
            for wave, plaintext, ciphertext in zip(waves, plaintexts, ciphertexts):
                project.traces.append(
                    cw.common.traces.Trace(wave, plaintext, ciphertext, key),
                    dtype=np.uint16
                )

            # Update the loop variable and the progress bar.
            rem_num_traces -= scope.num_segments
            pbar.update(scope.num_segments)

    # Before saving the project, re-enable all trace storage segments.
    for s in range(len(project.segments)):
        project.traces.tm.setTraceSegmentStatus(s, True)
    assert len(project.traces) == capture_cfg["num_traces"]

    # Save the project to disk.
    project.save()


@app_capture.command()
def aes_random_batch(ctx: typer.Context,
                     num_traces: int = opt_num_traces,
                     plot_traces: int = opt_plot_traces,
                     scope_type: ScopeType = opt_scope_type):
    """Capture AES traces in batch mode. Fixed key random texts."""
    capture_init(ctx, num_traces, plot_traces)
    capture_aes_random_batch(ctx.obj.ot, ctx.obj.ktp, ctx.obj.cfg["capture"], scope_type)
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
    key_generation = bytearray([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF1,
                                0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xE0, 0xF0])
    cipher_gen = AES.new(bytes(key_generation), AES.MODE_ECB)
    text_fixed = bytearray([0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
                            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA])
    text_random = bytearray([0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC,
                             0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC])
    key_fixed = bytearray([0x81, 0x1E, 0x37, 0x31, 0xB0, 0x12, 0x0A, 0x78,
                           0x42, 0x78, 0x1E, 0x22, 0xB2, 0x5C, 0xDD, 0xF9])
    key_random = bytearray([0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53,
                            0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53])

    tqdm.write(f'Fixed key: {binascii.b2a_hex(bytes(key_fixed))}')

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
        ret = cw.capture_trace(ot.scope, ot.target, text, key, ack=False, as_int=True)
        if not ret:
            raise RuntimeError('Capture failed.')
        expected = binascii.b2a_hex(cipher.encrypt(bytes(text)))
        got = binascii.b2a_hex(ret.textout)
        if got != expected:
            raise RuntimeError(f'Bad ciphertext: {got} != {expected}.')
        yield ret


@app_capture.command()
def aes_fvsr_key(ctx: typer.Context,
                 num_traces: int = opt_num_traces,
                 plot_traces: int = opt_plot_traces):
    """Capture AES traces from a target that runs the `aes_serial` program."""
    capture_init(ctx, num_traces, plot_traces)
    capture_loop(capture_aes_fvsr_key(ctx.obj.ot), ctx.obj.ot, ctx.obj.cfg["capture"])
    capture_end(ctx.obj.cfg)


def capture_aes_fvsr_key_batch(ot, ktp, capture_cfg, scope_type, gen_ciphertexts):
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
    # Seed the target's PRNG
    ot.target.simpleserial_write("s", capture_cfg["batch_prng_seed"].to_bytes(4, "little"))

    key_fixed = bytearray([0x81, 0x1E, 0x37, 0x31, 0xB0, 0x12, 0x0A, 0x78,
                           0x42, 0x78, 0x1E, 0x22, 0xB2, 0x5C, 0xDD, 0xF9])
    tqdm.write(f'Fixed key: {binascii.b2a_hex(bytes(key_fixed))}')
    ot.target.simpleserial_write("t", key_fixed)

    sample_fixed = 1
    is_first_batch = True

    # Create the ChipWhisperer project.
    project = cw.create_project(capture_cfg["project_name"], overwrite=True)
    # Capture traces.
    rem_num_traces = capture_cfg["num_traces"]
    num_segments_storage = 1
    # cw and waverunner scopes are supported for batch capture.
    scope = SCOPE_FACTORY[scope_type](ot, capture_cfg)

    with tqdm(total=rem_num_traces, desc="Capturing", ncols=80, unit=" traces") as pbar:
        while rem_num_traces > 0:
            # Determine the number of traces for this batch and arm the oscilloscope.
            scope.num_segments = min(rem_num_traces, scope.num_segments_max)
            scope.arm()

            # Start batch encryption. In order to increase capture rate, after the first batch
            # encryption, the device will start automatically to generate random keys and plaintexts
            # when this script is getting waves from the scope.
            if is_first_batch:
                ot.target.simpleserial_write("g", scope.num_segments_actual.to_bytes(4, "little"))
                is_first_batch = False
            ot.target.simpleserial_write("f", scope.num_segments_actual.to_bytes(4, "little"))

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
                    ciphertext = np.asarray(scared.aes.base.encrypt(plaintext, key))
                    ciphertexts.append(ciphertext)
                sample_fixed = plaintext[0] & 0x1
            if gen_ciphertexts:
                expected_last_ciphertext = ciphertexts[-1]
            else:
                expected_last_ciphertext = np.asarray(scared.aes.base.encrypt(plaintext, key))

            # dummy operation for synching PRNGs with the firmware when using key sharing
            for ii in range(scope.num_segments_actual):
                _ = np.asarray(ktp.next()[1])

            check_ciphertext(ot, expected_last_ciphertext, True)

            num_segments_storage = optimize_cw_capture(project, num_segments_storage)

            # Add traces of this batch to the project. By default we don't store the ciphertexts as
            # generating them on the host as well as transferring them over from the target
            # substantially reduces capture performance. It should therefore only be enabled if
            # absolutely needed.
            if gen_ciphertexts:
                for wave, plaintext, ciphertext, key in zip(waves, plaintexts, ciphertexts, keys):
                    project.traces.append(
                        cw.common.traces.Trace(wave, plaintext, ciphertext, key),
                        dtype=np.uint16
                    )
            else:
                for wave, plaintext, key in zip(waves, plaintexts, keys):
                    project.traces.append(
                        cw.common.traces.Trace(wave, plaintext, None, key),
                        dtype=np.uint16
                    )

            # Update the loop variable and the progress bar.
            rem_num_traces -= scope.num_segments
            pbar.update(scope.num_segments)

    # Before saving the project, re-enable all trace storage segments.
    for s in range(len(project.segments)):
        project.traces.tm.setTraceSegmentStatus(s, True)
    assert len(project.traces) == capture_cfg["num_traces"]

    # Save the project to disk.
    project.save()


@app_capture.command()
def aes_fvsr_key_batch(ctx: typer.Context,
                       num_traces: int = opt_num_traces,
                       plot_traces: int = opt_plot_traces,
                       scope_type: ScopeType = opt_scope_type,
                       gen_ciphertexts: bool = opt_ciphertexts_store):
    """Capture AES traces in batch mode. Fixed vs random keys, random texts."""
    capture_init(ctx, num_traces, plot_traces)
    capture_aes_fvsr_key_batch(
        ctx.obj.ot, ctx.obj.ktp, ctx.obj.cfg["capture"], scope_type, gen_ciphertexts
    )
    capture_end(ctx.obj.cfg)


@app_capture.command()
def aes_mix_column(ctx: typer.Context,
                   num_traces: int = opt_num_traces,
                   plot_traces: int = opt_plot_traces):
    """Capture AES traces. Fixed key, Random texts. 4 sets of traces. Mix Column HD CPA Attack.
    Attack implemented by ChipWhisperer:
    Repo: https://github.com/newaetech/chipwhisperer-jupyter/blob/master/experiments/MixColumn%20Attack.ipynb # noqa: E501
    Reference: https://eprint.iacr.org/2019/343.pdf
    See mix_columns_cpa_attack.py for attack portion.
    """
    capture_init(ctx, num_traces, plot_traces)

    ctx.obj.ktp = cw.ktp.VarVec()
    ctx.obj.ktp.key_len = ctx.obj.cfg['capture']['key_len_bytes']
    ctx.obj.ktp.text_len = ctx.obj.cfg['capture']['plain_text_len_bytes']
    project_name = ctx.obj.cfg["capture"]['project_name']
    # For each iteration, run a capture where only the bytes specified in
    # `text_range` are set to random values. All other bytes are set to a
    # fixed value.
    for var_vec in range(4):
        ctx.obj.cfg['capture']['project_name'] = f'{project_name}_{var_vec}'
        ctx.obj.ktp.var_vec = var_vec
        capture_loop(capture_aes_random(
            ctx.obj.ot, ctx.obj.ktp), ctx.obj.ot, ctx.obj.cfg["capture"]
        )

    capture_end(ctx.obj.cfg)


def capture_sha3_random(ot, ktp):
    """A generator for capturing SHA3 (KMAC) traces.
    Fixed key, Random texts.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
    """
    key, _ = ktp.next()
    tqdm.write(f'Using key: {binascii.b2a_hex(bytes(key))}')
    ot.target.simpleserial_write('k', key)
    while True:
        _, text = ktp.next()
        ret = cw.capture_trace(ot.scope, ot.target, text, key, ack=False, as_int=True)
        if not ret:
            raise RuntimeError('Capture failed.')
        expected = binascii.b2a_hex(pyxkcp.kmac128(key, ktp.key_len,
                                                   text, ktp.text_len,
                                                   ot.target.output_len,
                                                   b'\x00', 0))
        got = binascii.b2a_hex(ret.textout)
        if got != expected:
            raise RuntimeError(f'Bad digest: {got} != {expected}.')
        yield ret


@app_capture.command()
def sha3_random(ctx: typer.Context,
                num_traces: int = opt_num_traces,
                plot_traces: int = opt_plot_traces):
    """Capture SHA3 (KMAC) traces from a target that runs the `sha3_serial` program."""
    capture_init(ctx, num_traces, plot_traces)
    capture_loop(capture_sha3_random(ctx.obj.ot, ctx.obj.ktp), ctx.obj.ot, ctx.obj.cfg["capture"])
    capture_end(ctx.obj.cfg)


def capture_sha3_fvsr_key(ot):
    """A generator for capturing SHA3 (KMAC) traces.
    The data collection method is based on the derived test requirements (DTR) for TVLA:
    https://www.rambus.com/wp-content/uploads/2015/08/TVLA-DTR-with-AES.pdf
    The measurements are taken by using either fixed or randomly selected key.
    In order to simplify the analysis, the first sample has to use fixed key.
    The initial key and plaintext values as well as the derivation methods are as specified in the
    DTR.

    Args:
      ot: Initialized OpenTitan target.
    """

    key_generation = bytearray([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF1,
                                0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xE0, 0xF0])
    cipher = AES.new(bytes(key_generation), AES.MODE_ECB)
    text_fixed = bytearray([0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
                            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA])
    text_random = bytearray([0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC,
                             0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC])
    key_fixed = bytearray([0x81, 0x1E, 0x37, 0x31, 0xB0, 0x12, 0x0A, 0x78,
                           0x42, 0x78, 0x1E, 0x22, 0xB2, 0x5C, 0xDD, 0xF9])
    key_random = bytearray([0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53,
                            0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53])

    key_len = len(key_fixed)
    text_len = len(text_fixed)

    tqdm.write(f'Using fixed key: {binascii.b2a_hex(bytes(key_fixed))}')

    # Start sampling with the fixed key.
    sample_fixed = 1
    while True:
        if sample_fixed:
            text_fixed = bytearray(cipher.encrypt(text_fixed))
            ret = cw.capture_trace(ot.scope, ot.target, text_fixed, key_fixed, ack=False,
                                   as_int=True)
            if not ret:
                raise RuntimeError('Capture failed.')
            expected = binascii.b2a_hex(pyxkcp.kmac128(key_fixed, key_len,
                                                       text_fixed, text_len,
                                                       ot.target.output_len,
                                                       b'\x00', 0))
            got = binascii.b2a_hex(ret.textout)
        else:
            text_random = bytearray(cipher.encrypt(text_random))
            key_random = bytearray(cipher.encrypt(key_random))
            ret = cw.capture_trace(ot.scope, ot.target, text_random, key_random, ack=False,
                                   as_int=True)
            if not ret:
                raise RuntimeError('Capture failed.')
            expected = binascii.b2a_hex(pyxkcp.kmac128(key_random, key_len,
                                                       text_random, text_len,
                                                       ot.target.output_len,
                                                       b'\x00', 0))
            got = binascii.b2a_hex(ret.textout)
        sample_fixed = random.randint(0, 1)
        if got != expected:
            raise RuntimeError(f'Bad digest: {got} != {expected}.')
        yield ret


@app_capture.command()
def sha3_fvsr_key(ctx: typer.Context,
                  num_traces: int = opt_num_traces,
                  plot_traces: int = opt_plot_traces):
    """Capture SHA3 (KMAC) traces from a target that runs the `sha3_serial` program."""
    capture_init(ctx, num_traces, plot_traces)
    capture_loop(capture_sha3_fvsr_key(ctx.obj.ot), ctx.obj.ot, ctx.obj.cfg["capture"])
    capture_end(ctx.obj.cfg)


@app.command("plot")
def plot_cmd(ctx: typer.Context, num_traces: int = opt_plot_traces):
    """Plots previously captured traces."""

    if num_traces is not None:
        ctx.obj.cfg["plot_capture"]["num_traces"] = num_traces
    plot_results(ctx.obj.cfg["plot_capture"], ctx.obj.cfg["capture"]["project_name"])


@app.callback()
def main(ctx: typer.Context, cfg_file: str = None):
    """Capture traces for side-channel analysis."""

    cfg_file = 'capture_aes_cw310.yaml' if cfg_file is None else cfg_file
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Store config in the user data attribute (`obj`) of the context.
    ctx.obj = SimpleNamespace(cfg=cfg)


if __name__ == "__main__":
    app()
