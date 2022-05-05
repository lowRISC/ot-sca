#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Captures traces using the 'b' (batch absorb) command. 

Typical usage:
>>> ./simple_capture_traces_batch.py
"""

import argparse
import binascii
import random

from tqdm import tqdm
import chipwhisperer as cw
import numpy as np
import scared
import yaml

import simple_capture_traces as simple_capture
# from waverunner import WaveRunner
from cw_segmented import CwSegmented
from pyXKCP import pyxkcp


def next_lfsr():
    """ Generates pseudo-random bits to determine the sampling order.
    """
    global lfsr_state
    lfsr_out = bool(lfsr_state & 0x01)
    lfsr_state = lfsr_state >> 1
    if (lfsr_state & 0x01):
        lfsr_state ^= 0x8e
    return lfsr_out

def create_cw_segmented(ot, capture_cfg):
    return CwSegmented(num_samples=capture_cfg["num_samples"],
                           scope_gain=capture_cfg["scope_gain"], scope=ot.scope)


def run_batch_capture(capture_cfg, ot, ktp, scope):
    """Captures traces using the 'b' (batch encryption) command and WaveRunner.

    Args:
      capture_cfg: Dictionary with capture configuration settings.
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
      scope: Scope to use for capture.
    """
    global lfsr_state
    lfsr_state = 255
    fixed_key = bytearray([0x81, 0x1E, 0x37, 0x31, 0xB0, 0x12, 0x0A, 0x78, 
                           0x42, 0x78, 0x1E, 0x22, 0xB2, 0x5C, 0xDD, 0xF9])
    key = fixed_key
    ot.target.simpleserial_write(
        "s", capture_cfg["batch_prng_seed"].to_bytes(4, "little")
    )
    # Create the ChipWhisperer project.
    project_file = capture_cfg["project_name"]
    project = cw.create_project(project_file, overwrite=True)
    # Capture traces.
    rem_num_traces = capture_cfg["num_traces"]
    num_segments_storage = 1
    sample_fixed = True
    with tqdm(total=rem_num_traces, desc="Capturing", ncols=80, unit=" traces") as pbar:
        while rem_num_traces > 0:
            # Determine the number of traces for this batch and arm the oscilloscope.
            scope.num_segments = min(rem_num_traces, scope.num_segments_max)
            #scope.num_segments = min(rem_num_traces, 20)
            scope.arm()
            # Start batch encryption.
            ot.target.simpleserial_write(
                "b", scope.num_segments_actual.to_bytes(4, "little")
            )
            # Transfer traces
            waves = scope.capture_and_transfer_waves()
            assert waves.shape[0] == scope.num_segments
            plaintexts = []
            ciphertexts = []
            keys = []
            for i in range(scope.num_segments_actual):
                plaintext = ktp.next()[1]
                if sample_fixed:                 
                    key = fixed_key
                else:
                    random_key = ktp.next()[1]
                    key = random_key
                ciphertext = binascii.b2a_hex(pyxkcp.kmac128(key, ktp.key_len,
                                               np.asarray(plaintext), ktp.text_len,
                                               ot.target.output_len,
                                               b'\x00', 0))
                plaintexts.append(plaintext)
                ciphertexts.append(ciphertext)
                keys.append(key)
                sample_fixed = next_lfsr()
            
            
            # Check the last ciphertext of this batch to make sure we are in sync.
            actual_last_ciphertext = binascii.b2a_hex(ot.target.simpleserial_read("r", 32, ack=False))
            expected_last_ciphertext = ciphertexts[-1]
            assert actual_last_ciphertext == expected_last_ciphertext, (
                f"Incorrect encryption result!\n"
                f"actual: {actual_last_ciphertext}\n"
                f"expected: {expected_last_ciphertext}"
             )
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
            # Add traces of this batch to the project.
            for wave, plaintext, ciphertext, key in zip(waves, plaintexts, ciphertexts, keys):
                project.traces.append(
                    cw.common.traces.Trace(wave, plaintext, bytearray(ciphertext), key)
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


def main():
    with open("capture_sha3.yaml") as f:
        cfg = yaml.safe_load(f)
    ot = simple_capture.initialize_capture(cfg["device"], cfg["capture"])
    # Seed the PRNG.
    random.seed(cfg["capture"]["batch_prng_seed"])
    # Configure the key and plaintext generator.
    ktp = cw.ktp.Basic()
    ktp.key_len = cfg["capture"]["key_len_bytes"]
    ktp.text_len = cfg["capture"]["plain_text_len_bytes"]
    # Init scope
    scope = create_cw_segmented(ot, cfg["capture"])
    # Capture traces.
    run_batch_capture(cfg["capture"], ot, ktp, scope)
    # Plot a few traces.
    if cfg["plot_capture"]["show"]:
        project_name = cfg["capture"]["project_name"]
        simple_capture.plot_results(cfg["plot_capture"], project_name)


if __name__ == "__main__":
    main()
