#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Captures traces using the 'b' (batch encryption) command. Supports both
ChipWhisperer-Lite and WaveRunner 9104.

Typical usage:
>>> ./simple_capture_traces_batch.py -s SCOPE

SCOPE must be either "cw_lite" or "waverunner".
"""

import argparse
import random

from tqdm import tqdm
import chipwhisperer as cw
import numpy as np
import scared
import yaml

import simple_capture_traces as simple_capture
from waverunner import WaveRunner
from cw_lite_segmented import CwLiteSegmented


def create_waverunner(ot, capture_cfg):
    return WaveRunner(capture_cfg["waverunner_ip"])


def create_cw_lite_segmented(ot, capture_cfg):
    # TODO: Remove this disconnect after removing cw-lite init from device.py.
    ot.scope.dis()
    return CwLiteSegmented(num_samples=capture_cfg["num_samples"],
                           scope_gain=capture_cfg["scope_gain"])


SCOPE_FACTORY = {
    "cw_lite": create_cw_lite_segmented,
    "waverunner": create_waverunner,
}


def run_batch_capture(capture_cfg, ot, ktp, scope):
    """Captures traces using the 'b' (batch encryption) command and WaveRunner.

    Args:
      capture_cfg: Dictionary with capture configuration settings.
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
      scope: Scope to use for capture.
    """
    # Set key
    assert ktp.fixed_key
    key = ktp.next_key()
    print(f"Using key: '{key.hex()}'.")
    ot.target.simpleserial_write("k", key)
    # Seed the target's PRNG (host's PRNG is currently seeded in main).
    ot.target.simpleserial_write(
        "s", capture_cfg["batch_prng_seed"].to_bytes(4, "little")
    )
    # Create the ChipWhisperer project.
    project_file = capture_cfg["project_name"]
    project = cw.create_project(project_file, overwrite=True)
    # Capture traces.
    rem_num_traces = capture_cfg["num_traces"]
    num_segments_storage = 1
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
            # Generate plaintexts and ciphertexts for the batch.
            # Note: Plaintexts are encrypted in parallel.
            plaintexts = [ktp.next()[1] for _ in range(scope.num_segments_actual)]
            ciphertexts = [
                bytearray(c)
                for c in scared.aes.base.encrypt(
                    np.asarray(plaintexts), np.asarray(key)
                )
            ]
            # Check the last ciphertext of this batch to make sure we are in sync.
            actual_last_ciphertext = ot.target.simpleserial_read("r", 16, ack=False)
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
            for wave, plaintext, ciphertext in zip(waves, plaintexts, ciphertexts):
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


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""Captures traces using the 'b' (batch encryption) command.
        Supports both ChipWhisperer-Lite and WaveRunner 9104."""
    )
    parser.add_argument(
        "-s",
        "--scope",
        choices=["cw_lite", "waverunner"],
        required=True,
        help="scope to use",
    )
    return parser.parse_args()


def main():
    """Loads the configuration file, parses command-line arguments, captures and plots traces."""
    args = parse_args()
    with open("capture_aes.yaml") as f:
        cfg = yaml.safe_load(f)
    ot = simple_capture.initialize_capture(cfg["device"], cfg["capture"])
    # Seed the PRNG.
    # TODO: Replace this with a dedicated PRNG to avoid other packages breaking
    # our code.
    random.seed(cfg["capture"]["batch_prng_seed"])
    # Configure the key and plaintext generator.
    ktp = cw.ktp.Basic()
    ktp.key_len = cfg["capture"]["key_len_bytes"]
    ktp.text_len = cfg["capture"]["plain_text_len_bytes"]
    ot.target.output_len = cfg["capture"]["plain_text_len_bytes"]
    # Init scope
    # TODO: Define a proper interface and cleanup this part.
    scope = SCOPE_FACTORY[args.scope](ot, cfg["capture"])
    # Capture traces.
    run_batch_capture(cfg["capture"], ot, ktp, scope)
    # Plot a few traces.
    project_name = cfg["capture"]["project_name"]
    simple_capture.plot_results(cfg["plot_capture"], project_name)


if __name__ == "__main__":
    main()
