#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Test WaveRunner class."""

from datetime import datetime

from waverunner import WaveRunner

if __name__ == "__main__":
    # Create WaveRunner
    waverunner = WaveRunner("172.26.111.125")

    # Save WaveRunner setup to timestamped file
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    file_name_local = f"scope_config_{now_str}.lss"
    waverunner.save_setup_to_local_file(file_name_local)

    # Write setup to local file for later restore (no time-stamps)
    file_name_local = "scope_config.lss"
    waverunner.save_setup_to_local_file(file_name_local)
    # Load setup from local file
    waverunner.load_setup_from_local_file(file_name_local)

    # Configuration: Choose num_samples and first_point
    # num_segments, sparsing, num_samples, first_point, acqu_channel
    waverunner.configure_waveform_transfer_general(5, 1, 1000, 0, "C1")

    # Loop
    waverunner.arm()
    waves = waverunner.capture_and_transfer_waves()

    # plot waves
    # TODO
