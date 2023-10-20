#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import binascii
# import random
import signal
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import chipwhisperer as cw
import numpy as np
import yaml
from Crypto.Cipher import AES
from husky_dispatcher import HuskyDispatcher
from tqdm import tqdm
from waverunner import WaveRunner

from util import device, plot


def abort_handler_during_loop(this_project, sig, frame):
    # Handler for ctrl-c keyboard interrupts
    # FIXME: Has to be modified according to database (i.e. CW project atm) used
    if this_project is not None:
        print("\nHandling keyboard interrupt")
        this_project.close(save=True)
    sys.exit(0)


if __name__ == '__main__':
    # Load configuration from file
    with open('simple_capture_aes_sca.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Determine scope and single/batch from configuration ----------------------
    HUSKY = False
    WAVERUNNER = False

    if "waverunner" in cfg and cfg["capture"]["scope_select"] == "waverunner":
        WAVERUNNER = True
        NUM_SEGMENTS = cfg["waverunner"]["num_segments"]
        print(f"Using WaveRunner scope with {NUM_SEGMENTS} segments per trace")
    elif cfg["capture"]["scope_select"] == "husky":
        HUSKY = True
        NUM_SEGMENTS = cfg["cwfpgahusky"]["num_segments"]
        print(f"Using Husky scope with {NUM_SEGMENTS} segments per trace")
    else:
        print("Warning: No valid scope selected in configuration")
        sys.exit(0)

    # Create ChipWhisperer project for storage of traces and metadata ----------
    project = cw.create_project(cfg["capture"]["project_name"], overwrite=True)

    # Create device and scope --------------------------------------------------

    # Create OpenTitan encapsulating ChipWhisperer Husky and FPGA
    # NOTE: A clean separation of the two seems infeasible since
    # scope needs FPGA (PLL?) to be configured and target constructor needs scope as input.
    cwfpgahusky = device.OpenTitan(cfg["cwfpgahusky"]["fpga_bitstream"],
                                   cfg["cwfpgahusky"]["force_program_bitstream"],
                                   cfg["cwfpgahusky"]["fw_bin"],
                                   cfg["cwfpgahusky"]["pll_frequency"],
                                   cfg["cwfpgahusky"]["baudrate"],
                                   cfg["cwfpgahusky"]["scope_gain"],
                                   cfg["cwfpgahusky"]["num_samples"],
                                   cfg["cwfpgahusky"]["offset"],
                                   cfg["cwfpgahusky"]["output_len_bytes"])

    if HUSKY:
        # Use Husky as scope
        scope = HuskyDispatcher(cwfpgahusky, NUM_SEGMENTS)

        # Upgrade Husky FW (manually uncomment if needed)
        # scope.scope.upgrade_firmware()
        # quit()

    if WAVERUNNER:
        # Use WaveRunner as scope
        scope = WaveRunner(cfg["waverunner"]["waverunner_ip"])
        # Capture configuration: num_segments, sparsing, num_samples, first_point, acqu_channel
        scope.configure_waveform_transfer_general(cfg["waverunner"]["num_segments"],
                                                  1,
                                                  cfg["waverunner"]["num_samples"],
                                                  cfg["waverunner"]["sample_offset"],
                                                  "C1")
        # We assume manual setup of channels, gain etc. (e.g. run this script modify while running)
        # Save setup to timestamped file for reference
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
        project_name = cfg["capture"]["project_name"]
        file_name_local = f"{project_name}_data/scope_config_{now_str}.lss"
        scope.save_setup_to_local_file(file_name_local)

    # Preparation of Key and plaintext generation ------------------------------

    # This test currently uses one fixed key and generates random texts through
    # AES encryption using a generation key. The batch mode uses ciphertexts as
    # next text input here and in the OT device SW.

    # Load fixed key and initial text values from cfg
    key_fixed = bytearray(cfg["test"]["key_fixed"])
    print(f'Using key: {binascii.b2a_hex(bytes(key_fixed))}')
    text = bytearray(cfg["test"]["text_fixed"])

    # Cipher to compute expected responses
    cipher = AES.new(bytes(key_fixed), AES.MODE_ECB)

    # Prepare generation of new texts/keys by encryption using key_for_generation
    key_for_gen = bytearray(cfg["test"]["key_for_gen"])
    cipher_gen = AES.new(bytes(key_for_gen), AES.MODE_ECB)

    # Seed the target's PRNGs for initial key masking, and additionally turn off masking when '0'
    cwfpgahusky.target.simpleserial_write("l", cfg["test"]["lfsr_seed"].to_bytes(4, "little"))

    # Set key
    cwfpgahusky.target.simpleserial_write("k", key_fixed)

    if NUM_SEGMENTS > 1:
        # Set initial plaintext for batch mode
        cwfpgahusky.target.simpleserial_write("i", text)

    # Template to generate key at random based on test_random_seed (not used)
    # random.seed(cfg["test"]["test_random_seed"])
    # key = bytearray(cfg["test"]["key_len_bytes"])
    # for i in range(0, cfg["test"]["key_len_bytes"]):
    #    key[i] = random.randint(0, 255)

    # Main loop for measurements with progress bar -----------------------------

    # Register ctrl-c handler to store traces on abort
    signal.signal(signal.SIGINT, partial(abort_handler_during_loop, project))

    remaining_num_traces = cfg["capture"]["num_traces"]
    with tqdm(total=remaining_num_traces, desc="Capturing", ncols=80, unit=" traces") as pbar:
        while remaining_num_traces > 0:

            # Note: Capture performance tested Oct. 2023:
            # Husy with 1200 samples per trace: 50 it/s
            # WaveRunner with 1200 - 50000 samples per trace: ~30 it/s
            #   +10% by setting 'Performance' to 'Analysis' in 'Utilies->Preferences' in GUI
            # WaveRunner batchmode (6k samples, 100 segmets, 1 GHz): ~150 it/s

            # Arm scope --------------------------------------------------------
            scope.arm()

            # Trigger execution(s) ---------------------------------------------
            if NUM_SEGMENTS > 1:
                # Perform batch encryptions
                cwfpgahusky.target.simpleserial_write("a", NUM_SEGMENTS.to_bytes(4, "little"))

            else:  # single encryption
                # Load text and trigger execution
                # First iteration uses initial text, new texts are generated below
                cwfpgahusky.target.simpleserial_write('p', text)

            # Capture trace(s) -------------------------------------------------
            waves = scope.capture_and_transfer_waves()
            assert waves.shape[0] == NUM_SEGMENTS

            # Storing traces ---------------------------------------------------

            # Loop through num_segments to compute ciphertexts and store traces
            # Note this batch capture command uses the ciphertext as next text
            # while the first text is the initial one
            for i in range(NUM_SEGMENTS):
                ciphertext = bytearray(cipher.encrypt(bytes(text)))

                # Sanity check retrieved data (wave) and create CW Trace
                assert len(waves[i, :]) >= 1
                trace = cw.Trace(waves[i, :], text, ciphertext, key_fixed)

                # Append CW trace to CW project storage
                # FIXME Also use uint16 as dtype for 8 bit WaveRunner for tvla processing
                project.traces.append(trace, dtype=np.uint16)

                # Use ciphertext as next text
                text = ciphertext

            # Get (last) ciphertext from device and verify ---------------------
            response = cwfpgahusky.target.simpleserial_read('r',
                                                            cwfpgahusky.target.output_len,
                                                            ack=False)
            if binascii.b2a_hex(response) != binascii.b2a_hex(ciphertext):
                raise RuntimeError(f'Bad ciphertext: {response} != {ciphertext}.')

            # Update the loop variable and the progress bar --------------------
            remaining_num_traces -= NUM_SEGMENTS
            pbar.update(NUM_SEGMENTS)

    # Create and show test plot ------------------------------------------------
    # Use this plot to check for clipping and adjust gain appropriately
    if cfg["capture"]["show_plot"]:
        plot.save_plot_to_file(project.waves, None, cfg["capture"]["plot_traces"],
                               cfg["capture"]["trace_image_filename"], add_mean_stddev=True)
        print(f'Created plot with {cfg["capture"]["plot_traces"]} traces: '
              f'{Path(cfg["capture"]["trace_image_filename"]).resolve()}')

    # Save metadata and entire configuration cfg to project file ---------------
    project.settingsDict['datetime'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    project.settingsDict['cfg'] = cfg
    if HUSKY:
        sample_rate = int(round(cwfpgahusky.scope.clock.adc_freq, -6))
        project.settingsDict['sample_rate'] = sample_rate
    project.save()
