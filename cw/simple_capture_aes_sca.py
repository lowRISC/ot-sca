#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import binascii
# import random
import signal
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import chipwhisperer as cw
import numpy as np
import yaml
from Crypto.Cipher import AES
from cw_segmented import CwSegmented
from tqdm import tqdm
from waverunner import WaveRunner

from util import device, plot, trace_util


def abort_handler_during_loop(this_project, sig, frame):
    # Handler for ctrl-c keyboard interrupts
    # TODO: Has to be modified according to database (i.e. CW project atm) used
    if this_project is not None:
        print("\nHandling keyboard interrupt")
        this_project.close(save=True)
    sys.exit(0)


if __name__ == '__main__':
    # Load configuration from file
    with open('simple_capture_aes_sca.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Determine scope and single/batch from configuration ----------------------
    BATCH_MODE = False
    NUM_SEGMENTS = 1
    if "waverunner" in cfg and cfg["capture"]["scope_select"] == "waverunner":
        print("Using Waverunner scope")
        WAVERUNNER = True
        HUSKY = False

        # Determine single or batch mode
        if cfg["waverunner"]["num_segments"] > 1:
            BATCH_MODE = True
            NUM_SEGMENTS = cfg["waverunner"]["num_segments"]
    elif cfg["capture"]["scope_select"] == "husky":
        print("Using Husky scope")
        WAVERUNNER = False
        HUSKY = True

        # Batch not supported for Husky at the moment
        if cfg["cwfpgahusky"]["num_segments"] > 1:
            BATCH_MODE = True
            NUM_SEGMENTS = cfg["cwfpgahusky"]["num_segments"]
    else:
        print("Warning: No valid scope selected in configuration")

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

    # Support Husky batch mode
    if NUM_SEGMENTS > 1:
        husky_batch_scope = CwSegmented(num_samples=cfg["cwfpgahusky"]["num_samples"],
                                        offset=cfg["cwfpgahusky"]["offset"],
                                        scope_gain=cfg["cwfpgahusky"]["scope_gain"],
                                        scope=cwfpgahusky.scope,
                                        pll_frequency=cfg["cwfpgahusky"]["pll_frequency"])
        husky_batch_scope.num_segments = NUM_SEGMENTS

    # Upgrade Husky FW (manually uncomment if needed)
    # cwfpgahusky.scope.upgrade_firmware()
    # quit()

    if WAVERUNNER:
        # Create WaveRunner
        waverunner = WaveRunner(cfg["waverunner"]["waverunner_ip"])
        # Capture configuration: num_segments, sparsing, num_samples, first_point, acqu_channel
        waverunner.configure_waveform_transfer_general(cfg["waverunner"]["num_segments"],
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
        waverunner.save_setup_to_local_file(file_name_local)

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

    if BATCH_MODE:
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
            if HUSKY:
                if BATCH_MODE:
                    husky_batch_scope.arm()
                else:
                    cwfpgahusky.scope.arm()

            if WAVERUNNER:
                waverunner.arm()

            # Trigger execution(s) ---------------------------------------------
            if BATCH_MODE:
                # Perform batch encryptions
                cwfpgahusky.target.simpleserial_write("a", NUM_SEGMENTS.to_bytes(4, "little"))

            else:
                # Generate new text for next iteration, first uses initial text
                text = bytearray(cipher_gen.encrypt(text))
                # Load text and trigger execution
                cwfpgahusky.target.simpleserial_write('p', text)

            # Capture trace(s) -------------------------------------------------
            if HUSKY:
                if BATCH_MODE:
                    waves = husky_batch_scope.capture_and_transfer_waves()
                    assert waves.shape[0] == NUM_SEGMENTS
                else:
                    ret = cwfpgahusky.scope.capture(poll_done=False)
                    i = 0
                    while not cwfpgahusky.target.is_done():
                        i += 1
                        time.sleep(0.05)
                        if i > 100:
                            print("Warning: Target did not finish operation")
                    if ret:
                        print("Warning: Timeout happened during capture")
                    # Get Husky trace (single mode only)
                    wave = cwfpgahusky.scope.get_last_trace(as_int=True)

            if WAVERUNNER:
                waves = waverunner.capture_and_transfer_waves()
                assert waves.shape[0] == NUM_SEGMENTS
                # Put into uint8 range
                waves = waves + 128

            # Storing traces ---------------------------------------------------
            if BATCH_MODE:
                # Loop through num_segments to store traces and compute ciphertexts
                # Note this batch capture command uses the ciphertext as next text
                for i in range(NUM_SEGMENTS):
                    ciphertext = bytearray(cipher.encrypt(bytes(text)))

                    # Sanity check retrieved data (wave) and create CW Trace
                    assert len(waves[i, :]) >= 1
                    wave = waves[i, :]
                    trace = cw.Trace(wave, text, ciphertext, key_fixed)

                    # Append CW trace to CW project storage
                    # Also use uint16 as dtype so that tvla processing works
                    project.traces.append(trace, dtype=np.uint16)

                    # Use ciphertext as next text
                    text = ciphertext

            else:  # not BATCH_MODE
                ciphertext = bytearray(cipher.encrypt(bytes(text)))

                if WAVERUNNER:
                    # For single capture on WaveRunner, waves[0] contains data
                    wave = waves[0, :]

                # Sanity check retrieved data (wave) and create CW Trace
                assert len(wave) >= 1
                trace = cw.Trace(wave, text, ciphertext, key_fixed)
                if HUSKY:
                    # Check if ADC range has been exceeded for Husky.
                    # Not done for WaveRunner because clipping can be inspected on screen.
                    trace_util.check_range(trace.wave, cwfpgahusky.scope.adc.bits_per_sample)

                # Append CW trace to CW project storage
                # Also use uint16 for WaveRunner even though 8 bit so that tvla processing works
                project.traces.append(trace, dtype=np.uint16)

            # Get (last) ciphertext after all calls from device and verify -----
            response = cwfpgahusky.target.simpleserial_read('r',
                                                            cwfpgahusky.target.output_len,
                                                            ack=False)
            if binascii.b2a_hex(response) != binascii.b2a_hex(ciphertext):
                raise RuntimeError(f'Bad ciphertext: {response} != {ciphertext}.')

            # Update the loop variable and the progress bar --------------------
            remaining_num_traces -= NUM_SEGMENTS
            pbar.update(NUM_SEGMENTS)

    # Create and show test plot ------------------------------------------------
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
