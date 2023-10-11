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
from functools import partial
from pathlib import Path

import chipwhisperer as cw
import numpy as np
import yaml
from Crypto.Cipher import AES
from tqdm import tqdm
from waverunner import WaveRunner

from util import device, plot, trace_util


def abort_handler_during_loop(project, sig, frame):
    # Handler for ctrl-c keyboard interrupts
    # TODO: Has to be modified according to database (i.e. CW project atm) used
    if project is not None:
        print("\nHandling keyboard interrupt")
        project.close(save=True)
    sys.exit(0)


if __name__ == '__main__':
    # Load configuration from file
    with open('simple_capture_aes_sca.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Choose scope from configuration
    if cfg["capture"]["scope_select"] == "waverunner":
        print("Using Waverunner scope")
        USE_WAVERUNNER = True
        USE_HUSKY = False
    elif cfg["capture"]["scope_select"] == "husky":
        print("Using Husky scope")
        USE_WAVERUNNER = False
        USE_HUSKY = True

    # Create ChipWhisperer project for storage of traces and metadata
    project = cw.create_project(cfg["capture"]["project_name"], overwrite=True)

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

    if USE_WAVERUNNER:
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

    # Check if batch mode is requested TODO: not supported yet
    if cfg["waverunner"]["num_segments"] > 1:
        print("Warning: Sequence (batch) mode not supported yet")
    if cfg["cwfpgahusky"]["num_segments"] > 1:
        print("Warning: Sequence (batch) mode not supported yet")

    # Register ctrl-c handler to store traces on abort
    signal.signal(signal.SIGINT, partial(abort_handler_during_loop, project))

    # Preparation of Key and plaintext generation
    # Generate key at random based on test_random_seed (not used atm)
    random.seed(cfg["test"]["test_random_seed"])
    key = bytearray(cfg["test"]["key_len_bytes"])
    for i in range(0, cfg["test"]["key_len_bytes"]):
        key[i] = random.randint(0, 255)
    # Load initial key and text values from cfg
    key = bytearray(cfg["test"]["key"])
    print(f'Using key: {binascii.b2a_hex(bytes(key))}')
    text = bytearray(cfg["test"]["text"])
    # Prepare generation of new texts/keys by encryption using key_for_generation
    key_for_gen = bytearray(cfg["test"]["key_for_gen"])
    cipher_gen = AES.new(bytes(key_for_gen), AES.MODE_ECB)

    # Set key
    cwfpgahusky.target.simpleserial_write("k", key)

    # Cipher to compute expected responses
    cipher = AES.new(bytes(key), AES.MODE_ECB)

    # Main loop for measurements with progress bar
    for _ in tqdm(range(cfg["capture"]["num_traces"]), desc='Capturing', ncols=80):

        # TODO: Useful code line for batch capture
        # cwfpgahusky..simpleserial_write("s", capture_cfg["batch_prng_seed"].to_bytes(4, "little"))

        # Note: Capture performance tested Oct. 2023:
        #   Using husky with 1200 samples per trace leads to 48 it/s
        #   Using Waverunner with 1200 - 50000 samples per trace leads to 27 it/s
        #       Increases to 31 it/s when 'Performance' set to 'Analysis' in Utilies->Preferences
        #       Transfer over UART only slows down if .e.g transfering key 5 additional times

        if USE_HUSKY:
            # Arm Husky scope
            cwfpgahusky.scope.arm()

        if USE_WAVERUNNER:
            # Arm Waverunner scope
            waverunner.arm()

        # Generate new text for this iteration
        text = bytearray(cipher_gen.encrypt(text))
        # Load text and trigger execution
        cwfpgahusky.target.simpleserial_write('p', text)

        if USE_HUSKY:
            # Capture Husky trace
            ret = cwfpgahusky.scope.capture(poll_done=False)
            i = 0
            while not cwfpgahusky.target.is_done():
                i += 1
                time.sleep(0.05)
                if i > 100:
                    print("Warning: Target did not finish operation")
            if ret:
                print("Warning: Timeout happened during capture")

            # Get Husky trace
            wave = cwfpgahusky.scope.get_last_trace(as_int=True)

        if USE_WAVERUNNER:
            # Capture and get Waverunner trace
            waves = waverunner.capture_and_transfer_waves()
            assert waves.shape[0] == cfg["waverunner"]["num_segments"]
            # For single capture, 1st dim contains wave data
            wave = waves[0, :]
            # Put into uint8 range
            wave = wave + 128

        # Get response from device and verify
        response = cwfpgahusky.target.simpleserial_read('r',
                                                        cwfpgahusky.target.output_len, ack=False)
        if binascii.b2a_hex(response) != binascii.b2a_hex(cipher.encrypt(bytes(text))):
            raise RuntimeError(f'Bad ciphertext: {response} != {cipher.encrypt(bytes(text))}.')

        # TODO: Useful code line for batch capture
        # waves = scope.capture_and_transfer_waves()

        # Sanity check retrieved data (wave) and create CW Trace
        if len(wave) >= 1:
            trace = cw.Trace(wave, text, response, key)
        else:
            raise RuntimeError('Capture failed.')

        if USE_HUSKY:
            # Check if ADC range has been exceeded for Husky.
            # Not done for WaveRunner because clipping can be inspected on screen.
            trace_util.check_range(trace.wave, cwfpgahusky.scope.adc.bits_per_sample)

        # Append CW trace to CW project storage
        if USE_HUSKY:
            project.traces.append(trace, dtype=np.uint16)
        if USE_WAVERUNNER:
            project.traces.append(trace, dtype=np.uint8)

    # Save metadata and entire configuration cfg to project file
    project.settingsDict['datetime'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    project.settingsDict['cfg'] = cfg
    sample_rate = int(round(cwfpgahusky.scope.clock.adc_freq, -6))
    project.settingsDict['sample_rate'] = sample_rate
    project.save()

    # Create and show test plot
    if cfg["capture"]["show_plot"]:
        plot.save_plot_to_file(project.waves, None, cfg["capture"]["plot_traces"],
                               cfg["capture"]["trace_image_filename"])
        print(f'Created plot with {cfg["capture"]["plot_traces"]} traces: '
              f'{Path(cfg["capture"]["trace_image_filename"]).resolve()}')
