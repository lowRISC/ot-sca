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

    # Key and plaintext generation building blocks
    # Generate key at random based on test_random_seed. TODO: not used atm
    random.seed(cfg["test"]["test_random_seed"])
    key = bytearray(cfg["test"]["key_len_bytes"])
    for i in range(0, cfg["test"]["key_len_bytes"]):
        key[i] = random.randint(0, 255)
    # Load initial values for key and text from cfg
    key = bytearray(cfg["test"]["key"])
    print(f'Using key: {binascii.b2a_hex(bytes(key))}')
    text = bytearray(cfg["test"]["text"])
    # Generating new texts/keys by encrypting using key_for_generation
    key_for_gen = bytearray(cfg["test"]["key_for_gen"])
    cipher_gen = AES.new(bytes(key_for_gen), AES.MODE_ECB)

    # Cipher to generate expected responses
    cipher = AES.new(bytes(key), AES.MODE_ECB)

    # Create OpenTitan encapsulating ChipWhisperer Husky and FPGA
    # NOTE: Johann tried to split them up into classes,
    # BUT scope needs FPGA (PLL?) to be configured
    # and target constructor needs scope as input.
    # A clean separation seems infeasible.
    cwfpgahusky = device.OpenTitan(cfg["cwfpgahusky"]["fpga_bitstream"],
                                   cfg["cwfpgahusky"]["force_program_bitstream"],
                                   cfg["cwfpgahusky"]["fw_bin"],
                                   cfg["cwfpgahusky"]["pll_frequency"],
                                   cfg["cwfpgahusky"]["baudrate"],
                                   cfg["cwfpgahusky"]["scope_gain"],
                                   cfg["cwfpgahusky"]["num_samples"],
                                   cfg["cwfpgahusky"]["offset"],
                                   cfg["cwfpgahusky"]["output_len_bytes"])

    # Create ChipWhisperer project for storage of traces and metadata
    project = cw.create_project(cfg["capture"]["project_name"], overwrite=True)

    # Register ctrl-c handler to store traces on abort
    signal.signal(signal.SIGINT, partial(abort_handler_during_loop, project))

    # Set key
    cwfpgahusky.target.simpleserial_write("k", key)
    # TODO: Alternative line: cwfpgahusky.target.set_key(key, ack=False)

    # Main loop for measurements with progress bar
    for _ in tqdm(range(cfg["capture"]["num_traces"]), desc='Capturing', ncols=80):

        # Generate and load new text
        text = bytearray(cipher_gen.encrypt(text))
        cwfpgahusky.target.simpleserial_write('p', text)

        # TODO: Useful code line for batch capture
        # cwfpgahusky..simpleserial_write("s", capture_cfg["batch_prng_seed"].to_bytes(4, "little"))

        # Arm scope
        cwfpgahusky.scope.arm()

        # Capture trace
        ret = cwfpgahusky.scope.capture(poll_done=False)
        i = 0
        while not cwfpgahusky.target.is_done():
            i += 1
            time.sleep(0.05)
            if i > 100:
                print("Warning: Target did not finish operation")
        if ret:
            print("Warning: Timeout happened during capture")

        # Get response and verify
        response = cwfpgahusky.target.simpleserial_read('r',
                                                        cwfpgahusky.target.output_len, ack=False)
        if binascii.b2a_hex(response) != binascii.b2a_hex(cipher.encrypt(bytes(text))):
            raise RuntimeError(f'Bad ciphertext: {response} != {cipher.encrypt(bytes(text))}.')

        # Get trace
        wave = cwfpgahusky.scope.get_last_trace(as_int=True)
        if len(wave) >= 1:
            trace = cw.Trace(wave, text, response, key)
        else:
            raise RuntimeError('Capture failed.')

        # TODO: Useful code line for batch capture
        # waves = scope.capture_and_transfer_waves()

        # Check if ADC range has been exceeded
        trace_util.check_range(trace.wave, cwfpgahusky.scope.adc.bits_per_sample)

        # Append traces to storage
        project.traces.append(trace, dtype=np.uint16)

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
