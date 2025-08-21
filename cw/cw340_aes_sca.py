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

def setup_target(scope):
    target = cw.target(scope)
    target.baud = int(115200)
    target.flush()
    version = None
    ping_cnt = 0
    while not version:
        if ping_cnt == 3:
            print("no response from target")
            sys.exit(0)
        target.write('v' + '\n')
        ping_cnt += 1
        time.sleep(0.5)
        version = target.read().strip()
    print(version)
    return target

def setup_scope(cfg):
    scope = cw.scope()
    scope.gain.db = cfg["cwfpgahusky"]["scope_gain"]
    scope.adc.basic_mode = "rising_edge"
    scope.clock.clkgen_src = 'extclk'
    scope.clock.clkgen_freq = cfg["cwfpgahusky"]["pll_frequency"]
    scope.clock.adc_mul = cfg["cwfpgahusky"]["adc_mul"]
    scope.clock.extclk_monitor_enabled = False
    scope.adc.samples = cfg["cwfpgahusky"]["num_samples"]
    if cfg["cwfpgahusky"]["offset"] >= 0:
        scope.adc.offset = cfg["cwfpgahusky"]["offset"]
    else:
        scope.adc.offset = 0
        scope.adc.presamples = -cfg["cwfpgahusky"]["offset"]
    scope.trigger.triggers = "tio4"
    scope.io.tio1 = "serial_tx"
    scope.io.tio2 = "serial_rx"
    scope.io.hs2 = "disabled"
    scope.clock.clkgen_src = 'extclk'
    if not scope.clock.clkgen_locked:
        print("scope.clock.clkgen is not locked")
        sys.exit(0)
    if not scope.clock.adc_locked:
        print("scope.clock.adc is not locked")
        sys.exit(0)
    return scope

def prepare_aes(cfg):
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
    return key, text, key_for_gen, cipher_gen


if __name__ == '__main__':
    # Load configuration from file
    with open('simple_capture_aes_sca_cw340.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    scope = setup_scope(cfg)
    target = setup_target(scope)

    # Create ChipWhisperer project for storage of traces and metadata
    project = cw.create_project(cfg["capture"]["project_name"], overwrite=True)

    key, text, key_for_gen, cipher_gen = prepare_aes(cfg)

    # Set key
    target.simpleserial_write("k", key)

    # Cipher to compute expected responses
    cipher = AES.new(bytes(key), AES.MODE_ECB)

    # Main loop for measurements with progress bar
    for _ in tqdm(range(cfg["capture"]["num_traces"]), desc='Capturing', ncols=80):
        scope.arm()

        # Generate new text for this iteration
        text = bytearray(cipher_gen.encrypt(text))
        # Load text and trigger execution
        target.simpleserial_write('p', text)

        # Capture trace
        ret = scope.capture(poll_done=False)
        i = 0
        while not target.is_done():
            i += 1
            time.sleep(0.05)
            if i > 100:
                print("Warning: Target did not finish operation")
        if ret:
            print("Warning: Timeout happened during capture")

        # Get trace
        wave = scope.get_last_trace(as_int=True)


        # Get response from device and verify
        response = target.simpleserial_read('r', target.output_len, ack=False)
        if binascii.b2a_hex(response) != binascii.b2a_hex(cipher.encrypt(bytes(text))):
            raise RuntimeError(f'Bad ciphertext: {response} != {cipher.encrypt(bytes(text))}.')

        # Sanity check retrieved data (wave) and create CW Trace
        if len(wave) >= 1:
            trace = cw.Trace(wave, text, response, key)
        else:
            raise RuntimeError('Capture failed.')

        # Check if ADC range has been exceeded for Husky.
        trace_util.check_range(trace.wave, scope.adc.bits_per_sample)

        # Append CW trace to CW project storage
        project.traces.append(trace, dtype=np.uint16)

    # Save metadata and entire configuration cfg to project file
    project.settingsDict['datetime'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    project.settingsDict['cfg'] = cfg
    sample_rate = int(round(scope.clock.adc_freq, -6))
    project.settingsDict['sample_rate'] = sample_rate
    project.save()

    # Create and show test plot
    if cfg["capture"]["show_plot"]:
        plot.save_plot_to_file(project.waves, None, cfg["capture"]["plot_traces"],
                               cfg["capture"]["trace_image_filename"], add_mean_stddev=True)
        print(f'Created plot with {cfg["capture"]["plot_traces"]} traces: '
              f'{Path(cfg["capture"]["trace_image_filename"]).resolve()}')
