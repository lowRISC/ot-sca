#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import argparse
import binascii
from Crypto.Cipher import AES
import numpy as np
import time
from tqdm import tqdm
import yaml

import chipwhisperer as cw

from util import device
from util import plot


def initialize_capture(device_cfg, capture_cfg):
    """Initialize capture."""
    ot = device.OpenTitan(device_cfg['fpga_bitstream'],
                          device_cfg['fw_bin'],
                          device_cfg['pll_frequency'],
                          device_cfg['baudrate'],
                          capture_cfg['scope_gain'],
                          capture_cfg['num_samples'])
    print(f'Scope setup with sampling rate {ot.scope.clock.adc_rate} S/s')
    # Ping target
    print('Reading from FPGA using simpleserial protocol.')
    version = None
    ping_cnt = 0
    while not version:
        if ping_cnt == 3:
            raise RuntimeError(f'No response from the target (attempts: {ping_cnt}).')
        ot.target.write('v' + '\n')
        ping_cnt += 1
        time.sleep(0.5)
        version = ot.target.read().strip()
    print(f'Target simpleserial version: {version} (attempts: {ping_cnt}).')
    return ot


def run_capture(capture_cfg, ot, ktp):
    """Run ChipWhisperer capture.

    Args:
      capture_cfg: Dictionary with capture configuration settings.
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
    """
    key, text = ktp.next()

    cipher = AES.new(bytes(key), AES.MODE_ECB)
    print(f'Using key: {binascii.b2a_hex(bytes(key))}')

    project_file = capture_cfg['project_name']
    project = cw.create_project(project_file, overwrite=True)

    for i in tqdm(range(capture_cfg['num_traces']), desc='Capturing', ncols=80):
        key, text = ktp.next()
        ret = cw.capture_trace(ot.scope, ot.target, text, key, ack=False)
        if not ret:
            print('Failed capture')
            continue

        expected = binascii.b2a_hex(cipher.encrypt(bytes(text)))
        got = binascii.b2a_hex(ret.textout)
        assert (got == expected), (
            f'Incorrect encryption result!\ngot: {got}\nexpected: {expected}\n'
        )

        project.traces.append(ret)

        ot.target.flush()

    project.save()


def plot_results(plot_cfg, project_name):
    """Plots traces from `project_name` using `plot_cfg` settings."""
    project = cw.open_project(project_name)

    if len(project.waves) == 0:
        print('Project contains no traces. Did the capture fail?')
        return

    # The ADC output is in the interval [-0.5, 0.5). Check that the recorded
    # traces are within that range with some safety margin.
    if not (np.all(np.greater(project.waves, -plot_cfg['amplitude_max'])) and
            np.all(np.less(project.waves, plot_cfg['amplitude_max']))):
        print('WARNING: Some traces have samples outside the range (' +
              str(-plot_cfg['amplitude_max']) + ', ' +
              str(plot_cfg['amplitude_max']) + ').')
        print('         The ADC has a max range of [-0.5, 0.5) and might saturate.')
        print('         It is recommended to reduce the scope gain (see device.py).')

    plot.save_plot_to_file(project.waves, plot_cfg['num_traces'],
                           plot_cfg['trace_image_filename'])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-traces',
                        '-n',
                        type=int,
                        help="Override number of traces.")
    parser.add_argument('--plot-traces',
                        '-p',
                        type=int,
                        help="Plot number of traces.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open('capture_aes.yaml') as f:
        cfg_file = yaml.load(f, Loader=yaml.FullLoader)

    if args.num_traces:
        cfg_file['capture']['num_traces'] = args.num_traces

    if args.plot_traces:
        cfg_file['plot_capture']['show'] = True
        cfg_file['plot_capture']['num_traces'] = args.plot_traces

    ot = initialize_capture(cfg_file['device'], cfg_file['capture'])

    # Key and plaintext generator
    ktp = cw.ktp.Basic()
    ktp.key_len = cfg_file['capture']['key_len_bytes']
    ktp.text_len = cfg_file['capture']['plain_text_len_bytes']
    ot.target.output_len = cfg_file['capture']['plain_text_len_bytes']

    run_capture(cfg_file['capture'], ot, ktp)

    project_name = cfg_file['capture']['project_name']
    plot_results(cfg_file['plot_capture'], project_name)
