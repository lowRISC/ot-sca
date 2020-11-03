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
from util import spiflash


def initialize_capture(device_cfg, spiflash_cfg):
  """Initialize capture."""
  fw_programmer = spiflash.FtdiProgrammer(
    spiflash_cfg['bin'],
    spiflash_cfg['dev_id'],
    spiflash_cfg['dev_sn'],
    device_cfg['fw_bin'])

  ot = device.OpenTitan(
    fw_programmer,
    device_cfg['fpga_bitstream'],
    device_cfg['pll_frequency'],
    device_cfg['baudrate'])
  print(f'Scope setup with sampling rate {ot.scope.clock.adc_rate} S/s')
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

  print('Reading from FPGA using simpleserial protocol.')
  ot.target.write('v'+'\n')
  time.sleep(0.5)
  print(f'Checking version: {ot.target.read().strip()}')

  project_file = capture_cfg['project_name']
  project = cw.create_project(project_file, overwrite=True)

  for i in tqdm(range(capture_cfg['num_traces']), desc='Capturing', ncols=80):
    key, text = ktp.next()
    ret = cw.capture_trace(ot.scope, ot.target, text, key)
    if not ret:
      print('Failed capture')
      continue
    # This value may need to be updated if the trace dB factor changes.
    elif min(ret.wave) < -0.45:
      continue

    expected = binascii.b2a_hex(cipher.encrypt(bytes(text)))
    got = binascii.b2a_hex(ret.textout)
    assert (got == expected), (
        f'Incorrect encryption result!\ngot: {got}\nexpected: {expected}\n')

    project.traces.append(ret)

    ot.target.flush()

  project.save()


def plot_results(plot_cfg, project_name):
  """Plots traces from `project_name` using `plot_cfg` settings."""
  project = cw.open_project(project_name)

  if len(project.waves) == 0:
    print('Project contains no traces. Did the capture fail?')
    return

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

  with open('capture.yaml') as f:
    cfg_file = yaml.load(f, Loader=yaml.FullLoader)

  if args.num_traces:
    cfg_file['capture']['num_traces'] = args.num_traces

  if args.plot_traces:
    cfg_file['plot_capture']['show'] = True
    cfg_file['plot_capture']['num_traces'] = args.plot_traces

  ot = initialize_capture(cfg_file['device'], cfg_file['spiflash'])

  # Key and plaintext generator
  ktp = cw.ktp.Basic()
  ktp.key_len = cfg_file['capture']['key_len_bytes']
  ktp.text_len = cfg_file['capture']['plain_text_len_bytes']
  ot.target.output_len = cfg_file['capture']['plain_text_len_bytes']

  run_capture(cfg_file['capture'], ot, ktp)

  project_name = cfg_file['capture']['project_name']
  plot_results(cfg_file['plot_capture'], project_name)
