#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Mix column HD Attack - Capture portion

Attack implemented by ChipWhisperer:
Repo: https://github.com/newaetech/chipwhisperer-jupyter/blob/master/experiments/MixColumn%20Attack.ipynb
Reference: https://eprint.iacr.org/2019/343.pdf

See mix_columns_cpa_attack.py for attack portion.
"""
import chipwhisperer as cw
import yaml

import capture_traces as cp

if __name__ == "__main__":
  with open('capture.yaml') as f:
    cfg_file = yaml.load(f, Loader=yaml.FullLoader)
  ot = cp.initialize_capture(cfg_file['device'], cfg_file['spiflash'])
  ot.target.output_len = cfg_file['capture']['plain_text_len_bytes']

  # Key and plaintext generator
  ktp = cw.ktp.VarVec()
  ktp.key_len = cfg_file['capture']['key_len_bytes']
  ktp.text_len = cfg_file['capture']['plain_text_len_bytes']

  project_name = cfg_file['capture']['project_name']

  # For each iteration, run a capture where only the bytes specified in
  # `text_range` are set to random values. All other bytes are set to a
  # fixed value.
  for var_vec in range(4):
    cfg_file['capture']['project_name'] = f'{project_name}_{var_vec}'
    ktp.var_vec = var_vec
    cp.run_capture(cfg_file['capture'], ot, ktp)

  ot.scope.dis()
  ot.target.dis()
