#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import binascii
import chipwhisperer as cw
from chipwhisperer.analyzer.attacks.attack_mix_columns import AttackMixColumns
import scared
import numpy as np

PROJECTS=[
  'projects/opentitan_simple_aes_0',
  'projects/opentitan_simple_aes_1',
  'projects/opentitan_simple_aes_2',
  'projects/opentitan_simple_aes_3',
]

print('loading projects')
projects = [cw.open_project(p) for p in PROJECTS]

attack = AttackMixColumns(projects)
results = attack.run()

known_key = binascii.b2a_hex(bytearray(projects[0].keys[0]))
print('known_key: {}'.format(known_key))

key_guess = binascii.b2a_hex(bytearray(results['guess']))
print('key guess: {}'.format(key_guess))

if key_guess != known_key:
  print('FAIL: key_guess != known_key')
else:
  print('ATTACK SUCCEED!')
