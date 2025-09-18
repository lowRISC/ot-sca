#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import binascii

import chipwhisperer as cw
from chipwhisperer.analyzer.attacks.attack_mix_columns import AttackMixColumns

PROJECTS = [
    "projects/opentitan_simple_aes_0",
    "projects/opentitan_simple_aes_1",
    "projects/opentitan_simple_aes_2",
    "projects/opentitan_simple_aes_3",
]

print("loading projects")
projects = [cw.open_project(p) for p in PROJECTS]

attack = AttackMixColumns(projects)
results = attack.run()

known_key_bytes = projects[0].keys[0]
key_guess_bytes = results["guess"]

known_key = binascii.b2a_hex(bytearray(known_key_bytes))
print("known_key: {}".format(known_key))

key_guess = binascii.b2a_hex(bytearray(key_guess_bytes))
print("key guess: {}".format(key_guess))

if key_guess != known_key:
    num_bytes_match = 0
    for i in range(len(known_key_bytes)):
        if known_key_bytes[i] == key_guess_bytes[i]:
            num_bytes_match += 1
    print("FAILED: key_guess != known_key")
    print("        " + str(num_bytes_match) + "/" + str(len(known_key_bytes)) +
          " bytes guessed correctly.")
else:
    print("SUCCESS!")
