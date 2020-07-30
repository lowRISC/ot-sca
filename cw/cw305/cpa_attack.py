#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import binascii
import pickle

import tqdm

import chipwhisperer as cw
import chipwhisperer.analyzer as cwa


project_file = "projects/opentitan_simple_aes"
project = cw.open_project(project_file)

attack = cwa.cpa(project, cwa.leakage_models.last_round_state_diff)

update_interval = 25
progress_bar = tqdm.tqdm(total=len(project.traces), ncols=80)
progress_bar.set_description('Performing Attack')

def cb():
  progress_bar.update(update_interval)

attack_results = attack.run(callback=cb, update_interval=update_interval)
progress_bar.close()

key_last_round = [kguess[0][0] for kguess in attack_results.find_maximums()]
key_guess = cwa.attacks.models.aes.key_schedule.key_schedule_rounds(
    key_last_round, 10, 0)

known_key = binascii.b2a_hex(bytearray(project.keys[0]))
print('known_key: {}'.format(known_key))

key_guess = binascii.b2a_hex(bytearray(key_guess))
print('key guess: {}'.format(key_guess))

print(attack_results)

if key_guess != known_key:
  print('FAIL: key_guess != known_key')

print('Saving results')
pickle_file = project_file + ".results.pickle"
pickle.dump(attack_results, open(pickle_file, "wb"))
