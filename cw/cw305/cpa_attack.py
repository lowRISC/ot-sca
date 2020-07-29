import binascii
import pickle

import tqdm

import chipwhisperer as cw
import chipwhisperer.analyzer as cwa


project_file = "projects/opentitan_simple_aes"
project = cw.open_project(project_file)

attack = cwa.cpa(project, cwa.leakage_models.last_round_state)

update_interval = 25
progress_bar = tqdm.tqdm(total=len(project.traces), ncols=80)
progress_bar.set_description('Performing Attack')

def cb():
  progress_bar.update(update_interval)

attack_results = attack.run(callback=cb, update_interval=update_interval)
progress_bar.close()

known_key = binascii.b2a_hex(bytearray(project.keys[0]))
print('known_key: {}'.format(known_key))

key_guess = binascii.b2a_hex(bytearray(attack_results.key_guess()))
print('key guess: {}'.format(key_guess))

print(attack_results)

if key_guess != known_key:
  print('FAIL: key_guess != known_key')

print('Saving results')
pickle_file = project_file + ".results.pickle"
pickle.dump(attack_results, open(pickle_file, "wb"))
