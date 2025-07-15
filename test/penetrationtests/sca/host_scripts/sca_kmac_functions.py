from communication.sca_kmac_commands import OTKMAC
from communication.sca_prng_commands import OTPRNG
from communication.sca_trigger_commands import OTTRIGGER
from communication.chip import *
from communication.dut import DUT
import time
import random

def char_kmac_single(opentitantool, iterations, masking, key, text):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    kmacsca = OTKMAC(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = kmacsca.init(0)

    if masking:
        lfsr_seed = [0, 1, 2, 3]
    else:
        lfsr_seed = [0, 0, 0, 0]
    kmacsca.write_lfsr_seed(lfsr_seed)

    kmacsca.write_key(key)
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)
    
    for _ in range(iterations):
        kmacsca.absorb(text, len(text))
        response = target.read_response()
    return response

def char_kmac_batch_daisy_chain(opentitantool, iterations, num_segments, masking, key, text):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    kmacsca = OTKMAC(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = kmacsca.init(0)

    if masking:
        lfsr_seed = [0, 1, 2, 3]
    else:
        lfsr_seed = [0, 0, 0, 0]
    kmacsca.write_lfsr_seed(lfsr_seed)
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)
    
    for _ in range(iterations):
        kmacsca.absorb_daisy_chain(text, key, num_segments)
        response = target.read_response()
    return response

def char_kmac_batch(opentitantool, iterations, num_segments, masking, key, text):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    kmacsca = OTKMAC(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = kmacsca.init(0)

    if masking:
        lfsr_seed = [0, 1, 2, 3]
    else:
        lfsr_seed = [0, 0, 0, 0]
    kmacsca.write_lfsr_seed(lfsr_seed)
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)

    # Set the internal prng
    ot_prng = OTPRNG(target=target, protocol="ujson")
    ot_prng.seed_prng([1,0,0,0])

    kmacsca.fvsr_key_set(key, len(key))
    
    for _ in range(iterations):
        kmacsca.absorb_batch(num_segments)
        response = target.read_response()
    return response