from communication.sca_sha3_commands import OTSHA3
from communication.sca_prng_commands import OTPRNG
from communication.sca_trigger_commands import OTTRIGGER
from communication.chip import *
from communication.dut import DUT
import time
import random

def char_sha3_single_absorb(opentitantool, iterations, masking, text):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    sha3sca = OTSHA3(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = sha3sca.init(0)

    if masking:
        lfsr_seed = [0, 1, 2, 3]
        sha3sca.write_lfsr_seed(lfsr_seed)
    else:
        sha3sca.set_mask_off()
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)
    
    for _ in range(iterations):
        sha3sca.absorb(text, len(text))
        response = target.read_response()
    return response

def char_sha3_batch_absorb(opentitantool, iterations, num_segments, masking, text):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    sha3sca = OTSHA3(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = sha3sca.init(0)

    if masking:
        lfsr_seed = [0, 1, 2, 3]
        sha3sca.write_lfsr_seed(lfsr_seed)
    else:
        sha3sca.set_mask_off()
    
    # Set the internal prng
    ot_prng = OTPRNG(target=target, protocol="ujson")
    ot_prng.seed_prng([1,0,0,0])

    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)

    sha3sca.fvsr_fixed_msg_set(text, len(text))
    
    for _ in range(iterations):
        sha3sca.absorb_batch(num_segments)
        response = target.read_response()
    return response