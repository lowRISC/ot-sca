from communication.sca_aes_commands import OTAES
from communication.sca_prng_commands import OTPRNG
from communication.sca_trigger_commands import OTTRIGGER
from communication.chip import *
from communication.dut import DUT
import time

def char_aes_single_encrypt(opentitantool, iterations, masking, key, text):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    aessca = OTAES(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = aessca.init(0)

    # Check whether we enable the masking
    if masking:
        lfsr_seed = 1
    else:
        lfsr_seed = 0
    aessca.seed_lfsr(lfsr_seed.to_bytes(4, "little"))
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)

    aessca.key_set(key)
    
    for _ in range(iterations):
        aessca.single_encrypt(text)
        response = target.read_response()
    return response

def char_aes_batch_alternative_encrypt(opentitantool, iterations, num_segments, masking, key, text):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    aessca = OTAES(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = aessca.init(0)

    # Check whether we enable the masking
    if masking:
        lfsr_seed = 1
    else:
        lfsr_seed = 0
    aessca.seed_lfsr(lfsr_seed.to_bytes(4, "little"))
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)

    aessca.key_set(key, len(key))
    aessca.batch_plaintext_set(text, len(text))
    
    for _ in range(iterations):
        aessca.batch_alternative_encrypt(num_segments)
        response = target.read_response()
    return response

def char_aes_batch_data_fvsr_encrypt(opentitantool, iterations, num_segments, masking):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    aessca = OTAES(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = aessca.init(0)

    # Check whether we enable the masking
    if masking:
        lfsr_seed = 1
    else:
        lfsr_seed = 0
    aessca.seed_lfsr(lfsr_seed.to_bytes(4, "little"))
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)

    # Set the internal prng
    ot_prng = OTPRNG(target=target, protocol="ujson")
    ot_prng.seed_prng([0,0,0,0])

    # Generate plaintexts and keys for first batch.
    aessca.start_fvsr_batch_generate(2)
    
    for _ in range(iterations):
        aessca.fvsr_data_batch_encrypt(num_segments)
        response = target.read_response()
    return response

def char_aes_batch_key_fvsr_encrypt(opentitantool, iterations, num_segments, masking):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    aessca = OTAES(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = aessca.init(0)

    # Check whether we enable the masking
    if masking:
        lfsr_seed = 1
    else:
        lfsr_seed = 0
    aessca.seed_lfsr(lfsr_seed.to_bytes(4, "little"))
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)

    # Set the internal prng
    ot_prng = OTPRNG(target=target, protocol="ujson")
    ot_prng.seed_prng([0,0,0,0])

    # Generate plaintexts and keys for first batch.
    aessca.start_fvsr_batch_generate(1)
    aessca.write_fvsr_batch_generate(num_segments)
    
    for _ in range(iterations):
        aessca.fvsr_key_batch_encrypt(num_segments)
        response = target.read_response()
    return response

