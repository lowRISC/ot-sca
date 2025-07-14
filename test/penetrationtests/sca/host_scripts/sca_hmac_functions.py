from communication.sca_hmac_commands import OTHMAC
from communication.sca_prng_commands import OTPRNG
from communication.sca_trigger_commands import OTTRIGGER
from communication.chip import *
from communication.dut import DUT
import time

def char_hmac_single(opentitantool, iterations, trigger, key, text):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    hmacsca = OTHMAC(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = hmacsca.init()
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)
    
    for _ in range(iterations):
        hmacsca.single(text, key, trigger)
        response = target.read_response()
    return response

def char_hmac_daisy_chain(opentitantool, iterations, num_segments, trigger, key, text):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    hmacsca = OTHMAC(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = hmacsca.init()
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)
    
    for _ in range(iterations):
        hmacsca.daisy_chain(text, key, num_segments, trigger)
        response = target.read_response()
    return response

def char_hmac_random_batch(opentitantool, iterations, num_segments, trigger):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    hmacsca = OTHMAC(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = hmacsca.init()
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)

    # Set the internal prng
    ot_prng = OTPRNG(target=target, protocol="ujson")
    ot_prng.seed_prng([0,0,0,0])
    
    for _ in range(iterations):
        hmacsca.random_batch(num_segments, trigger)
        response = target.read_response()
    return response

def char_hmac_fvsr_batch(opentitantool, iterations, num_segments, trigger, key):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    hmacsca = OTHMAC(target, "ujson")
    # Initialize our chip and catch its output
    device_id, owner_page, boot_log, boot_measurements, version = hmacsca.init()
    
    # Set the trigger
    triggersca = OTTRIGGER(target, "ujson")
    triggersca.select_trigger(0)

    # Set the internal prng
    ot_prng = OTPRNG(target=target, protocol="ujson")
    ot_prng.seed_prng([0,0,0,0])
    
    for _ in range(iterations):
        hmacsca.fvsr_batch(key, num_segments, trigger)
        response = target.read_response()
    return response