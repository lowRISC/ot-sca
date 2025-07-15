from target.communication.fi_crypto_commands import OTFICrypto
from target.chip import *
from target.dut import DUT
import time

def char_aes(opentitantool, iterations, trigger):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    cryptofi = OTFICrypto(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = cryptofi.init()
    for _ in range(iterations):
        if trigger == 0:
            # Trigger over loading the key
            cryptofi.crypto_aes_key()
        if trigger == 1:
            # Trigger over loading the plaintext
            cryptofi.crypto_aes_plaintext()
        if trigger == 2:
            # Trigger over encryption
            cryptofi.crypto_aes_encrypt()
        if trigger == 3:
            # Trigger over reading the ciphertext
            cryptofi.crypto_aes_ciphertext()
        response = target.read_response()
    # Return the ciphertext that is read out
    return response

def char_kmac(opentitantool, iterations, trigger):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    cryptofi = OTFICrypto(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = cryptofi.init()
    for _ in range(iterations):
        if trigger == 0:
            # Trigger over loading the key
            cryptofi.crypto_kmac_key()
        if trigger == 1:
            # Trigger over loading the input
            cryptofi.crypto_kmac_absorb()
        if trigger == 2:
            # Trigger over the mac itself
            cryptofi.crypto_kmac_static()
        if trigger == 3:
            # Trigger over reading the output
            cryptofi.crypto_kmac_squeeze()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_kmac_state(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    cryptofi = OTFICrypto(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = cryptofi.init()
    for _ in range(iterations):
        cryptofi.crypto_kmac_state()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_sha256(opentitantool, iterations, msg, trigger):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    cryptofi = OTFICrypto(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = cryptofi.init()
    for _ in range(iterations):
        cryptofi.crypto_sha2(msg, trigger)
        response = target.read_response()
    # Return the result that is read out
    return response

def char_shadow_reg_access(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    cryptofi = OTFICrypto(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = cryptofi.init()
    for _ in range(iterations):
        cryptofi.crypto_shadow_reg_access()
        response = target.read_response()
    # Return the result that is read out
    return response

def char_shadow_reg_read(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    cryptofi = OTFICrypto(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = cryptofi.init()
    for _ in range(iterations):
        cryptofi.crypto_shadow_reg_read()
        response = target.read_response()
    # Return the result that is read out
    return response