from target.communication.fi_otp_commands import OTFIOtp
from target.chip import *
from target.dut import DUT
import time

def char_vendor_test(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otpfi = OTFIOtp(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otpfi.init()
    for _ in range(iterations):
        otpfi.otp_fi_vendor_test()
        response = target.read_response()
    return response

def char_owner_sw_cfg(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otpfi = OTFIOtp(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otpfi.init()
    for _ in range(iterations):
        otpfi.otp_fi_owner_sw_cfg()
        response = target.read_response()
    return response

def char_hw_cfg(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otpfi = OTFIOtp(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otpfi.init()
    for _ in range(iterations):
        otpfi.otp_fi_hw_cfg()
        response = target.read_response()
    return response

def char_life_cycle(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    otpfi = OTFIOtp(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = otpfi.init()
    for _ in range(iterations):
        otpfi.otp_fi_life_cycle()
        response = target.read_response()
    return response
