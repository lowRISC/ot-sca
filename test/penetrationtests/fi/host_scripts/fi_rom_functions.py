from target.communication.fi_rom_commands import OTFIRom
from target.chip import *
from target.dut import DUT
import time

def char_rom_read(opentitantool, iterations):
    target = DUT()
    reset_target(opentitantool)
    # Clear the output from the reset
    target.dump_all()

    romfi = OTFIRom(target)
    # Initialize our chip and catch its output
    device_id, sensors, alerts, owner_page, boot_log, boot_measurements, version = romfi.init()
    for _ in range(iterations):
        romfi.handle_rom_read()
        response = target.read_response()
    return response