from communication.fi_ibex_commands import OTFIIbex
from communication.chip import *
from communication.dut import DUT
import time

def single_beq_test(opentitantool):
    
    target = DUT()
    reset_target(opentitantool)
    target.print_all()


