from fi_ibex_function_test import single_beq_test
from python.runfiles import Runfiles
from communication.chip import *
import os

def main():
    r = Runfiles.Create()
    opentitantool_path = r.Rlocation("lowrisc_opentitan/sw/host/opentitantool/opentitantool")

    firmware_target_name = os.environ.get("SELECTED_FIRMWARE_TARGET", "pen_test_fi_ibex_silicon_owner_gb_rom_ext")
    firmware_path = r.Rlocation(f"lowrisc_opentitan/sw/device/tests/penetrationtests/firmware/{firmware_target_name}.img")

    flash_target(opentitantool_path, firmware_path)

    single_beq_test(opentitantool_path)

if __name__ == "__main__":
  main()
