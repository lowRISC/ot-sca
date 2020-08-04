# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Utilities used to download firmware images into the OpenTitan system."""

import subprocess

class FtdiProgrammer(object):
  """Spiflash executable wrapper.

  See doc/getting_started.md for details on how to connect the FTDI cable to
  the CW305 FPGA target board.
  """
  def __init__(self, spiflash_path, dev_id, dev_sn, input):
    """Initializes spiflash FTDI programmer.

    Args:
      spiflash_path: Path to spiflash executable.
      dev_id: FTDI USB device ID in vendor:product hex format.
      dev_sn: FTDI USB serial number as reported by lsusb.
      input: Path to OpenTitan firmware image.
    """
    self.cmd = [spiflash_path, f'--dev-id={dev_id}', f'--dev-sn={dev_sn}',
                f'--input={input}']

  def run(self):
    """Run spiflash utility with predefined command line arguments."""
    subprocess.check_call(self.cmd)
