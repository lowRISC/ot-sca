#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import serial

from target.chip import Chip
from target.cw_fpga import CWFPGA


@dataclass
class TargetConfig:
    """ Target configuration.
    Stores information about the target.
    """
    target_type: str
    fw_bin: str
    protocol: str
    pll_frequency: int
    output_len: int
    bitstream: Optional[str] = None
    force_program_bitstream: Optional[bool] = False
    baudrate: Optional[int] = None
    port: Optional[str] = None
    read_timeout: Optional[int] = 1
    usb_serial: Optional[str] = None
    interface: Optional[str] = "hyper310"


class Target:
    """ Target class.

    Represents a SCA/FI target. Currently, ChipWhisperer FPGA boards
    or the discrete OpenTitan chip are supported.
    """
    def __init__(self, target_cfg: TargetConfig) -> None:
        self.target_cfg = target_cfg
        self.target = self._init_target()
        self.com_interface = self._init_communication()

    def _init_target(self):
        """ Init target.

        Configure OpenTitan on CW FPGA or the discrete chip.
        """
        target = None
        if self.target_cfg.target_type == "cw305" or self.target_cfg.target_type == "cw310":
            target = CWFPGA(
                bitstream = self.target_cfg.bitstream,
                force_programming = self.target_cfg.force_program_bitstream,
                firmware = self.target_cfg.fw_bin,
                pll_frequency = self.target_cfg.pll_frequency,
                baudrate = self.target_cfg.baudrate,
                output_len = self.target_cfg.output_len,
                protocol = self.target_cfg.protocol
            )
        elif self.target_cfg.target_type == "chip":
            target = Chip(firmware = self.target_cfg.fw_bin,
                          opentitantool_path = "../objs/opentitantool",
                          usb_serial = self.target_cfg.usb_serial,
                          interface = self.target_cfg.interface)
        else:
            raise RuntimeError("Error: Target not supported!")
        return target

    def _init_communication(self):
        """ Open the communication interface.

        Configure OpenTitan on CW FPGA or the discrete chip.
        """
        com_interface = None
        if self.target_cfg.protocol == "simpleserial":
            com_interface = self.target.target
        elif self.target_cfg.protocol == "ujson":
            if self.target_cfg.port is None or self.target_cfg.baudrate is None:
                raise RuntimeError("Error: Invalid port or baudrate provided!")
            com_interface = serial.Serial(self.target_cfg.port)
            com_interface.baudrate = self.target_cfg.baudrate
            com_interface.timeout = self.target_cfg.read_timeout
        else:
            raise RuntimeError("Error: Communication protocol not supported!")
        return com_interface

    def reset_target(self, com_reset: Optional[bool] = False):
        """Resets the target. """
        self.target.reset_target()
        if com_reset and self.target_cfg.protocol == "ujson":
            self.com_interface = self._init_communication()

    def write(self, data, cmd: Optional[str] = ""):
        """Write data to the target. """
        if self.target_cfg.protocol == "simpleserial":
            self.com_interface.simpleserial_write(cmd, data)
        else:
            self.com_interface.write(data)

    def readline(self):
        """read line. Only for uJSON. """
        if self.target_cfg.protocol == "ujson":
            return self.com_interface.readline()
        else:
            raise RuntimeError("Error: read_line only available for uJSON!")

    def read(self, cmd: str, len_bytes: int, ack: Optional[bool] = False):
        """Read. Only for simpleserial. """
        if self.target_cfg.protocol == "simpleserial":
            return self.com_interface.simpleserial_read(cmd, len_bytes, ack=ack)
        else:
            raise RuntimeError("Error: read only available for simpleserial!")

    def wait_ack(self, time: Optional[int] = None):
        """Wait_ack. Only for simpleserial. """
        if self.target_cfg.protocol == "simpleserial":
            if time is None:
                return self.com_interface.simpleserial_wait_ack()
            else:
                return self.com_interface.simpleserial_wait_ack(time)
        else:
            raise RuntimeError("Error: wait_ack only available for simpleserial!")

    def is_done(self):
        """Check if target is done. Only for CWFPGA."""
        if self.target_cfg.target_type == "cw305" or self.target_cfg.target_type == "cw310":
            return self.target.target.is_done()
        else:
            return True
