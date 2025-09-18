#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import serial
from serial.tools.list_ports import comports

from target.chip import Chip
from target.cw_fpga import CWFPGA


@dataclass
class TargetConfig:
    """Target configuration.
    Stores information about the target.
    """

    target_type: str
    fw_bin: str
    pll_frequency: Optional[int] = 100000000
    bitstream: Optional[str] = None
    force_program_bitstream: Optional[bool] = False
    baudrate: Optional[int] = None
    port: Optional[str] = None
    read_timeout: Optional[int] = 1
    interface: Optional[str] = "hyper310"
    usb_serial: Optional[str] = None
    husky_serial: Optional[str] = None
    opentitantool: Optional[str] = None


class Target:
    """Target class.

    Represents a SCA/FI target. Currently, ChipWhisperer FPGA boards
    or the discrete OpenTitan chip are supported.
    """

    # This is a fixed baudrate.
    baudrate = 115200
    # Due to a bug in the UART of the CW340, we need to send each byte separately
    # and add a small timeout before sending the next one.
    # This contains the calculation of the delay.
    pacing = 10 / baudrate

    def __init__(self, target_cfg: TargetConfig) -> None:
        self.target_cfg = target_cfg

        if target_cfg.baudrate is None:
            target_cfg.baudrate = 115200

        self.com_interface = self._init_communication(self.find_target_port(),
                                                      self.target_cfg.baudrate)

        if target_cfg.fw_bin is not None:
            self.target = self._init_target()

    def _init_target(self):
        """Init target.

        Configure OpenTitan on CW FPGA or the discrete chip.
        """
        target = None
        if (self.target_cfg.target_type == "cw305" or
                self.target_cfg.target_type == "cw310"):
            target = CWFPGA(
                bitstream=self.target_cfg.bitstream,
                force_programming=self.target_cfg.force_program_bitstream,
                firmware=self.target_cfg.fw_bin,
                pll_frequency=self.target_cfg.pll_frequency,
                baudrate=self.target_cfg.baudrate,
                usb_serial=self.target_cfg.usb_serial,
                husky_serial=self.target_cfg.husky_serial,
            )
        elif self.target_cfg.target_type == "chip":
            target = Chip(opentitantool_path=self.target_cfg.opentitantool)
            target.flash_target(self.target_cfg.fw_bin)
        else:
            raise RuntimeError("Error: Target not supported!")
        # Flush the output
        self.dump_all()
        return target

    def _init_communication(self, port, baudrate):
        """Open the communication interface.

        Configure OpenTitan on CW FPGA or the discrete chip.
        """
        com_interface = None
        com_interface = serial.Serial(port)
        com_interface.baudrate = baudrate
        com_interface.timeout = 1
        return com_interface

    def find_target_port(self):
        # First go to the manual set port.
        if self.target_cfg.port is not None:
            return self.target_cfg.port
        # Depending on the target find the port automatically.
        if self.target_cfg.target_type == "cw310":
            return self.find_target_port_cw310()
        if self.target_cfg.target_type == "chip":
            return self.find_target_port_hyperdebug()

        # Here we failed to find a known target_type.
        print("Unknown target_type!")
        return None

    def find_target_port_hyperdebug(self):
        for port in comports():
            if "UART2" in port.description and "HyperDebug" in port.description:
                return port.device
        print("Target not found!")
        return None

    def find_target_port_cw310(self):
        for port in comports():
            # TODO: needs to be tested and specified to take the correct UART
            if "c310" in port.product and "11070" in port.vid:
                return port.device
        print("Target not found!")
        return None

    def reset_target(self, com_reset: Optional[bool] = False):
        """Resets the target."""
        self.target.reset_target()
        if com_reset:
            self.com_interface = self._init_communication(
                self.find_target_port(), self.target_cfg.baudrate)

    def write(self, data, cmd: Optional[str] = ""):
        """Write data to the target."""
        self.com_interface.write(data)

    def readline(self):
        """read line. Only for uJSON."""
        return self.com_interface.readline()

    def is_done(self):
        """Check if target is done. Only for CWFPGA."""
        if (self.target_cfg.target_type == "cw305" or
                self.target_cfg.target_type == "cw310"):
            return self.target.target.is_done()
        else:
            return True

    def print_all(self, max_tries=50):
        it = 0
        while it != max_tries:
            read_line = str(self.readline().decode().strip())
            if len(read_line) > 0:
                print(read_line, flush=True)
            else:
                break
            it += 1

    def dump_all(self, max_tries=50):
        it = 0
        while it != max_tries:
            read_line = str(self.readline())
            if len(read_line) <= 5:
                break
            it += 1

    def check_fault_or_read_reponse(self, max_tries=50):
        """
        Args:
            max_tries: Maximum number of attempts to read from UART.

        Returns:
            - The JSON response of OpenTitan or the line containing FAULT.
            - True if the chip gave a response, False if it ran into a fault.
        """
        it = 0
        while it != max_tries:
            try:
                read_line = str(self.readline())
                if "FAULT" in read_line:
                    return read_line, False
                if "RESP_OK" in read_line:
                    return read_line.split("RESP_OK:")[1].split(
                        " CRC:")[0], True
                it += 1
            except UnicodeDecodeError:
                it += 1
                continue
        return "", False

    def check_reset_or_read_reponse(self, max_tries=50):
        """
        Args:
            max_tries: Maximum number of attempts to read from UART.

        Returns:
            - The JSON response of OpenTitan or the line containing Chip flashed.
            - True if the chip gave a response, False if the chip resetted.
        """
        it = 0
        while it != max_tries:
            try:
                read_line = str(self.readline())
                if "Chip flashed" in read_line:
                    return read_line, False
                if "RESP_OK" in read_line:
                    return read_line.split("RESP_OK:")[1].split(
                        " CRC:")[0], True
                it += 1
            except UnicodeDecodeError:
                it += 1
                continue
        return "", False

    def read_response(self, max_tries: Optional[int] = 50):
        """
        Args:
            max_tries: Maximum number of attempts to read from UART.

        Returns:
            The JSON response of OpenTitan.
        """
        it = 0
        while it < max_tries:
            try:
                read_line = str(self.readline().decode().strip())
            except UnicodeDecodeError:
                break
            if len(read_line) > 0:
                if "RESP_OK" in read_line:
                    return read_line.split("RESP_OK:")[1].split(" CRC:")[0]
            else:
                break
            it += 1
        return ""
