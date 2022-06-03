#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Class for programming OpenTitan over SPI using the SAM3X/U on CW310/305."""

import time
from functools import partial
from collections import namedtuple
from chipwhisperer.capture.targets import CW310
from chipwhisperer.capture.targets import CW305
from contextlib import contextmanager


class SpiProgrammer:
    """Programs OpenTitan over SPI using the SAM3X/U on CW310/305.

    Initializes pins, resets OpenTitan, assert strap pins, and programs the flash.
    """
    # Pin mappings for CW305/CW310 boards.
    PinMapping = namedtuple('PinMapping', [
        'sck',
        'sdi',
        'sdo',
        'cs',
        'trst',
        'srst',
        'sw_strap0',
        'sw_strap1',
        'sw_strap2',
        'tap_strap0',
        'tap_strap1',
    ])
    PIN_MAPPINGS = {}
    PIN_MAPPINGS[id(CW305)] = PinMapping(
        sck='USB_A9',
        sdi='USB_A10',
        sdo='USB_A11',
        cs='USB_A12',
        trst='USB_A13',
        srst='USB_A14',
        sw_strap0='USB_A15',
        sw_strap1='USB_A16',
        sw_strap2='USB_A17',
        tap_strap0='USB_A18',
        tap_strap1='USB_A19',
    )
    PIN_MAPPINGS[id(CW310)] = PinMapping(
        sck='USB_SPI_SCK',
        sdi='USB_SPI_COPI',
        sdo='USB_SPI_CIPO',
        cs='USB_SPI_CS',
        trst='USB_A13',
        srst='USB_A14',
        sw_strap0='USB_A15',
        sw_strap1='USB_A16',
        sw_strap2='USB_A17',
        tap_strap0='USB_A18',
        tap_strap1='USB_A19',
    )
    INITIAL_VALUES = PinMapping(sck=0,
                                sdi=0,
                                sdo=0,
                                cs=0,
                                trst=1,
                                srst=1,
                                sw_strap0=0,
                                sw_strap1=0,
                                sw_strap2=0,
                                tap_strap0=0,
                                tap_strap1=0)

    RESET_DELAY = 0.1
    BUSY_POLL_DELAY = 0.01
    PAYLOAD_SIZE = 256
    MAX_BUSY_ITER_CNT = 500

    def __init__(self, fpga):
        """Inits a SpiProgrammer with a CW310/305.

        Args:
          fpga: CW310/305 to be programmed, ``chipwhisperer.capture.targets.CW310/305``.
        """
        self.pins = self.PIN_MAPPINGS[id(type(fpga))]
        fpga.con()
        self.io = fpga.gpio_mode()
        # Set strap pins as outputs
        for pin, mapped_to in self.pins._asdict().items():
            self.io.pin_set_output(mapped_to)
            self.io.pin_set_state(mapped_to, getattr(self.INITIAL_VALUES, pin))
        # Initialize SPI pins
        self.io.spi1_setpins(sck=self.pins.sck,
                             sdo=self.pins.sdi,
                             sdi=self.pins.sdo,
                             cs=self.pins.cs)
        self.io.spi1_enable(True)

    def reset(self):
        """Resets OpenTitan."""
        # Select JTAG and reset
        self.io.pin_set_state(self.pins.tap_strap1, 1)
        self.io.pin_set_state(self.pins.srst, 0)
        time.sleep(self.RESET_DELAY)
        # Deassert reset and return to SPI
        self.io.pin_set_state(self.pins.srst, 1)
        self.io.pin_set_state(self.pins.tap_strap1, 0)
        time.sleep(self.RESET_DELAY)

    def sw_strap_pins_set(self, val):
        """Sets the state of the sw_strap pins to `val`."""
        self.io.pin_set_state(self.pins.sw_strap0, val)
        self.io.pin_set_state(self.pins.sw_strap1, val)
        self.io.pin_set_state(self.pins.sw_strap2, val)

    @contextmanager
    def bootstrapper(self):
        """Context manager for handling strap pins.

        Puts OpenTitan in bootstrap mode and resets it at the end.
        """
        self.sw_strap_pins_set(1)
        self.reset()
        yield
        self.sw_strap_pins_set(0)
        self.reset()

    def transceive(self, data):
        """Transmits and receives data over SPI."""
        res = self.io.spi1_transfer(data)
        return res

    def read_status(self):
        """Reads the SPI Flash status register."""
        return self.transceive(bytes([0x05, 0xff]))[1]

    def write_enable(self):
        """Sends the write enable command."""
        self.transceive(bytes([0x06]))

    def write_enable_and_chip_erase(self):
        """Sends a chip erase command.

        Also handles enabling writes and busy polling.
        """
        self.write_enable()
        self.transceive(bytes([0xc7]))
        self.busy_poll()

    def write_enable_and_page_program(self, addr, data):
        """Sends a page program command.

        Also handles enabling writes and busy polling.
        """
        self.write_enable()
        packet = bytes([0x02]) + addr.to_bytes(
            3, byteorder="big", signed=False) + data
        self.transceive(packet)
        self.busy_poll()

    def busy_poll(self):
        """Polls until device is ready."""
        status = 1
        iter_cnt = 0
        while status & 1 != 0:
            if iter_cnt > self.MAX_BUSY_ITER_CNT:
                raise RuntimeError("Reached maximum iteration count")
            status = self.read_status()
            iter_cnt += 1
            time.sleep(self.BUSY_POLL_DELAY)

    def bootstrap(self, binary):
        """Bootstraps OpenTitan with the given binary."""
        with open(binary, mode='rb') as f:
            with self.bootstrapper() as _:
                self.write_enable_and_chip_erase()
                # Read fixed-size blocks from the firmware image.
                # Note: The second argument ``b''`` to ``iter`` below is the sentinel value that
                # ends the loop, i.e. the value returned by ``f.read`` at EOF.
                addr = 0
                for data in iter(partial(f.read, self.PAYLOAD_SIZE), b''):
                    print(
                        f'Programming {len(data)} bytes at address 0x{addr:08x}.'
                    )
                    self.write_enable_and_page_program(addr, data)
                    addr += len(data)
