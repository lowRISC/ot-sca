# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the SHA3 SCA application on OpenTitan.

Communication with OpenTitan either happens over simpleserial or the uJson
command interface.
"""


class OTOTBNVERT:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        self.simple_serial = True
        if protocol == "ujson":
            self.simple_serial = False

    def choose_otbn_app(self, app):
        """ Select the OTBN application.
        Args:
            app: OTBN application
        """
        if self.simple_serial:
            # Select the otbn app on the device (0 -> keygen, 1 -> modinv).
            if app == 'keygen':
                self.target.write(cmd="a", data=bytearray([0x00]))
            if app == 'modinv':
                self.target.write(cmd="a", data=bytearray([0x01]))

    def write_batch_prng_seed(self, seed):
        """ Seed the PRNG.
        Args:
            seed: The 4-byte seed.
        """
        if self.simple_serial:
            self.target.write(cmd="s", data=seed)

    def write_keygen_seed(self, seed):
        """ Write the seed used for the keygen app.
        Args:
            seed: byte array containing the seed.
        """
        if self.simple_serial:
            self.target.write(cmd='x', data=seed)

    def write_keygen_key_constant_redundancy(self, const):
        """ Write the constant redundancy value for the keygen app.
        Args:
            seed: byte array containing the redundancy value.
        """
        if self.simple_serial:
            self.target.write(cmd="c", data=const)

    def config_keygen_masking(self, off):
        """ Disable/enable masking.
        Args:
            off: boolean value.
        """
        if self.simple_serial:
            # Enable/disable masking.
            if off is True:
                self.target.write(cmd="m", data=bytearray([0x00]))
            else:
                self.target.write(cmd="m", data=bytearray([0x01]))

    def start_keygen(self, mask):
        """ Write the seed mask and start the keygen app.
        Args:
            mask: byte array containing the mask value.
        """
        if self.simple_serial:
            # Send the mask and start the keygen operation.
            self.target.write('k', mask)

    def start_modinv(self, scalar_k0, scalar_k1):
        """ Write the two scalar shares and start the modinv app.
        Args:
            scalar_k0: byte array containing the first scalar share.
            scalar_k1: byte array containing the second scalar share.
        """
        if self.simple_serial:
            # Start modinv device computation.
            self.target.write('q', scalar_k0 + scalar_k1)

    def start_keygen_batch(self, test_type, num_segments):
        """ Start the keygen app in batch mode.
        Args:
            test_type: string selecting the test type (KEY or SEED).
            num_segments: number of keygen executions to perform.
        """
        if self.simple_serial:
            # Start batch keygen.
            if test_type == 'KEY':
                self.target.write(cmd="e", data=num_segments)
            else:
                self.target.write(cmd="b", data=num_segments)

    def read_output(self, len_bytes):
        """ Read the output from whichever OTBN app.
        Args:
            len_bytes: Number of bytes to read.

        Returns:
            The received output.
        """
        if self.simple_serial:
            return self.target.read("r", len_bytes, ack=False)
