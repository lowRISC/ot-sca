# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for different OpenTitan ciphers.

Currently, communication with each cipher happens over simpleserial. This file
provides communication interface classes for each of these ciphers.
"""


class OTAES:
    def __init__(self, target) -> None:
        self.target = target

    def write_key(self, key):
        """ Write key to AES.
        Args:
            key: Bytearray containing the key.
        """
        self.target.simpleserial_write("k", key)

    def fvsr_key_set(self, key):
        """ Write key to AES.
        Args:
            key: Bytearray containing the key.
        """
        self.target.simpleserial_write("f", key)

    def write_lfsr_seed(self, seed):
        """ Seed the LFSR.
        Args:
            seed: The 4-byte seed.
        """
        self.target.simpleserial_write("l", seed)

    def write_batch_prng_seed(self, seed):
        """ Seed the PRNG.
        Args:
            seed: The 4-byte seed.
        """
        self.target.simpleserial_write("s", seed)

    def start_fvsr_batch_generate(self):
        """Set SW PRNG to starting values for FvsR data
        generation.
        """
        command = 1
        self.target.simpleserial_write("d", command.to_bytes(4, "little"))
        self.target.simpleserial_wait_ack()

    def write_fvsr_batch_generate(self, num_segments):
        """ Generate random plaintexts for FVSR.
        Args:
            num_segments: Number of encryptions to perform.
        """
        self.target.simpleserial_write("g", num_segments)

    def encrypt_batch(self, num_segments):
        """ Start encryption for batch.
        Args:
            num_segments: Number of encryptions to perform.
        """
        self.target.simpleserial_write("a", num_segments)
        self.target.simpleserial_wait_ack()

    def encrypt_fvsr_key_batch(self, num_segments):
        """ Start encryption for FVSR.
        Args:
            num_segments: Number of encryptions to perform.
        """
        self.target.simpleserial_write("e", num_segments)

    def write_init_text(self, text):
        """ Write plaintext to OpenTitan AES.
        Args:
            text: The plaintext bytearray.
        """
        self.target.simpleserial_write("i", text)

    def encrypt(self, text):
        """ Write plaintext to OpenTitan AES & start encryption.
        Args:
            text: The plaintext bytearray.
        """
        self.target.simpleserial_write("p", text)

    def read_ciphertext(self, len_bytes):
        """ Read ciphertext from OpenTitan AES.
        Args:
            len_bytes: Number of bytes to read.

        Returns:
            The received ciphertext.
        """
        return self.target.simpleserial_read("r", len_bytes, ack=False)
