# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from Crypto.Cipher import AES


class data_generator():

    key_generation = bytearray([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF1,
                                0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xE0, 0xF0])

    text_fixed = bytearray([0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
                            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA])
    text_random = bytearray([0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC,
                             0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC])
    key_fixed = bytearray([0x81, 0x1E, 0x37, 0x31, 0xB0, 0x12, 0x0A, 0x78,
                           0x42, 0x78, 0x1E, 0x22, 0xB2, 0x5C, 0xDD, 0xF9])
    key_random = bytearray([0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53,
                            0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53])

    cipher_gen = AES.new(bytes(key_generation), AES.MODE_ECB)

    def __init__(self):
        self.cipher_gen = AES.new(bytes(self.key_generation), AES.MODE_ECB)

    def set_start(self):
        self.text_fixed = bytearray([0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
                                     0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA])
        self.text_random = bytearray([0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC,
                                      0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC])
        self.key_fixed = bytearray([0x81, 0x1E, 0x37, 0x31, 0xB0, 0x12, 0x0A, 0x78,
                                    0x42, 0x78, 0x1E, 0x22, 0xB2, 0x5C, 0xDD, 0xF9])
        self.key_random = bytearray([0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53,
                                     0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x53])

    def advance_fixed(self):
        self.text_fixed = bytearray(self.cipher_gen.encrypt(self.text_fixed))

    def advance_random(self):
        self.text_random = bytearray(self.cipher_gen.encrypt(self.text_random))
        self.key_random = bytearray(self.cipher_gen.encrypt(self.key_random))

    def get_fixed(self):
        pt = np.asarray(self.text_fixed)
        key = np.asarray(self.key_fixed)
        cipher_fixed = AES.new(bytes(self.key_fixed), AES.MODE_ECB)
        ct = np.asarray(bytearray(cipher_fixed.encrypt(self.text_fixed)))
        del (cipher_fixed)
        self.advance_fixed()
        return pt, ct, key

    def get_random(self):
        pt = np.asarray(self.text_random)
        key = np.asarray(self.key_random)
        cipher_random = AES.new(bytes(self.key_random), AES.MODE_ECB)
        ct = np.asarray(bytearray(cipher_random.encrypt(self.text_random)))
        del (cipher_random)
        self.advance_random()
        return pt, ct, key


# ----------------------------------------------------------------------
# Create one instance, and export its methods as module-level functions.
# The functions share state across all uses.

_inst = data_generator()
set_start = _inst.set_start
get_fixed = _inst.get_fixed
get_random = _inst.get_random
