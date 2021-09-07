# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import binascii

import cffi

from . import _xkcp

ffi = cffi.FFI()


def kmac128_raw(key, keyLen, input, inputLen, output, outputLen, customization,
                customLen):
    # all data (besides lengths) are passed as references, like in the C implementation

    # from bytearray to uint8 pointer
    key_p = ffi.from_buffer(key)
    input_p = ffi.from_buffer(input)
    output_p = ffi.from_buffer(output)
    customization_p = ffi.from_buffer(customization)
    status = _xkcp.lib.KMAC128(key_p, keyLen * 8, input_p, inputLen * 8,
                               output_p, outputLen * 8, customization_p,
                               customLen * 8)
    return status


def kmac128(key, keyLen, input, inputLen, outputLen, customization, customLen):
    # experimental:  output is returned, no need for a reference
    output = bytearray(outputLen)
    status = kmac128_raw(key, keyLen, input, inputLen, output, outputLen,
                         customization, customLen)
    if status == 0:
        return output


def kmac256_raw(key, keyLen, input, inputLen, output, outputLen, customization,
                customLen):
    ### NOT TESTED ATM ###

    #from bytearray to uint8 pointer
    key_p = ffi.from_buffer(key)
    input_p = ffi.from_buffer(input)
    output_p = ffi.from_buffer(output)
    customization_p = ffi.from_buffer(customization)
    status = _xkcp.lib.KMAC256(key_p, keyLen * 8, input_p, inputLen * 8,
                               output_p, outputLen * 8, customization_p,
                               customLen * 8)
    return status


def kmac256(key, keyLen, input, inputLen, outputLen, customization, customLen):
    ### NOT TESTED ATM ###

    output = bytearray(outputLen)
    status = kmac256_raw(key, keyLen, input, inputLen, output, outputLen,
                         customization, customLen)
    if status == 0:
        return output


def test():
    print(
        "https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/KMAC_samples.pdf"
    )
    print("KMAC: Sample #1")
    print("Security Strength: 128-bits")
    print("Length of Key is 256-bits")
    print("Length of data is 32-bits")
    print("Requested output length is 256-bits")

    keyLen = 32
    inputLen = 4
    outputLen = 32
    customLen = 0

    # with python buffers/bytearray
    key = bytearray(keyLen)
    input = bytearray(inputLen)
    output = bytearray(outputLen)
    customization = bytearray(customLen)

    key = b'\x40\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4A\x4B\x4C\x4D\x4E\x4F\x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5A\x5B\x5C\x5D\x5E\x5F'
    input = b'\x00\x01\x02\x03'
    customization = b'\x00'
    expected_output = b'\xE5\x78\x0B\x0D\x3E\xA6\xF7\xD3\xA4\x29\xC5\x70\x6A\xA4\x3A\x00\xFA\xDB\xD7\xD4\x96\x28\x83\x9E\x31\x87\x24\x3F\x45\x6E\xE1\x4E'

    print("key:                  ", binascii.hexlify(key))
    print("input:                ", binascii.hexlify(input))
    print("S:                    ", "\"(null)\"")
    print("expected output:      ", binascii.hexlify(expected_output))

    kmac128_raw(key, keyLen, input, inputLen, output, outputLen, customization,
                customLen)
    print("Output by reference : ", binascii.hexlify(output))
    if output == expected_output:
        print("Output by reference matches. PASS!")
    else:
        print("Output by reference does not match!")

    ret = kmac128(key, keyLen, input, inputLen, outputLen, customization,
                  customLen)
    print("Output by value:      ", binascii.hexlify(ret))
    if ret == expected_output:
        print("Output by value matches. PASS!")
    else:
        print("Output by value does not match!")
