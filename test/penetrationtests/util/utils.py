import json
import struct

def compare_json_data(expected_data: dict, actual_data: dict, ignored_keys: set) -> bool:
    expected_comparable_keys = set(expected_data.keys()) - ignored_keys
    actual_comparable_keys = set(actual_data.keys()) - ignored_keys

    if expected_comparable_keys != actual_comparable_keys:
        return False

    for key in expected_comparable_keys:
        if expected_data[key] != actual_data[key]:
            return False
    return True

def to_signed32(n_unsigned):
    n_unsigned = n_unsigned & 0xFFFFFFFF 
    if n_unsigned >= 0x80000000: 
        return n_unsigned - 0x100000000 
    return n_unsigned

def bytes_to_32bit_words(byte_array):
    if not isinstance(byte_array, (bytes, bytearray)):
        raise TypeError("Input must be a bytes object or bytearray.")

    word_list = []
    for i in range(0, len(byte_array), 4):
        chunk = byte_array[i:i+4]
        word = struct.unpack('>I', chunk)[0]
        word_list.append(word)
    return word_list