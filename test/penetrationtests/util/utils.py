import json

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