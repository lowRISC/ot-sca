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