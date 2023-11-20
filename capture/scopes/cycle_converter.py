# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0


def convert_num_cycles(cfg: dict, scope_type: str) -> int:
    """ Convert number of cycles to number of samples.

    As Husky is configured in number of samples, this function converts the
    number of cycles to samples. The number of samples must be divisble by 3
    for batch captures.

    Args:
        dict: The scope configuration.
        scope_type: The used scope (Husky or WaveRunner).

    Returns:
        The number of samples.
    """
    sampling_target_ratio = cfg[scope_type].get("sampling_rate") / cfg["target"].get("target_freq")
    num_samples = int(cfg[scope_type].get("num_cycles") * sampling_target_ratio)
    if num_samples % 3:
        num_samples = num_samples + 3 - (num_samples % 3)
    return num_samples


def convert_offset_cycles(cfg: dict, scope_type: str) -> int:
    """ Convert offset in cycles to offset in samples.

    Args:
        dict: The scope configuration.
        scope_type: The used scope (Husky or WaveRunner).

    Returns:
        The offset in samples.
    """
    sampling_target_ratio = cfg[scope_type].get("sampling_rate") / cfg["target"].get("target_freq")
    return int(cfg[scope_type].get("offset_cycles") * sampling_target_ratio)
