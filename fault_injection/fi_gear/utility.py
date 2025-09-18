#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import random


def random_float_range(min: float, max: float, step: float) -> float:
    """Returns a random float between min and max with step."""
    return round(
        random.randint(0, int(round((max - min) / step))) * step + min, 4)
