#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from chipwhisperer.analyzer import aes_funcs


def bit_count(int_no):
    """Computes Hamming weight of a number."""
    c = 0
    while int_no:
        int_no &= int_no - 1
        c += 1
    return c


def compute_leakage_aes(keys, plaintexts, leakage_model = 'HAMMING_WEIGHT'):
    """
    Computes AES leakage for a given list of plaintexts and keys.

    The output "leakage" contains leakage of all state-register bytes after each round.
        leakage[X][Y][Z] - Leakage (e.g. hamming weight) of AES round X, byte Y for trace Z
    Leakage is computed based on the specified leakage_model.
    Two leakage models are available:
        HAMMING_WEIGHT - based on the hamming weight of the state register byte.
        HAMMING_DISTANCE - based on the hamming distance between the curent and previous state.
    """
    num_traces = len(keys)
    leakage = np.zeros((11, 16, num_traces), dtype=np.uint8)

    # Checks if all keys in the list are the same.
    key_fixed = np.all(keys == keys[0])
    subkey = np.zeros((11, 16))

    if key_fixed:
        for j in range(11):
            subkey[j] = np.asarray(
                aes_funcs.key_schedule_rounds(keys[0], 0, j))
        subkey = subkey.astype(int)

    for i in range(num_traces):

        if not key_fixed:
            for j in range(11):
                subkey[j] = np.asarray(
                    aes_funcs.key_schedule_rounds(keys[i], 0, j))
            subkey = subkey.astype(int)

        # Init
        state = plaintexts[i]

        # Round 0
        old_state = state
        state = np.bitwise_xor(state, subkey[0])
        for k in range(16):
            if leakage_model == 'HAMMING_DISTANCE':
                leakage[0][k][i] = bit_count(
                    np.bitwise_xor(state[k], old_state[k]))
            else:
                leakage[0][k][i] = bit_count(state[k])

        # Round 1 - 10
        for j in range(1, 11):
            old_state = state
            state = aes_funcs.subbytes(state)
            state = aes_funcs.shiftrows(state)
            if (j < 10):
                state = aes_funcs.mixcolumns(state)
            state = np.bitwise_xor(state, subkey[j])
            for k in range(16):
                if leakage_model == 'HAMMING_DISTANCE':
                    leakage[j][k][i] = bit_count(
                        np.bitwise_xor(state[k], old_state[k]))
                else:
                    leakage[j][k][i] = bit_count(state[k])

    return leakage


def find_fixed_key(keys):
    """
    Finds a fixed key.

    In a fixed-vs-random analysis, only fixed_key will repeat multiple times,
    this will not necesserily be the first key on the list.
    This function looks at the input list of keys and finds the first one that
    is repeated multiple times.
    """

    for i_key in range(len(keys)):
        fixed_key = keys[i_key]
        num_hits = 0
        for i in range(len(keys)):
            num_hits += np.array_equal(fixed_key, keys[i])
        if num_hits > 1:
            break

    # If no key repeats, then the fixed key cannot be identified.
    assert num_hits > 1, "Cannot identify fixed key. Try using a longer list."

    return fixed_key


def compute_leakage_general(keys, fixed_key):
    """
    Computes leakage for TVLA fixed-vs-random general attaks.

    Output "leakage" shows whether a given trace belongs to the fixed or random
    group.
        leakage[i] = 1 - trace i belonges to the fixed group
        leakage[i] = 0 - trace i belonges to the random group
    """

    leakage = np.zeros((len(keys)), dtype=np.uint8)
    for i in range(len(keys)):
        leakage[i] = np.array_equal(fixed_key, keys[i])

    return leakage
