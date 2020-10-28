#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Correlation-Enhanced Power Analysis Collision Attack

Reference: https://eprint.iacr.org/2010/297.pdf

See simple_cpa_attack.py for capture portion.
"""

import binascii
import chipwhisperer as cw
import numpy as np
import scared

from util import plot

# Configuration
start_sample_use = 35
num_samples_use = 100
stop_sample_use = start_sample_use + num_samples_use

num_sigmas = 3  # Amount of tolerable deviation from average during filtering.

plot_debug = False


def get_okay_traces(traces, upper_trace, lower_trace):
    okay_traces = np.zeros(len(traces), np.int8)
    for i in range(len(traces)):
        if (np.all(traces[i] <= upper_trace) and
                np.all(traces[i] >= lower_trace)):
            okay_traces[i] = 1
    return okay_traces


def get_max_rho(m_alpha_j, a, b):
    max_rho = [0, 0]
    # Get all correlation coefficients for byte positions a and b.
    rho_mat = np.corrcoef(m_alpha_j[a], m_alpha_j[b])
    #
    # Dimensions and organization:
    # - m_alpha_j is 16 x 256 x num_samples_use
    # - We fix two values in the first dimension (the byte position) and need
    #   all correlation coefficients between the resulting two
    #   256 x num_samples_use matrices.
    # - rho_mat is 512 x 512 and has the following structure:
    #
    #   c(a_0,  a_0) .. c(a_0,  a_255) | c(a_0,   b_0) .. c(a_0,   b_255)
    #   c(a_1,  a_0) .. c(a_1,  a_255) | c(a_1,   b_0) .. c(a_1,   b_255)
    #        .       ..      .         |      .        ..      .
    #   c(a_255,a_0) .. c(a_255,a_255) | c(a_255, b_0) .. c(a_255, b_255)
    #   -------------------------------|---------------------------------
    #   c(b_0,  a_0) .. c(b_0,  a_255) | c(b_0,   b_0) .. c(b_0,   b_255)
    #   c(b_1,  a_0) .. c(b_1,  a_255) | c(b_1,   b_0) .. c(b_1,   b_255)
    #        .       ..      .         |      .        ..      .
    #   c(b_255,a_0) .. c(b_255,a_255) | c(b_255, b_0) .. c(b_255, b_255)
    #
    #   where b_255 is m_alpha_j[b,255], i.e., the average of all traces with
    #   plaintext byte 255 at position b.
    #
    #   We only need the upper right square matrix and need to average that
    #   along the same  delta = plaintext_a ^ plaintext_b.
    #
    # Extract the upper right quarter.
    rho_mat = rho_mat[0:256, 256:512]
    # Average along the same delta.
    rho_avg_delta = np.zeros((256), np.double)
    rho_avg_delta_num = np.zeros((256))
    for i in range(256):
        for j in range(256):
            delta = i ^ j
            rho_avg_delta[delta] += rho_mat[i, j]
            rho_avg_delta_num[delta] += 1
    for delta in range(256):
        rho_avg_delta[delta] /= rho_avg_delta_num[delta]
    # Extract the maximum
    max_rho[1] = np.argmax(rho_avg_delta)
    max_rho[0] = rho_avg_delta[max_rho[1]]
    # Also return the rho vector for plotting
    rho = rho_avg_delta
    return max_rho, rho


def bit_count(number):
    bit_count = 0
    while number:
        number &= (number - 1)
        bit_count += 1
    return bit_count


if __name__ == '__main__':

    # Open trace file
    project_file = 'projects/opentitan_simple_aes'
    project = cw.open_project(project_file)

    num_traces = len(project.waves)
    num_samples = len(project.waves[0])

    # Create a local, dense copy of the traces. This makes the remaining
    # operations much faster.
    traces = np.empty((num_traces, num_samples_use), np.double)
    for i_trace in range(num_traces):
        traces[i_trace] = project.waves[i_trace][
            start_sample_use:stop_sample_use]

    ############################
    # Filter out noisy traces. #
    ############################

    # Get the mean and standard deviation.
    mean = traces.mean(axis=0)
    std = traces.std(axis=0)

    # Define upper and lower limits.
    upper_trace = mean + num_sigmas * std
    lower_trace = mean - num_sigmas * std

    # Filter the traces.
    okay_traces = get_okay_traces(traces, upper_trace, lower_trace)

    print('Will work with ' + str(np.sum(okay_traces)) + '/' +
          str(num_traces) + ' traces.')

    # Compute average over the filtered traces.
    avg_trace = np.zeros(num_samples_use, np.double)
    for i in range(len(traces)):
        if okay_traces[i]:
            avg_trace += traces[i]

    avg_trace /= np.sum(okay_traces)

    # Plot average trace and bad traces.
    if plot_debug:
        data = []
        data.append(avg_trace)
        for i in range(len(traces)):
            if not okay_traces[i]:
                data.append(traces[i])

        plot.save_plot_to_file(data[0:3], 3, 'avg_and_bad_traces.html')

    ###########################################################
    # Get average traces with value alpha at byte position j. #
    ###########################################################
    # Generate lists of all traces with value alpha at plaintext byte
    # position j. That's 16 x 2^8 = 4096 lists.

    # Pre-allocate the array of lists and average traces.
    lists = [[[] for j in range(256)] for alpha in range(16)]
    m_alpha_j = np.zeros((16, 256, num_samples_use), np.double)

    # Generate the lists.
    for i in range(num_traces):
        if okay_traces[i]:
            for j in range(16):
                alpha = project.textins[i][j]
                lists[j][alpha].append(i)

    # Detect empty lists.
    empty_lists = []
    for j in range(16):
        for alpha in range(256):
            num_alpha_traces = len(lists[j][alpha])
            if not num_alpha_traces:
                # We will handle empty lists later.
                empty_lists.append([j, alpha])

    # Compute m_alpha_j = average traces with value alpha at byte
    # position j.
    for j in range(16):
        for alpha in range(256):
            # Get number of traces with value alpha at plaintext byte
            # position j.
            num_alpha_traces = len(lists[j][alpha])
            if num_alpha_traces:
                # Sum up all traces.
                for trace in range(num_alpha_traces):
                    m_alpha_j[j, alpha] += traces[lists[j][alpha]
                                                  [trace]][0:num_samples_use]
                # Get the average.
                m_alpha_j[j, alpha] /= num_alpha_traces

    # Assign average trace to m_alpha_j with zero traces.
    for j, alpha in empty_lists:
        m_alpha_j[j, alpha] = avg_trace
        print("Didn't get a single trace with value alpha = " + str(alpha) +
              " at byte position j = " + str(j) +
              ". Will use the average trace.")

    # Plot m_alpha_j.
    if plot_debug:
        plot_m_alpha_j = [[] for i in range(16 * 256)]
        for j in range(16):
            for alpha in range(256):
                plot_m_alpha_j[j * 256 + alpha] = m_alpha_j[j, alpha]

        plot.save_plot_to_file(plot_m_alpha_j[0:3], 3, 'm_alpha_j.html')

    ##########################################################################
    # Find maximum correlation of m_alpha_j for every pair of plaintext byte #
    # positions a and b.                                                     #
    ##########################################################################
    max_rho_list = []
    rho_list = []
    for a in range(16):
        for b in range(a + 1, 16):
            max_rho, rho = get_max_rho(m_alpha_j, a, b)
            max_rho_list.append(max_rho)
            rho_list.append(rho)

    # Plot correlation coefficients for delta of plaintext byte pairs.
    if plot_debug:
        plot.save_plot_to_file(rho_list[0:3], 3, 'rho.html')

    # Convert into matrix.
    max_rho_mat = [[[0, 0] for b in range(16)] for a in range(16)]
    i = 0
    for a in range(16):
        for b in range(a + 1, 16):
            max_rho_mat[a][b] = max_rho_list[i]
            i += 1

    # Create ordered list of relationships between byte positions.
    max_rho_deltas = [[0, 0, 0, 0]]
    for a in range(16):
        for b in range(a + 1, 16):
            rho = max_rho_mat[a][b][0]
            delta = max_rho_mat[a][b][1]
            for i in range(len(max_rho_deltas)):
                if rho > max_rho_deltas[i][0]:
                    max_rho_deltas.insert(i, [rho, delta, a, b])
                    break

    # Remove the zero element.
    del max_rho_deltas[-1]

    #########################################
    # Find the key through trial and error. #
    #########################################

    # We take have 120 deltas and take the 15 most promising ones to try out
    # the remaining 256 key possibilties.
    key_guess_bytes = np.zeros((16), np.uint8)
    num_bytes_match = 0
    plaintext = project.textins[0]
    ciphertext = project.textouts[0]
    for i in range(256):
        key_temp = np.zeros((16), np.uint8)
        num_bytes_match_temp = 0
        status = np.zeros((16), np.bool_)
        # Start with the most promising delta.
        rho, delta, a, b = max_rho_deltas[0]
        key_temp[a] = i
        status[a] = True
        key_temp[b] = i ^ delta
        status[b] = True
        for byte in range(15):
            # Take the most promising delta that involves the available key
            # bytes.
            for rho, delta, a, b in max_rho_deltas:
                if status[a] ^ status[b]:
                    if status[a]:
                        key_temp[b] = key_temp[a] ^ delta
                        status[b] = True
                    elif status[b]:
                        key_temp[a] = key_temp[b] ^ delta
                        status[a] = True
                    continue

        # Encrypt and compare.
        ciphertext_temp = scared.aes.base.encrypt(plaintext, key_temp)
        for byte in range(16):
            if ciphertext_temp[byte] == ciphertext[byte]:
                num_bytes_match_temp += 1
        if num_bytes_match_temp >= num_bytes_match:
            num_bytes_match = num_bytes_match_temp
            key_guess_bytes = key_temp

        # Stop if we get a full match.
        if num_bytes_match == 16:
            break

    ##############################
    # Comparison with known key. #
    ##############################
    known_key_bytes = project.keys[0]

    known_key = binascii.b2a_hex(bytearray(known_key_bytes))
    print('known_key: {}'.format(known_key))

    key_guess = binascii.b2a_hex(bytearray(key_guess_bytes))
    print('key guess: {}'.format(key_guess))

    if key_guess != known_key:
        num_bytes_match = 0
        for i in range(len(known_key_bytes)):
            if known_key_bytes[i] == key_guess_bytes[i]:
                num_bytes_match += 1
        print('FAILED: key_guess != known_key')
        print('        ' + str(num_bytes_match) + '/' +
              str(len(known_key_bytes)) + ' bytes guessed correctly.')
    else:
        print('SUCCESS!')

    # Get known deltas.
    known_deltas = np.zeros((16, 16), np.uint8)
    known_rho_deltas = []
    for a in range(16):
        for b in range(a + 1, 16):
            known_deltas[a][b] = known_key_bytes[b] ^ known_key_bytes[a]
            known_rho_deltas.append([1.0, known_deltas[a][b], a, b])

    # Compare deltas.
    delta_diff = np.zeros((16, 16), np.uint8)
    num_bits_delta_diff = np.zeros((16, 16), np.uint8)
    for a in range(16):
        for b in range(16):
            delta_diff[a][b] = known_deltas[a][b] ^ max_rho_mat[a][b][1]
            num_bits_delta_diff[a][b] = bit_count(delta_diff[a][b])

    # Compare deltas for ordered list.
    delta_diff_list = []
    num_bits_delta_diff_list = []
    for rho, delta, a, b in max_rho_deltas:
        delta_diff_list.append(known_deltas[a, b] ^ delta)
        num_bits_delta_diff_list.append(bit_count(delta_diff_list[-1]))

    # Count number of matching deltas
    num_matching_deltas = 0
    for i in range(len(max_rho_deltas)):
        if not num_bits_delta_diff_list[i]:
            num_matching_deltas += 1

    print(
        str(num_matching_deltas) + '/' + str(len(max_rho_deltas)) +
        ' deltas guessed correctly.')
