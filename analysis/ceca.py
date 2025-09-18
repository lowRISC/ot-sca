#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""A distributed implementation of the correlation-enhanced power analysis
collision attack.

See "Correlation-Enhanced Power Analysis Collision Attack" by A. Moradi, O.
Mischke, and T. Eisenbarth (https://eprint.iacr.org/2010/297.pdf) for more
information.

Typical usage:
>>> ./ceca.py -f PROJECT_FILE -n 400000 -w 5 -a 117 127 -d output -s 3
"""

import argparse
import enum
import logging
import os
import sys
from pathlib import Path

import chipwhisperer.analyzer as cwa
import codetiming
import more_itertools
import networkx as nx
import numpy as np
import ray
import scared

# Append ot-sca root directory to path such that ceca.py can find the
# project_library module located in the capture/ directory.
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_PATH + "/..")
from capture.project_library.project import ProjectConfig  # noqa : E402
from capture.project_library.project import SCAProject  # noqa : E402


def timer():
    """A customization of the ``codetiming.Timer`` decorator."""

    def decorator(func):

        @codetiming.Timer(
            name=func.__name__,
            text=f"{func.__name__} took {{seconds:.1f}}s",
            logger=logging.info,
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class AttackDirection(str, enum.Enum):
    """Enumeration for attack direction."""

    INPUT = "input"
    OUTPUT = "output"


@ray.remote
class TraceWorker:
    """Class for performing distributed computations on power traces.

    This class provides methods for performing distributed computations on power
    traces such as computing the mean, the standard deviation, and filtering.
    After creating multiple instances (workers) of this class and initializing
    each worker with a subset of the available traces, distributed computations
    can be performed by simply calling the methods of these workers. Individual
    results of these workers can then be aggregated to produce the final result.

    This class is a Ray actor (https://docs.ray.io/en/master/index.html) and can
    be used as follows:
    >>> workers = [TraceWorker.remote(...) for ...]
    >>> tasks = [worker.compute_stats.remote() for worker in workers]
    >>> results = ray.get(tasks)
    """

    def __init__(self, project_file, trace_slice, attack_window,
                 attack_direction):
        """Inits a TraceWorker.

        Args:
            project_file: A Chipwhisperer or ot_trace_library project file.
            trace_slice: Traces assigned to this worker.
            attack_window: Samples to process.
            attack_direction: Attack direction.
        """
        # ChipWhisperer or ot_trace_library project?
        project_type = "cw"
        if ".db" in project_file:
            project_type = "ot_trace_library"

        # Open the project.
        project_cfg = ProjectConfig(type=project_type,
                                    path=project_file,
                                    wave_dtype=np.uint16,
                                    overwrite=False)
        self.project = SCAProject(project_cfg)
        self.project.open_project()

        # TODO: Consider more efficient formats.
        self.num_samples = attack_window.stop - attack_window.start
        if attack_direction == AttackDirection.INPUT:
            self.texts = np.vstack(
                self.project.get_plaintexts(trace_slice.start,
                                            trace_slice.stop))
        else:
            self.texts = np.vstack(
                self.project.get_ciphertexts(trace_slice.start,
                                             trace_slice.stop))

        self.traces = np.asarray(
            self.project.get_waves(trace_slice.start,
                                   trace_slice.stop))[:, attack_window]

        self.project.close(save=False)

    def compute_stats(self):
        """Computes sums and sums of deviation products of traces.

        Results from multiple workers can be aggregated to compute the standard
        deviation of a set of traces in a distributed manner using Eq. 22 in
        "Numerically Stable Parallel Computation of (Co-)Variance" by E.
        Schubert and M. Gertz (https://dbs.ifi.uni-heidelberg.de/files/Team/
        eschubert/publications/SSDBM18-covariance-authorcopy.pdf).

        Returns:
            Number of traces, their sums, and sums of deviation products.
        """
        cnt = self.traces.shape[0]
        sum_ = self.traces.sum(axis=0)
        mean = sum_ / cnt
        sum_dev_prods = ((self.traces - mean)**2).sum(axis=0)
        return (cnt, sum_, sum_dev_prods)

    def filter_noisy_traces(self, min_trace, max_trace):
        """Filters traces with values outside the allowable range.

        Args:
            min_trace: Minimum allowable values.
            max_trace: Maximum allowable values.

        Returns:
            Number of remaining traces.
        """
        traces_to_use = np.all(
            (self.traces >= min_trace) & (self.traces <= max_trace), axis=1)
        self.traces = self.traces[traces_to_use]
        self.texts = self.texts[traces_to_use]
        return self.traces.shape[0]

    def count_and_sum_text_traces(self):
        """Computes the number of traces and sums of these traces for all values
        of each text byte.

        Returns:
            A tuple ``(cnts, sums)``, where
                - ``cnts`` is a (16, 256, 1) array where ``cnts[i, j, 0]`` gives the
                  number of traces where text byte i is j, and
                - ``sums`` is a (16, 256, NUM_SAMPLES) array where ``sums[i, j, :]``
                  gives the sum of traces where text byte i is j.
        """
        sums = np.zeros((16, 256, self.num_samples))
        # Need to specify the last dimension for broadcasting to work during
        # aggregation.
        cnts = np.zeros((16, 256, 1))
        for byte_pos in range(16):
            # While a little bit more complex, below code is more efficient than
            # a naive implementation that searches for all possible byte values
            # in ``self.texts``.
            sorted_indices = self.texts[:, byte_pos].argsort()
            sorted_bytes = self.texts[sorted_indices, byte_pos]
            # Find the indices where byte values change.
            val_changes = np.where(np.roll(sorted_bytes, 1) != sorted_bytes)[0]
            # Append the number of rows to be able to use ``pairwise``.
            val_indices = list(val_changes) + [sorted_bytes.shape[0]]
            for start, end in more_itertools.pairwise(val_indices):
                byte_val = sorted_bytes[start]
                cnts[byte_pos, byte_val] = end - start
                act_indices = sorted_indices[start:end]
                sums[byte_pos, byte_val] = self.traces[act_indices].sum(axis=0)
        return cnts, sums


def compute_mean_and_std(workers):
    """Computes mean and standard deviation of all traces.

    This function uses Eq. 22 in "Numerically Stable Parallel Computation of
    (Co-)Variance" by E. Schubert and M. Gertz (https://dbs.ifi.uni-heidelberg.
    de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf) to
    distribute standard deviation computation to multiple ``TraceWorker``
    instances.

    Args:
        workers: ``TraceWorker`` handles.

    Returns:
        Mean and standard deviation of all traces.
    """
    tasks = [worker.compute_stats.remote() for worker in workers]

    running_sum_dev_prods = None
    running_sum = None
    running_cnt = 0
    while tasks:
        done, tasks = ray.wait(tasks)
        cnt, sum_, sum_dev_prods = ray.get(done[0])
        if running_cnt == 0:
            running_sum_dev_prods = np.copy(sum_dev_prods)
            running_sum = np.copy(sum_)
            running_cnt += cnt
        else:
            running_sum_dev_prods += sum_dev_prods + (
                (cnt * running_sum - running_cnt * sum_)**2 /
                (cnt * running_cnt * (cnt + running_cnt)))
            running_sum += sum_
            running_cnt += cnt
    return running_sum / running_cnt, np.sqrt(running_sum_dev_prods /
                                              running_cnt)


def filter_noisy_traces(workers, mean_trace, std_trace, max_std):
    """Signals ``TraceWorker`` instances to filter noisy traces.

    Args:
        workers:``TraceWorker`` handles.
        mean_trace: Mean of all traces.
        std_trace: Standard deviation of all traces.
        max_std: Allowed number of standard deviations from the mean trace.

    Returns:
        Number of remaining traces.
    """
    min_trace = mean_trace - max_std * std_trace
    max_trace = mean_trace + max_std * std_trace
    tasks = [
        worker.filter_noisy_traces.remote(min_trace, max_trace)
        for worker in workers
    ]

    running_cnt = 0
    while tasks:
        done, tasks = ray.wait(tasks)
        running_cnt += ray.get(done[0])
    return running_cnt


def compute_mean_text_traces(workers):
    """Computes mean traces for all values of all text bytes.

    This function distributes work to ``TraceWorker`` instances and collects
    their results.

    Args:
        workers: ``TraceWorker`` handles.

    Returns:
        A (16, 256, NUM_SAMPLES) array A, where A[i, j, :] is the mean of all
        traces where text byte i is j.
    """
    tasks = [worker.count_and_sum_text_traces.remote() for worker in workers]

    running_cnt = None
    running_sum = None
    while tasks:
        done, tasks = ray.wait(tasks)
        cnt, sum_ = ray.get(done[0])
        if running_cnt is None:
            running_cnt = np.copy(cnt)
            running_sum = np.copy(sum_)
        else:
            running_cnt += cnt
            running_sum += sum_

    return running_sum / running_cnt


@timer()
def compute_pairwise_diffs_and_scores(mean_traces):
    """Computes pairwise differences and their confidence scores between key
    bytes.

    This function correlates mean text traces to pick the most likely
    differences.

    Args:
        mean_traces: A (16, 256, NUM_SAMPLES) array of mean text traces.

    Returns:
        A (16, 16, 2) array A, where A[i, j, 0] is the difference with the
        largest correlation coefficient between text bytes i and j, and
        A[i, j, 1] is the corresponding confidence score.
    """
    pairwise_diffs_scores = np.zeros((16, 16, 2))
    # All possible values of text byte a.
    alphas = np.arange(256)
    # All possible differences.
    diffs = np.arange(256)[:, np.newaxis]
    # Values of text byte b for all possible differences: A (256, 256)
    # array where betas[i, j] = diffs[i] ^ alphas[j]
    betas = alphas ^ diffs
    # Compute the most likely differences between all pairs of text bytes.
    for a in range(16):
        for b in range(a + 1, 16):
            corrcoefs = np.corrcoef(mean_traces[a], mean_traces[b])
            # np.corrcoef returns the values that we want in the upper right
            # quadrant.
            diff_corrcoefs = corrcoefs[alphas, 256 + betas].sum(axis=1)
            best_diff = diff_corrcoefs.argmax()
            # TODO: Analyze the effect of /diff_corrcoefs.mean() below.
            pairwise_diffs_scores[(a, b), (b, a)] = (
                best_diff,
                diff_corrcoefs[best_diff] / diff_corrcoefs.mean(),
            )
    return pairwise_diffs_scores


class DiffScore:
    """Class for using confidence scores of differences as edge weights.

    This class overloads + and < operators such that
        - ``a + b`` returns ``min(a._val, b._val)``, and
        - ``a < b`` returns ``a._val > b._val``.
    This allows reuse of well-known graph algorithms, such as Dijkstra's
    shortest path algorithm, to find the most likely differences between key
    bytes without further modifications.

    This class can be modified further to accommodate additional requirements.

    See also: ``find_best_diffs()``.
    """

    def __init__(self, val):
        """Inits a DiffScore.

        Args:
            val: Confidence score of this difference.
        """
        self._val = val

    def __add__(self, other):
        # Special case for 0 is necessary because it's hardcoded as the distance
        # to source node.
        return self if other == 0 else DiffScore(min(self._val, other._val))

    def __radd__(self, other):
        return self + other

    def __lt__(self, other):
        # Special case for 0 is necessary because it's hardcoded as the distance
        # to source node.
        return False if other == 0 else self._val > other._val

    def __repr__(self):
        return f"DiffScore({self._val})"


def find_best_diffs(pairwise_diffs_scores):
    """Finds the most likely differences between key bytes.

    This function finds the most likely differences between key bytes using
    pairwise differences between all key bytes and their scores computed from
    mean text traces. In order to utilize all available information, this
    function runs a modified version of Dijkstra's shortest path algorithm on a
    graph, where:
        - Nodes are key bytes,
        - Edges represent the differences between key bytes, and
        - Edge weights are confidence scores of these differences.
    The behavior of the shortest path algorithm is modified using the
    ``DiffScore`` class such that:
        - The cost/distance of a path is the minimum of the weights, i.e.
          confidence scores, of the edges on the path, and
        - Paths with higher confidence scores are preferred.
    Consequently, paths found by the algorithm are those that maximize the
    confidence scores of the differences between key bytes.

    Args:
        pairwise_diffs_scores: A 16x16x2 matrix of pairwise differences between
            key bytes and their confidence scores.

    Returns:
        An array of differences where the ith element is key[0] ^ key[i].
    """
    # Define an undirected graph where nodes are key bytes, edges represent
    # differences, and edge weights are confidence scores.
    G = nx.Graph()
    for a in range(16):
        for b in range(a + 1, 16):
            # Use the ``DiffScore`` class, instead of ``float``s, to find
            # the most likely differences between key bytes.
            G.add_edge(a, b, weight=DiffScore(pairwise_diffs_scores[a, b, 1]))
    # Find paths from key byte 0 to all other bytes.
    paths = nx.algorithms.shortest_paths.weighted.single_source_dijkstra_path(
        G, 0)
    # Recover the paths and corresponding differences from key byte 0 to all
    # other bytes.
    diffs = np.zeros(16, dtype=np.uint8)
    for byte in range(1, 16):
        for a, b in more_itertools.pairwise(paths[byte]):
            diffs[byte] ^= int(pairwise_diffs_scores[a, b, 0])
    return diffs


def recover_key(diffs, attack_direction, plaintext, ciphertext):
    """Recovers the AES key using the differences between key bytes.

    This function uses the differences between key bytes to recover the key by
    iterating over all possible values of the first key byte. The correct key is
    identified by comparing the ciphertexts obtained using all key candidates to
    the actual ciphertext.

    Args:
        diffs: Differences between key bytes.
        attack_direction: Attack direction, 'input' or 'output'.
        plaintext: A plaintext, used for identifying the correct key.
        ciphertext: A ciphertext, used for identifying the correct key.

    Returns:
        The AES key if successful, ``None`` otherwise.
    """
    # Create a matrix of all possible keys.
    keys = np.zeros((256, 16), np.uint8)
    for first_byte_val in range(256):
        key = np.asarray([diffs[i] ^ first_byte_val for i in range(16)],
                         np.uint8)
        if attack_direction == AttackDirection.OUTPUT:
            key = np.asarray(cwa.aes_funcs.key_schedule_rounds(key, 10, 0),
                             np.uint8)
        keys[first_byte_val] = key
    # Encrypt the plaintext using all candidates in parallel.
    ciphertexts = scared.aes.base.encrypt(plaintext, keys)
    # Recover the key.
    key = keys[(ciphertexts == ciphertext).all(axis=1).nonzero()]
    if key.size > 0:
        return key
    return None


def compare_diffs(pairwise_diffs_scores, attack_direction, correct_key):
    """Compares pairwise_diffs_scores with diffs between bytes in correct_key.

    This function takes the differences between key bytes computed in
    pairwise_diffs_scores and compares them with the actual differences in the
    correct key.

    Args:
        pairwise_diffs_scores: A 16x16x2 matrix of pairwise differences between
            key bytes and their confidence scores.
        attack_direction: Attack direction.
        correct_key: The correct key.

    Returns:
        A 16x16 matrix indicating which pairswise differences between key bytes
            have been recovered correctly.
    """
    if attack_direction == AttackDirection.OUTPUT:
        end_key = cwa.aes_funcs.key_schedule_rounds(correct_key, 0, 10)
        correct_key = np.asarray(end_key, np.uint8)
    correct_diffs = np.zeros((16, 16), np.uint8)
    for i in range(16):
        for j in range(i, 16):
            correct_diffs[i, j] = correct_key[i] ^ correct_key[j]
            correct_diffs[j, i] = correct_diffs[i, j]
    return pairwise_diffs_scores[:, :, 0] == correct_diffs


@timer()
def perform_attack(project_file, num_traces, attack_window, attack_direction,
                   max_std, num_workers):
    """Performs a correlation-enhanced power analysis collision attack.

    This function:
        - Computes the mean and standard deviation of all traces (*),
        - Filters noisy traces (*),
        - Computes mean traces for all values of all plaintext/ciphertext bytes (*),
        - Guesses differences between each key byte, and
        - Recovers the key using these differences.

    Steps marked with (*) above are implemented in a distributed manner: After
    creating ``num_workers`` number of ``TraceWorker`` instances, this function
    assigns a subset of traces to each worker and aggregates their results to be
    used in the subsequent steps of the attack.

    See "Correlation-Enhanced Power Analysis Collision Attack" by A. Moradi, O.
    Mischke, and T. Eisenbarth (https://eprint.iacr.org/2010/297.pdf) for more
    information.

    Args:
        project_file: A Chipwhisperer or ot_trace_library project file.
        num_traces: Number of traces to use, must be less than or equal to the
            number of traces in ``project_file``.
        attack_window: Attack window as a pair of sample indices, inclusive.
        attack_direction: Attack direction.
        max_std: Allowed number of standard deviations from the mean trace for
            filtering noisy traces.
        num_workers: Number of workers to use for processing traces.

    Returns:
        Recovered key if the attack was successful, ``None`` otherwise.
    """
    # Translate potential relative path into absolute path. Needed for ray().
    project_file = str(Path(project_file).resolve())
    # ChipWhisperer or ot_trace_library project?
    project_type = "cw"
    if ".db" in project_file:
        project_type = "ot_trace_library"

    # Open the project.
    project_cfg = ProjectConfig(type=project_type,
                                path=project_file,
                                wave_dtype=np.uint16,
                                overwrite=False)
    project = SCAProject(project_cfg)
    project.open_project()

    # Check arguments
    num_total_traces = len(project.get_waves())
    if num_traces > num_total_traces:
        raise ValueError(
            f"Invalid num_traces: {num_traces} (must be less than {num_total_traces})"
        )
    last_sample = len(project.get_waves(0)) - 1
    if min(attack_window) < 0 or max(attack_window) > last_sample:
        raise ValueError(
            f"Invalid attack window: {attack_window} (must be in [0, {last_sample}])"
        )
    if max_std <= 0:
        raise ValueError(
            f"Invalid max_std: {max_std} (must be greater than zero)")
    if num_workers <= 0:
        raise ValueError(
            f"Invalid num_workers: {num_workers} (must be greater than zero)")

    # Instantiate workers
    def worker_trace_slices():
        """Determines the traces of each worker.

        Assigns the remainder, if any, to the first worker.
        """
        traces_per_worker = int(num_traces / num_workers)
        first_worker_num_traces = traces_per_worker + num_traces % num_workers
        yield slice(0, first_worker_num_traces)
        for trace_begin in range(first_worker_num_traces, num_traces,
                                 traces_per_worker):
            yield slice(trace_begin, trace_begin + traces_per_worker)

    # Attack window is inclusive.
    attack_window = slice(attack_window[0], attack_window[1] + 1)
    workers = [
        TraceWorker.remote(project_file, trace_slice, attack_window,
                           attack_direction)
        for trace_slice in worker_trace_slices()
    ]
    assert len(workers) == num_workers
    # Compute mean and standard deviation.
    mean, std_dev = compute_mean_and_std(workers)
    # Filter noisy traces.
    orig_num_traces = num_traces
    num_traces = filter_noisy_traces(workers, mean, std_dev, max_std)
    logging.info(f"Will use {num_traces} traces "
                 f"({100 * num_traces / orig_num_traces:.1f}% of all traces)")
    # Mean traces for all values of all text bytes.
    mean_text_traces = compute_mean_text_traces(workers)
    # Guess the differences between key bytes.
    pairwise_diffs_scores = compute_pairwise_diffs_and_scores(mean_text_traces)
    diffs = find_best_diffs(pairwise_diffs_scores)
    logging.info(f"Difference values (delta_0_i): {diffs}")
    # Recover the key.
    key = recover_key(diffs, attack_direction, project.get_plaintexts(0),
                      project.get_ciphertexts(0))
    if key is not None:
        logging.info(f"Recovered AES key: {bytes(key).hex()}")
    else:
        logging.error("Failed to recover the AES key")
    # Compare differences - both matrices are symmetric and have an all-zero main diagonal.
    correct_diffs = compare_diffs(pairwise_diffs_scores, attack_direction,
                                  project.get_keys(0))
    logging.info(
        f"Recovered {((np.sum(correct_diffs) - 16) / 2).astype(int)}/120 "
        "differences between key bytes")
    project.close(save=False)
    return key


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""A distributed implementation of the attack described in
        "Correlation-Enhanced Power Analysis Collision Attack" by A. Moradi, O.
        Mischke, and T. Eisenbarth (https://eprint.iacr.org/2010/297.pdf).""")
    parser.add_argument(
        "-f",
        "--project-file",
        required=True,
        help="chipwhisperer or ot_trace_library project file",
    )
    parser.add_argument(
        "-n",
        "--num-traces",
        type=int,
        required=True,
        help="""number of traces to use, must be less than or equal to the
        number of traces in PROJECT_FILE""",
    )
    parser.add_argument(
        "-a",
        "--attack-window",
        type=int,
        nargs=2,
        metavar=("FIRST_SAMPLE", "LAST_SAMPLE"),
        required=True,
        help="""attack window as a pair of sample indices, inclusive""",
    )
    parser.add_argument(
        "-d",
        "--attack-direction",
        type=AttackDirection,
        required=True,
        help="attack direction, input or output",
    )
    parser.add_argument(
        "-s",
        "--max-std",
        type=int,
        required=True,
        help="""allowed number of standard deviations from the mean trace for
        filtering noisy traces, must be greater than zero""",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        required=True,
        help="""number of workers to use for processing traces, must be greater
        than zero""",
    )
    return parser.parse_args()


def config_logger():
    """Configures the root logger."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s")
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


@timer()
def main():
    """Parses command-line arguments, configures logging, and performs the
    attack."""
    args = parse_args()
    config_logger()
    ray.init(
        runtime_env={
            "working_dir": "../",
            "excludes":
            ["*.db", "*.cwp", "*.npy", "*.bit", "*/lfs/*", "*.pack"],
        })

    key = perform_attack(**vars(args))
    sys.exit(0 if key is not None else 1)


if __name__ == "__main__":
    main()
