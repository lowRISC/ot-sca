#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import argparse
import binascii
from Crypto.Cipher import AES
import numpy as np
import time
from tqdm import tqdm
import yaml
from types import SimpleNamespace
import typer
from pathlib import Path

import chipwhisperer as cw

from util import device
from util import plot
from pyXKCP import pyxkcp

app = typer.Typer(add_completion=False)
# To be able to define subcommands for the "capture" command.
app_capture = typer.Typer()
app.add_typer(app_capture, name="capture", help="Capture traces for SCA")
# Shared options for "capture aes" and "capture sha3".
opt_num_traces = typer.Option(None, help="Number of traces to capture.")
opt_plot_traces = typer.Option(None, help="Number of traces to plot.")


# Note: initialize_capture and plot_results are also used by other scripts.
def initialize_capture(device_cfg, capture_cfg):
    """Initialize capture."""
    ot = device.OpenTitan(device_cfg["fpga_bitstream"],
                          device_cfg["fw_bin"],
                          device_cfg["pll_frequency"],
                          device_cfg["baudrate"],
                          capture_cfg["scope_gain"],
                          capture_cfg["num_samples"],
                          capture_cfg["output_len_bytes"])
    print(f'Scope setup with sampling rate {ot.scope.clock.adc_freq} S/s')
    # Ping target
    print('Reading from FPGA using simpleserial protocol.')
    version = None
    ping_cnt = 0
    while not version:
        if ping_cnt == 3:
            raise RuntimeError(
                f'No response from the target (attempts: {ping_cnt}).')
        ot.target.write('v' + '\n')
        ping_cnt += 1
        time.sleep(0.5)
        version = ot.target.read().strip()
    print(f'Target simpleserial version: {version} (attempts: {ping_cnt}).')
    return ot


def plot_results(plot_cfg, project_name):
    """Plots traces from `project_name` using `plot_cfg` settings."""
    project = cw.open_project(project_name)

    if len(project.waves) == 0:
        print('Project contains no traces. Did the capture fail?')
        return

    # The ADC output is in the interval [-0.5, 0.5). Check that the recorded
    # traces are within that range with some safety margin.
    if not (np.all(np.greater(project.waves, -plot_cfg["amplitude_max"]))
            and np.all(np.less(project.waves, plot_cfg["amplitude_max"]))):
        print('WARNING: Some traces have samples outside the range (' +
              str(-plot_cfg["amplitude_max"]) + ', ' +
              str(plot_cfg["amplitude_max"]) + ').')
        print('The ADC has a max range of [-0.5, 0.5) and might saturate.')
        print('It is recommended to reduce the scope gain (see device.py).')

    plot.save_plot_to_file(project.waves, plot_cfg["num_traces"],
                           plot_cfg["trace_image_filename"])
    print(
        f'Created plot with {plot_cfg["num_traces"]} traces: {Path(plot_cfg["trace_image_filename"]).resolve()}'
    )


@app.command()
def init(ctx: typer.Context):
    """Initalize target for SCA."""
    initialize_capture(ctx.obj.cfg["device"], ctx.obj.cfg["capture"])


def capture_init(ctx, num_traces, plot_traces):
    """Initializes the user data stored in the context and programs the target."""
    cfg = ctx.obj.cfg
    if num_traces:
        cfg["capture"]["num_traces"] = num_traces

    if plot_traces:
        cfg["plot_capture"]["show"] = True
        cfg["plot_capture"]["num_traces"] = plot_traces

    # Key and plaintext generator
    ctx.obj.ktp = cw.ktp.Basic()
    ctx.obj.ktp.key_len = cfg["capture"]["key_len_bytes"]
    ctx.obj.ktp.text_len = cfg["capture"]["plain_text_len_bytes"]

    ctx.obj.ot = initialize_capture(cfg["device"], cfg["capture"])


def capture_loop(trace_gen, capture_cfg):
    """Main capture loop.

    Args:
      trace_gen: A trace generator.
      capture_cfg: Capture configuration.
    """
    project = cw.create_project(capture_cfg["project_name"], overwrite=True)
    for _ in tqdm(range(capture_cfg["num_traces"]), desc='Capturing', ncols=80):
        project.traces.append(next(trace_gen))
    project.save()


def capture_end(cfg):
    if cfg["plot_capture"]["show"]:
        plot_results(cfg["plot_capture"], cfg["capture"]["project_name"])


def capture_aes(ot, ktp):
    """A generator for capturing AES traces.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
    """
    key, _ = ktp.next()
    tqdm.write(f'Using key: {binascii.b2a_hex(bytes(key))}')
    cipher = AES.new(bytes(key), AES.MODE_ECB)
    while True:
        _, text = ktp.next()
        ret = cw.capture_trace(ot.scope, ot.target, text, key, ack=False)
        if not ret:
            raise RuntimeError('Capture failed.')
        expected = binascii.b2a_hex(cipher.encrypt(bytes(text)))
        got = binascii.b2a_hex(ret.textout)
        if got != expected:
            raise RuntimeError(f'Bad ciphertext: {got} != {expected}.')
        yield ret


@app_capture.command()
def aes(ctx: typer.Context,
        num_traces: int = opt_num_traces,
        plot_traces: int = opt_plot_traces):
    """Capture AES traces from a target that runs the `aes_serial` program."""
    capture_init(ctx, num_traces, plot_traces)
    capture_loop(capture_aes(ctx.obj.ot, ctx.obj.ktp), ctx.obj.cfg["capture"])
    capture_end(ctx.obj.cfg)


def capture_kmac(ot, ktp):
    """A generator for capturing KMAC traces.

    Args:
      ot: Initialized OpenTitan target.
      ktp: Key and plaintext generator.
    """
    key, _ = ktp.next()
    tqdm.write(f'Using key: {binascii.b2a_hex(bytes(key))}')
    ot.target.simpleserial_write('k', key)
    while True:
        _, text = ktp.next()
        ret = cw.capture_trace(ot.scope, ot.target, text, key, ack=False)
        if not ret:
            raise RuntimeError('Capture failed.')
        expected = binascii.b2a_hex(pyxkcp.kmac128(key, ktp.key_len,
                                                   text, ktp.text_len,
                                                   ot.target.output_len,
                                                   b'\x00', 0))
        got = binascii.b2a_hex(ret.textout)
        if got != expected:
            raise RuntimeError(f'Bad digest: {got} != {expected}.')
        yield ret


@app_capture.command()
def sha3(ctx: typer.Context,
         num_traces: int = opt_num_traces,
         plot_traces: int = opt_plot_traces):
    """Capture KMAC traces from a target that runs the `sha3_serial` program."""
    capture_init(ctx, num_traces, plot_traces)
    capture_loop(capture_kmac(ctx.obj.ot, ctx.obj.ktp), ctx.obj.cfg["capture"])
    capture_end(ctx.obj.cfg)


@app.command("plot")
def plot_cmd(ctx: typer.Context, num_traces: int = opt_plot_traces):
    """Plots previously captured traces."""

    if num_traces is not None:
        ctx.obj.cfg["plot_capture"]["num_traces"] = num_traces
    plot_results(ctx.obj.cfg["plot_capture"], ctx.obj.cfg["capture"]["project_name"])


@app.callback()
def main(ctx: typer.Context, cfg_file: str = None):
    """Capture traces for side-channel analysis."""

    cfg_file = 'capture_aes.yaml' if cfg_file is None else cfg_file
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Store config in the user data attribute (`obj`) of the context.
    ctx.obj = SimpleNamespace(cfg=cfg)


if __name__ == "__main__":
    app()
