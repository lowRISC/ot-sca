# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import itertools
from pathlib import Path

import numpy as np
from bokeh.io import output_file
from bokeh.models import tools
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import figure, show

from fault_injection.project_library.project import FISuccess


def save_plot_to_file(traces,
                      set_indices,
                      num_traces,
                      outfile,
                      add_mean_stddev=False,
                      ref_trace=None):
    """Save plot figure to file."""
    if set_indices is None:
        colors = itertools.cycle(palette)
    else:
        assert len(traces) == len(set_indices)
        # set_indices[x] indicates to which set trace x belongs.
        trace_colors = []
        for i in range(len(set_indices)):
            color_idx = set_indices[i] % len(palette)
            trace_colors.append(palette[color_idx])
        # Generate iterable from list.
        colors = itertools.cycle(tuple(trace_colors))
    xrange = range(len(traces[0]))
    plot = figure(plot_width=800)
    plot.add_tools(tools.CrosshairTool())
    plot.add_tools(tools.HoverTool())
    for i in range(min(len(traces), num_traces)):
        if set_indices is None:
            if add_mean_stddev or ref_trace is not None:
                plot.line(xrange, traces[i], line_color="grey")
            else:
                plot.line(xrange, traces[i], line_color=next(colors))
        else:
            plot.line(
                xrange,
                traces[i],
                line_color=next(colors),
                legend_label=str(set_indices[i]),
            )

    if ref_trace is not None:
        plot.line(xrange,
                  ref_trace,
                  line_color="firebrick",
                  line_width=2,
                  legend_label="mean")

    if add_mean_stddev:
        # Add mean and std dev to figure
        # Convert selected traces to np array before processing
        traces_new = np.empty((num_traces, len(traces[0])), dtype=np.uint16)
        for i_trace in range(num_traces):
            traces_new[i_trace] = traces[i_trace]
        mean = traces_new.mean(axis=0)
        std = traces_new.std(axis=0)
        mean_stddev_upper = mean + std
        mean_stddev_lower = mean - std
        plot.line(
            xrange,
            mean_stddev_upper,
            line_color="firebrick",
            line_width=2,
            legend_label="std",
        )
        plot.line(
            xrange,
            mean_stddev_lower,
            line_color="firebrick",
            line_width=2,
            legend_label="std",
        )
        plot.line(xrange,
                  mean,
                  line_color="black",
                  line_width=2,
                  legend_label="mean")

    output_file(Path(str(outfile) + ".html"))
    show(plot)


def save_fi_plot_to_file(cfg: dict, fi_results: [], outfile: str) -> None:
    """Print FI plot of traces.

    Printing the plot helps to narrow down the fault injection parameters.

    Args:
        cfg: The capture configuration.
        fi_results: The captured FI results.
        outfile: The path of the html file.
    """
    output_file(Path(str(outfile) + ".html"))
    x_axis = cfg["fiproject"]["plot_x_axis"]
    y_axis = cfg["fiproject"]["plot_y_axis"]
    plot = figure(
        plot_width=800,
        x_range=(cfg["fisetup"][x_axis + "_min"],
                 cfg["fisetup"][x_axis + "_max"]),
        y_range=(cfg["fisetup"][y_axis + "_min"],
                 cfg["fisetup"][y_axis + "_max"]),
    )
    plot.xaxis.axis_label = x_axis + " " + cfg["fiproject"][
        "plot_x_axis_legend"]
    plot.yaxis.axis_label = y_axis + " " + cfg["fiproject"][
        "plot_y_axis_legend"]

    exp_x = []
    exp_y = []
    unexp_x = []
    unexp_y = []
    no_x = []
    no_y = []
    for fi_result in fi_results:
        fi_result_dict = dataclasses.asdict(fi_result)
        if fi_result.fi_result == FISuccess.SUCCESS:
            unexp_x.append(fi_result_dict[x_axis])
            unexp_y.append(fi_result_dict[y_axis])
        elif fi_result.fi_result == FISuccess.EXPRESPONSE:
            exp_x.append(fi_result_dict[x_axis])
            exp_y.append(fi_result_dict[y_axis])
        else:
            no_x.append(fi_result_dict[x_axis])
            no_y.append(fi_result_dict[y_axis])

    if unexp_x:
        plot.scatter(
            unexp_x,
            unexp_y,
            line_color="green",
            fill_color="green",
            legend_label="Unexpected response",
        )
    if exp_x:
        plot.scatter(
            exp_x,
            exp_y,
            line_color="orange",
            fill_color="orange",
            legend_label="Expected response",
        )
    if no_x:
        plot.scatter(no_x,
                     no_y,
                     line_color="red",
                     fill_color="red",
                     legend_label="No response")

    show(plot)
