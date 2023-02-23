# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import itertools

from bokeh.io import output_file
from bokeh.models import tools
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import figure, show


def save_plot_to_file(traces, set_indices, num_traces, outfile):
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
            plot.line(xrange, traces[i], line_color=next(colors))
        else:
            plot.line(xrange, traces[i], line_color=next(colors), legend_label=str(set_indices[i]))
    output_file(outfile)
    show(plot)
