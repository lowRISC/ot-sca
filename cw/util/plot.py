# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import itertools

from bokeh.io import output_file
from bokeh.models import tools
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import figure, show


def save_plot_to_file(traces, num_traces, outfile):
    """Save plot figure to file."""
    colors = itertools.cycle(palette)
    xrange = range(len(traces[0]))
    plot = figure(plot_width=800)
    plot.add_tools(tools.CrosshairTool())
    plot.add_tools(tools.HoverTool())
    for i in range(min(len(traces), num_traces)):
        plot.line(xrange, traces[i], line_color=next(colors))
    output_file(outfile)
    show(plot)
