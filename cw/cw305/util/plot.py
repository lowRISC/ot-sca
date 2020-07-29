import itertools

from bokeh.plotting import figure, show
from bokeh.palettes import Dark2_5 as palette
from bokeh.io import output_file


def save_plot_to_file(traces, outfile):
  """Save plot figure to file."""
  colors = itertools.cycle(palette)
  xrange = range(len(traces[0]))
  plot = figure(plot_width=800)
  for i in range(min(len(traces), 100)):
    plot.line(xrange, traces[i], line_color=next(colors))
  output_file(outfile)
  show(plot)
