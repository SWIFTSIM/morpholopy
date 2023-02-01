#!/usr/bin/env python3

"""
plot.py

Convenience methods that can be used to uniformly plot data in other
parts of the pipeline.
"""

import numpy as np
import unyt
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl


def plot_broken_line_on_axis(
    ax: matplotlib.axis.Axis,
    x: unyt.unyt_array,
    y: unyt.unyt_array,
    color: str,
    linestyle: str = "-",
    marker: str = "o",
) -> matplotlib.lines.Line2D:
    """
    Plot a (broken) line on the given axis.

    This plots y versus x as a normal line plot, using the given color and line style.
    However, if some values in x or y are NaN, then some parts of this line might not
    show up: only points that are not NaN and are connected to at least one other point
    that is not NaN will yield a valid line segment that is visible on the plot.
    To avoid missing out non NaN values that happen to not be part of any valid segment,
    we filter out these unfortunate points and add them as individual points, using the
    given color and marker.

    We return a matplotlib.lines.Line2D object that pretends we actually used
      pl.plot(x, y, color=color, linestyle=linestyle, marker=marker)
    but then without having to actually plot markers for parts of the line that
    do show up anyway.
    This Line2D object can be used to add the line to a legend.
    """

    isnan = np.isnan(x) | np.isnan(y)
    part_of_line = np.zeros(x.shape, dtype=bool)
    part_of_line[0] = not (isnan[0] or isnan[1])
    part_of_line[1:-1] = (~(isnan[1:-1] | isnan[2:])) | (~(isnan[:-2] | isnan[1:-1]))
    part_of_line[-1] = not (isnan[-1] or isnan[-2])

    ax.plot(x, y, color=color, linestyle=linestyle, marker=None)
    ax.plot(
        x[~part_of_line], y[~part_of_line], color=color, linestyle="None", marker=marker
    )

    line = ax.plot([], [], color=color, linestyle=linestyle, marker=marker)[0]
    return line


def plot_data_on_axis(
    ax: matplotlib.axis.Axis,
    x: unyt.unyt_array,
    y: unyt.unyt_array,
    color: str,
    plot_scatter: bool = False,
    log_x: bool = True,
    log_y: bool = True,
    linestyle: str = "-",
    marker: str = "o",
    nbin: int = 10,
):
    """
    Plot the given x and y values on the given axis.

    By default, we plot the median line for the given values,
    but optionally a normal scatter plot can be added as well.
    Median lines are plotted using plot_broken_line_on_axis,
    using the given color, line style and marker. Optional
    scatter plots use the same color, but a smaller marker and
    a transparency level that scales with the number of points.

    All plot calls are wrapper in a unyt.matplotlib_support block,
    so units and names attached to the x and y arrays will be used
    for labels.

    The log_x and log_y parameters are not only used to set the
    scaling of the axes, but also to use an appropriate binning
    strategy for the median line.
    """

    xmin = None
    if log_x:
        xmin = np.nanmin(x[x > 0.0])
    else:
        xmin = np.nanmin(x)
    xmax = np.nanmax(x)

    if log_x:
        xmin = np.log10(xmin)
        xmax = np.log10(xmax)

    dx = xmax - xmin
    if dx == 0.0:
        dx = 0.1 * xmax

    xmin -= 0.01 * dx
    xmax += 0.01 * dx

    xbin_edges = np.linspace(xmin, xmax, nbin + 1)
    xbin_centres = 0.5 * (xbin_edges[1:] + xbin_edges[:-1])

    if log_x:
        xbin_edges = 10.0 ** xbin_edges
        xbin_centres = 10.0 ** xbin_centres

    mask = (~np.isnan(x)) & (~np.isnan(y))
    median, _, _ = stats.binned_statistic(
        x[mask], y[mask], statistic="median", bins=xbin_edges
    )

    xbin_centres = unyt.unyt_array(xbin_centres, units=x.units)
    xbin_centres.name = x.name
    median = unyt.unyt_array(median, units=y.units)
    median.name = y.name

    line = None
    with unyt.matplotlib_support:
        if plot_scatter:
            # scale the alpha with the number of points
            ax.plot(
                x,
                y,
                ".",
                color=color,
                alpha=np.minimum(1.0 / np.log10(x.shape[0]), 1.0),
            )

        line = plot_broken_line_on_axis(
            ax, xbin_centres, median, color=color, linestyle=linestyle, marker=marker
        )

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")

    return line
