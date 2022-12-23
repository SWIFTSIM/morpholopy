#!/usr/bin/env python3

"""
medians.py

Core functions to deal with 2D histogrammed medians.

Medians are represented by a dictionary that contains all
the necessary information about the plotting area, the
quantities involved and the binning strategy.
They are also represented by a numpy array of 2D histogram
number counts, and derived (approximate) median lines.

See the README for a more thorough explanation.
"""

import numpy as np
import unyt
import matplotlib

from .plot import plot_broken_line_on_axis

from typing import Dict, Tuple
from numpy.typing import NDArray


def accumulate_median_data(
    median: Dict, values_x: unyt.unyt_array, values_y: unyt.unyt_array
) -> NDArray[int]:
    """
    Bin the given x and y values into the 2D histogram for the given median.
    Returns the number counts in each bin.
    """

    # use copies of x and y to avoid changing the originals
    xval = values_x.copy()
    yval = values_y.copy()

    # convert to log space if required, mask out negative values
    # in this case
    if median["log x"]:
        mask = xval > 0.0
        xval = np.log10(xval[mask])
        yval = yval[mask]

    if median["log y"]:
        mask = yval > 0.0
        xval = xval[mask]
        yval = np.log10(yval[mask])

    nx = median["number of bins x"]
    ny = median["number of bins y"]
    counts = np.zeros(nx * ny, dtype=np.uint32)
    range_x = median["range in x"]
    range_y = median["range in y"]
    # values outside the x range need to be masked
    mask = (xval >= range_x[0]) & (xval <= range_x[1])
    xval = xval[mask]
    yval = yval[mask]
    # compute the grid indices of each pair of values
    index_x = np.clip(
        np.floor((xval - range_x[0]) / (range_x[1] - range_x[0]) * nx), 0, nx - 1
    )
    index_y = np.clip(
        np.floor((yval - range_y[0]) / (range_y[1] - range_y[0]) * ny), 0, ny - 1
    )
    # at this point, we know in which bin to store each of the values
    # however, we cannot simply access the bins, since multiple values
    # can end up in the same bin
    # so what we instead do is: we compute a combined index into a 1D array
    # and then find the unique index values and their counts
    # those can be used as an index array for the 1D array, and the counts
    # contain exactly the value we want
    combined_index = (index_x * ny + index_y).astype(np.uint32)
    idx, count = np.unique(combined_index, return_counts=True)
    counts[idx] = count
    return counts.reshape((nx, ny))


def compute_median(
    median: Dict, median_data: NDArray
) -> Tuple[NDArray[float], NDArray[float]]:
    """
    Compute an (approximate) median line from the given 2D histogram.

    Returns the x bin centres and the y median values.
    """

    nx = median["number of bins x"]
    ny = median["number of bins y"]
    range_x = median["range in x"]
    range_y = median["range in y"]

    # recover the median bins
    xbin_edges = np.linspace(range_x[0], range_x[1], nx + 1)
    ybin_edges = np.linspace(range_y[0], range_y[1], ny + 1)
    xbin_centres = 0.5 * (xbin_edges[:-1] + xbin_edges[1:])

    # convert the columns (fixed x, variable y) into cumulative
    # number counts
    ycum_counts = np.cumsum(median_data, axis=1)
    # get the target value in each column
    median_target = ycum_counts[:, -1] / 2
    # now find the edges that contain the median by counting values
    # below and above the target
    is_below_target = ycum_counts < median_target[:, None]
    is_above_target = ycum_counts > median_target[:, None]
    ymed_idx_m = np.clip(np.argmin(is_below_target, axis=1), 0, ny - 1)
    ymed_idx_p = np.clip(np.argmax(is_above_target, axis=1) + 1, 1, ny)

    # get the y edges and convert to log space if necessary
    ym = ybin_edges[ymed_idx_m]
    yp = ybin_edges[ymed_idx_p]
    if median["log y"]:
        ym = 10.0 ** ym
        yp = 10.0 ** yp
    # take the centre of the bin as the approximate value for the median
    ymedian = 0.5 * (ym + yp)

    if median["log x"]:
        xbin_centres = 10.0 ** xbin_centres

    # mask out empty bins; these do not have a median
    ymedian[median_target == 0] = np.nan
    return xbin_centres, ymedian


def plot_median_on_axis_as_line(
    ax: matplotlib.axis.Axis,
    median: Dict,
    color: str,
    linestyle: str = "-",
    marker: str = "o",
) -> matplotlib.lines.Line2D:
    """
    Plot the given median on the given axis as a (broken) line (see plot.py,
    plot_broken_line_on_axis()), using the given color, line style and marker.
    """

    x = unyt.unyt_array(median["x centers"], median["x units"])
    x.name = median.get("x label", "")
    y = unyt.unyt_array(median["y values"], median["y units"])
    y.name = median.get("y label", "")

    line = None
    with unyt.matplotlib_support:
        line = plot_broken_line_on_axis(
            ax, x, y, linestyle=linestyle, marker=marker, color=color
        )

    xlims = median["range in x"]
    if median["log x"]:
        ax.set_xscale("log")
        ax.set_xlim(10.0 ** xlims[0], 10.0 ** xlims[1])
    else:
        ax.set_xlim(*xlims)
        xscale = "linear"

    ylims = median["range in y"]
    if median["log y"]:
        ax.set_yscale("log")
        ax.set_ylim(10.0 ** ylims[0], 10.0 ** ylims[1])
    else:
        ax.set_ylim(*ylims)

    return line


def plot_median_on_axis_as_pdf(ax: matplotlib.axis.Axis, median: Dict):
    """
    Plot the given median on the given axis as an approximate PDF.

    We simply plot the 2D histogram used to compute the median.
    We use a transparency level of 50% to avoid cluttering the image
    too much and make sure the PDF is displayed in the background.
    """

    range_x = median["range in x"]
    range_y = median["range in y"]
    nx = median["number of bins x"]
    ny = median["number of bins y"]
    xbin_edges = np.linspace(range_x[0], range_x[1], nx + 1)
    ybin_edges = np.linspace(range_y[0], range_y[1], ny + 1)
    xscale = "linear"
    if median["log x"]:
        xbin_edges = 10.0 ** xbin_edges
        xscale = "log"
    yscale = "linear"
    if median["log y"]:
        ybin_edges = 10.0 ** ybin_edges
        yscale = "log"

    xbin_centres = 0.5 * (xbin_edges[1:] + xbin_edges[:-1])
    ybin_centres = 0.5 * (ybin_edges[1:] + ybin_edges[:-1])
    xv, yv = np.meshgrid(xbin_edges, ybin_edges)

    PDF = np.array(median["PDF"], dtype=np.float64)
    if PDF.sum() > 0:
        # mask out 0 bins, because pcolormesh does not correctly do that
        PDF[PDF == 0] = np.nan
        ax.pcolormesh(
            xv, yv, PDF.T, norm=matplotlib.colors.LogNorm(), alpha=0.5, zorder=-9000
        )

    if median["log x"]:
        ax.set_xscale("log")
        ax.set_xlim(10.0 ** range_x[0], 10.0 ** range_x[1])
    else:
        ax.set_xlim(*range_x)

    if median["log y"]:
        ax.set_yscale("log")
        ax.set_ylim(10.0 ** range_y[0], 10.0 ** range_y[1])
    else:
        ax.set_ylim(*range_y)


def test_median():
    """
    Unit test for the median functions.
    """

    import scipy.stats as stats

    seed = None
    do_plots = False

    if not seed is None:
        np.random.seed(seed)
    else:
        seed = np.random.randint(0xFFFFFFFF)
        np.random.seed(seed)
    print(f"random seed: {seed}")

    # normal test: linear-linear scale
    median = {
        "number of bins x": 50,
        "number of bins y": 100,
        "log x": False,
        "log y": False,
        "range in x": [0.0, 1.0],
        "range in y": [-0.1, 1.1],
    }

    xvals = np.random.random(1000)
    yvals = np.random.random(1000)

    median_data = accumulate_median_data(median, xvals, yvals)
    assert median_data.shape[0] == median["number of bins x"]
    assert median_data.shape[1] == median["number of bins y"]

    xmed, ymed = compute_median(median, median_data)
    assert xmed.shape[0] == median["number of bins x"]
    assert ymed.shape[0] == median["number of bins x"]

    refbins = np.linspace(
        median["range in x"][0], median["range in x"][1], median["number of bins x"] + 1
    )
    refmed, _, _ = stats.binned_statistic(
        xvals, yvals, statistic="median", bins=refbins
    )

    if do_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pl

        for edge in refbins:
            pl.gca().axvline(
                x=edge, color="k", alpha=0.5, linestyle="--", linewidth=0.5
            )
        pl.plot(xvals, yvals, ".")
        pl.plot(xmed, ymed, "o-")
        refbins = 0.5 * (refbins[1:] + refbins[:-1])
        pl.plot(refbins, refmed, "o--")
        pl.ylim(*median["range in y"])
        pl.xlim(*median["range in x"])
        pl.savefig("median_test_linlin.png", dpi=300)
        pl.close()

    assert len(ymed) == len(refmed)
    dy = (median["range in y"][1] - median["range in y"][0]) / median[
        "number of bins y"
    ]
    maxdev = 0.5 * dy
    for ym, rm in zip(ymed, refmed):
        if not (np.isnan(ym) or np.isnan(rm)):
            assert abs(ym - rm) < maxdev

    # variation 1: log-linear scale
    median = {
        "number of bins x": 50,
        "number of bins y": 100,
        "log x": True,
        "log y": False,
        "range in x": [-3.0, 0.0],
        "range in y": [-0.1, 1.1],
    }

    xvals = np.random.random(1000)
    yvals = np.random.random(1000)

    median_data = accumulate_median_data(median, xvals, yvals)

    xmed, ymed = compute_median(median, median_data)

    refbins = np.logspace(
        median["range in x"][0], median["range in x"][1], median["number of bins x"] + 1
    )
    refmed, _, _ = stats.binned_statistic(
        xvals, yvals, statistic="median", bins=refbins
    )

    if do_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pl

        for edge in refbins:
            pl.gca().axvline(
                x=edge, color="k", alpha=0.5, linestyle="--", linewidth=0.5
            )
        pl.semilogx(xvals, yvals, ".")
        pl.plot(xmed, ymed, "o-")
        refbins = 10.0 ** (0.5 * (np.log10(refbins[1:]) + np.log10(refbins[:-1])))
        pl.plot(refbins, refmed, "o--")
        pl.ylim(*median["range in y"])
        pl.xlim(10.0 ** median["range in x"][0], 10.0 ** median["range in x"][1])
        pl.savefig("median_test_loglin.png", dpi=300)
        pl.close()

    assert len(ymed) == len(refmed)
    dy = (median["range in y"][1] - median["range in y"][0]) / median[
        "number of bins y"
    ]
    maxdev = 0.5 * dy
    for ym, rm in zip(ymed, refmed):
        if not (np.isnan(ym) or np.isnan(rm)):
            assert abs(ym - rm) < maxdev

    # variation 2: linear-log scale
    median = {
        "number of bins x": 50,
        "number of bins y": 100,
        "log x": False,
        "log y": True,
        "range in x": [0.0, 1.0],
        "range in y": [-3.0, 0.0],
    }

    xvals = np.random.random(1000)
    yvals = np.random.random(1000)

    median_data = accumulate_median_data(median, xvals, yvals)

    xmed, ymed = compute_median(median, median_data)

    refbins = np.linspace(
        median["range in x"][0], median["range in x"][1], median["number of bins x"] + 1
    )
    refmed, _, _ = stats.binned_statistic(
        xvals, yvals, statistic="median", bins=refbins
    )

    if do_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pl

        for edge in refbins:
            pl.gca().axvline(
                x=edge, color="k", alpha=0.5, linestyle="--", linewidth=0.5
            )
        pl.semilogy(xvals, yvals, ".")
        pl.plot(xmed, ymed, "o-")
        refbins = 0.5 * (refbins[1:] + refbins[:-1])
        pl.plot(refbins, refmed, "o--")
        pl.ylim(10.0 ** median["range in y"][0], 10.0 ** median["range in y"][1])
        pl.xlim(*median["range in x"])
        pl.savefig("median_test_linlog.png", dpi=300)
        pl.close()

    assert len(ymed) == len(refmed)
    dy = (median["range in y"][1] - median["range in y"][0]) / median[
        "number of bins y"
    ]
    maxdev = dy
    for ym, rm in zip(ymed, refmed):
        if not (np.isnan(ym) or np.isnan(rm)):
            assert abs(np.log10(ym) - np.log10(rm)) < maxdev


if __name__ == "__main__":
    """
    Standalone mode: run the unit test.
    """
    test_median()
