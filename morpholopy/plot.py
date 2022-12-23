import numpy as np
import unyt
import scipy.stats as stats


def plot_broken_line_on_axis(ax, x, y, color, linestyle="-", marker="o"):
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
    ax,
    x,
    y,
    color,
    plot_scatter=False,
    log_x=True,
    log_y=True,
    linestyle="-",
    marker="o",
    nbin=20,
):

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
            ax.plot(x, y, ".", color=color, alpha=1.0 / np.log10(x.shape[0]))

        line = plot_broken_line_on_axis(
            ax, xbin_centres, median, color=color, linestyle=linestyle, marker=marker
        )

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")

    return line
