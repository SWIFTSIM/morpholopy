import numpy as np


def accumulate_median_data(median, values_x, values_y):
    if median["log x"]:
        xval = np.log10(values_x)
    else:
        xval = values_x
    if median["log y"]:
        yval = np.log10(values_y)
    else:
        yval = values_y
    nx = median["number of bins x"]
    ny = median["number of bins y"]
    counts = np.zeros(nx * ny, dtype=np.uint32)
    range_x = median["range in x"]
    range_y = median["range in y"]
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


def compute_median(median, median_data):
    nx = median["number of bins x"]
    ny = median["number of bins y"]
    range_x = median["range in x"]
    range_y = median["range in y"]
    xbin_edges = np.linspace(range_x[0], range_x[1], nx + 1)
    xbin_centres = 0.5 * (xbin_edges[:-1] + xbin_edges[1:])
    ycum_counts = np.cumsum(median_data, axis=1)
    ymed_idx = np.clip(
        np.argmax(ycum_counts[:, :] > ycum_counts[:, -1, None] // 2, axis=1), 0, ny - 1
    )
    ybin_edges = np.linspace(range_y[0], range_y[1], ny + 1)
    ymedian = 0.5 * (ybin_edges[ymed_idx] + ybin_edges[ymed_idx + 1])
    if median["log x"]:
        xbin_centres = 10.0 ** xbin_centres
    if median["log y"]:
        ymedian = 10.0 ** ymedian
    ymedian[ycum_counts[:, -1] == 0] = np.nan
    return xbin_centres, ymedian


def test_median():

    median = {
        "number of bins x": 100,
        "number of bins y": 1000,
        "log x": True,
        "log y": True,
        "range in x": [-3.0, 0.0],
        "range in y": [-4.0, 0.0],
    }

    xvals = np.random.random(1000)
    yvals = xvals ** 2 + 0.1 * np.random.random(1000)

    median_data = accumulate_median_data(median, xvals, yvals)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pl
    import scipy.stats as stats

    xmed, ymed = compute_median(median, median_data)
    refbins = np.logspace(-3.0, 0.0, 11)
    refmed, _, _ = stats.binned_statistic(
        xvals, yvals, statistic="median", bins=refbins
    )
    refbins = 0.5 * (refbins[1:] + refbins[:-1])

    pl.loglog(xvals, yvals, ".")
    pl.plot(xmed, ymed, "-")
    pl.plot(refbins, refmed, "-")
    pl.savefig("median_test.png", dpi=300)


if __name__ == "__main__":
    test_median()
