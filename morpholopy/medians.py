import numpy as np


def accumulate_median_data(median, values_x, values_y):

    xval = values_x.copy()
    yval = values_y.copy()

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
    ybin_edges = np.linspace(range_y[0], range_y[1], ny + 1)
    xbin_centres = 0.5 * (xbin_edges[:-1] + xbin_edges[1:])

    ycum_counts = np.cumsum(median_data, axis=1)
    median_target = ycum_counts[:, -1] / 2
    is_below_target = ycum_counts < median_target[:, None]
    is_above_target = ycum_counts > median_target[:, None]
    ymed_idx_m = np.clip(np.argmin(is_below_target, axis=1), 0, ny - 1)
    ymed_idx_p = np.clip(np.argmax(is_above_target, axis=1) + 1, 1, ny)

    ym = ybin_edges[ymed_idx_m]
    yp = ybin_edges[ymed_idx_p]
    if median["log y"]:
        ym = 10.0 ** ym
        yp = 10.0 ** yp
    ymedian = 0.5 * (ym + yp)

    if median["log x"]:
        xbin_centres = 10.0 ** xbin_centres

    ymedian[median_target == 0] = np.nan
    return xbin_centres, ymedian


def test_median():

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
    test_median()
