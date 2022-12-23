#!/usr/bin/env python3

"""
HI_size.py

Script that computes the HI size by fitting a Gaussian profile
to the face-on HI surface density map.
"""

import numpy as np
import swiftsimio as sw
import scipy.optimize as opt
import unyt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl
from velociraptor.observations import load_observations
import glob

from .plot import plot_data_on_axis

from typing import Dict, List, Tuple, Union
from numpy.typing import NDArray
from .logging import GalaxyLog
from swiftsimio import SWIFTDataset

# plot the surface density map and fit? Useful for debugging.
make_plots = False


def gauss_curve(
    x: NDArray, A: float, a: float, b: float, c: float, x0: float, y0: float
) -> NDArray:
    """
    General 2D elliptical Gaussian function, based on Rajohnson et al. (2022),
    equation (2).

    Parameters
    ----------
    x: (N,2) array containing positions (in kpc)

    A: central surface density (in Msun pc^-2)
    a, b, c: rotated major and minor axis of the ellipse (in kpc^-2)
    x0, y0: centre of the ellipse (in kpc)

    Returns
    -------
    (N,) array containing the predicted surface density for each position
    """
    return A * np.exp(
        -(
            a * (x[:, 0] - x0) ** 2
            + 2.0 * b * (x[:, 0] - x0) * (x[:, 1] - y0)
            + c * (x[:, 1] - y0) ** 2
        )
    )


def calculate_HI_size(
    galaxy_log: GalaxyLog,
    data: SWIFTDataset,
    face_on_rmatrix: NDArray[float],
    gas_mask: NDArray[bool],
    index: int,
    resolution: int = 128,
) -> Tuple[float, float]:
    """
    Calculate the HI size of the given galaxy by fitting a 2D elliptical function
    to the surface density profile and measuring the diameter along the major axis
    of the 1 Msun pc^-2 contour (see Rajohnson et al. (2022) section 4).
    The result is stored in the given HaloCatalogue (in units of kpc).

    Parameters
    ----------
    galaxy_log: GalaxyLog to write debug messages to
    data: SWIFTDataset containing the gas particles. Particle positions should already
          be recentred on the galaxy centre.
    face_on_rmatrix: rotation matrix used for face-on projections.
    index: halo index, used to index debug images.
    resolution: resolution of the image that is used to fit the elliptical profile
                the default value (128) provides a good mix of accuracy and speed.
    """

    R = sw.objects.cosmo_array(
        30.0 * unyt.kpc, comoving=False, cosmo_factor=data.gas.coordinates.cosmo_factor
    )

    image = sw.visualisation.project_gas(
        data=data,
        project="HI_mass",
        resolution=resolution,
        mask=gas_mask,
        rotation_center=unyt.unyt_array(
            [0.0, 0.0, 0.0], units=data.gas.coordinates.units
        ),
        rotation_matrix=face_on_rmatrix,
        region=[-R, R, -R, R, -R, R],
    )

    image.convert_to_units("Msun/pc**2")

    # Compute the HI mass as the integrated surface density of the image
    HImass = image.sum() * (R / resolution) ** 2

    if HImass == 0.0:
        galaxy_log.debug("Neutral gas mass is zero, not computing HI size.")
        return 0.0, 0.0

    # get the limits (for plotting)
    imax = np.nanmax(image)
    imin = max(1.0e-4 * imax, np.nanmin(image))

    # set up a grid of positions for the fit
    pixcenter = np.linspace(-R, R, resolution + 1)
    pixcenter = 0.5 * (pixcenter[1:] + pixcenter[:-1])
    xg, yg = np.meshgrid(pixcenter, pixcenter)
    xs = np.zeros((resolution ** 2, 2))
    xs[:, 0] = xg.flatten()
    xs[:, 1] = yg.flatten()
    # initial values for the fit (based on Rajohnson et al.)
    A = imax
    # the initial sig is set to 10 pixels
    # the pixel size in our case is 2*R / resolution
    sigX = 20.0 * R / resolution
    sigY = 20.0 * R / resolution
    a = 0.5 / sigX ** 2
    b = 0.0
    c = 0.5 / sigY ** 2
    # perform the fit
    try:
        params, _ = opt.curve_fit(
            gauss_curve, xs, image.flatten(), p0=(A, a, b, c, 0.0, 0.0)
        )
    except RuntimeError:
        galaxy_log.debug("Unable to fit HI profile. Will not compute HI size.")
        return 0.0, 0.0

    # extract the fitted parameters
    A, a, b, c, x0, y0 = params

    # the HI size is determined from the 1 Msun pc^-2 contour level. If the fitted
    # profile has a central surface density that is lower, the HI size is undefined
    if A <= 1.0:
        galaxy_log.debug(
            "Central surface density below 1 Msun pc^-2 limit, no HI size measurement possible!"
        )
        HIsize = 0.0
    else:
        # convert from the general elliptical coordinates to a coordinate frame where
        # the major axis is aligned with the x axis (and the minor axis with the y
        # axis). Determine the rotation angle theta of the general ellipse.
        # The relations below have been derived from Rajohnson et al., eq (3)-(5),
        # using some basic trigonometry.
        # Note that the derivation depends on the condition 1/sigY2 > 1/sigX2, so
        # that sigX2 is guaranteed to correspond to the major axis
        sigX2 = 1.0 / (a + c - np.sqrt((a - c) ** 2 + 4.0 * b ** 2))
        sigY2 = 1.0 / (np.sqrt((a - c) ** 2 + 4.0 * b ** 2) + a + c)
        theta = 0.5 * np.arctan2(-2.0 * b, c - a)
        Dx = np.sqrt(2.0 * sigX2 * np.log(A))
        Dy = np.sqrt(2.0 * sigY2 * np.log(A))

        if Dx < Dy:
            galaxy_log.debug("Error: major axis smaller than minor axis!")

        # Compute the HI size
        HIsize = 2.0 * Dx

    if make_plots:
        rcparams = {
            "font.size": 12,
            "font.family": "Times",
            "text.usetex": True,
            "figure.figsize": (5.5, 4),
            "figure.subplot.left": 0.05,
            "figure.subplot.right": 0.95,
            "figure.subplot.bottom": 0.15,
            "figure.subplot.top": 0.9,
            "figure.subplot.wspace": 0.3,
            "figure.subplot.hspace": 0.3,
            "lines.markersize": 2,
            "lines.linewidth": 1.0,
            "xtick.top": True,
            "ytick.right": True,
        }
        pl.rcParams.update(rcparams)

        # Create a figure to show how good the fit was
        fig, ax = pl.subplots(1, 1)

        # First, plot the surface density map
        levels = np.logspace(np.log10(imin), np.log10(imax), 100)
        cs = ax.contourf(
            xg,
            yg,
            image.data,
            norm=matplotlib.colors.LogNorm(vmin=imin, vmax=imax),
            levels=levels,
        )

        if A > 1.0:
            # Now overplot the 1 Msun pc^-2 contour, using the parametric equation for
            # the general ellipse
            tpar = np.linspace(0.0, 2.0 * np.pi, 1000)
            xpar = Dx * np.cos(theta) * np.cos(tpar) - Dy * np.sin(theta) * np.sin(tpar)
            ypar = Dx * np.sin(theta) * np.cos(tpar) + Dy * np.cos(theta) * np.sin(tpar)
            ax.plot(x0 + xpar, y0 + ypar, color="w", linestyle="--")

        # Overplot 9 contour levels of the Gaussian function
        image = gauss_curve(xs, *params).reshape((resolution, resolution))
        levels = np.logspace(np.log10(imin), np.log10(imax), 10)
        ax.contour(
            xg,
            yg,
            image,
            norm=matplotlib.colors.LogNorm(vmin=imin, vmax=imax),
            levels=levels,
            colors="w",
        )

        # Finalise and save the plot
        ax.set_aspect("equal")
        ax.set_xlabel("x (kpc)")
        ax.set_ylabel("y (kpc)")

        fig.colorbar(
            cs,
            label="Surface density (M$_\\odot{}$ pc$^{-2}$)",
            ticks=matplotlib.ticker.LogLocator(),
            format=matplotlib.ticker.LogFormatterMathtext(),
        )

        pl.tight_layout()
        pl.savefig(f"test_HI_image_{index:03d}.png", dpi=300)
        pl.close(fig)

    return HIsize, HImass


def plot_HI_size_mass(
    output_path: str,
    observational_data_path: str,
    name_list: List[str],
    all_galaxies_list: Union[List["AllGalaxyData"], List["GalaxyData"]],
) -> Dict:
    """
    Create HI size related plots.

    Parameters:
     - output_path: str
       Directory where images should be created.
     - observational_data_path: str
       Path to the observational data repository data (i.e. velociraptor-comparison-data/data).
     - name_list: List[str]
       Name labels for all the simulations that need to be plotted.
     - all_galaxies_list: Union[List[GalaxyData], List[AllGalaxyData]]
       Data for all the simulations that need to be plotted (can be a single galaxy for
       individual galaxy plots).

    Returns an image data dictionary compatible with MorphologyConfig.add_images().
    """

    fig, ax = pl.subplots(1, 1)

    sim_lines = []
    sim_labels = []
    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        HI_mass = unyt.unyt_array(data["HI_mass"], "Msun")
        HI_mass.name = "HI Mass"
        HI_size = unyt.unyt_array(data["HI_size"], "kpc")
        HI_size.name = "HI Size"

        line = plot_data_on_axis(
            ax, HI_mass, HI_size, color=f"C{i}", plot_scatter=(len(name_list) == 1)
        )
        sim_lines.append(line)
        sim_labels.append(name)

    observational_data = load_observations(
        sorted(glob.glob(f"{observational_data_path}/GalaxyHISizeMass/*.hdf5"))
    )
    with unyt.matplotlib_support:
        for obs_data in observational_data:
            obs_data.plot_on_axes(ax)

    sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
    ax.legend(loc="lower right")
    ax.add_artist(sim_legend)
    ax.set_xlim(1.0e6, 1.0e12)
    ax.set_ylim(1.0, 1.0e3)

    outputname = "HI_size_mass.png"
    pl.savefig(f"{output_path}/{outputname}", dpi=200)
    pl.close(fig)

    return {
        "HI sizes": {
            outputname: {
                "title": "HI size - HI mass",
                "caption": "HI size versus HI mass relation.",
            }
        }
    }
