#!/usr/bin/env python3

"""
KS.py

Kennicutt-Schmidt like quantities and plots.

Mostly copied over from the old morphology pipeline.
"""

import numpy as np
import swiftsimio as sw
import unyt
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl
from velociraptor.observations import load_observations
import glob

from .medians import plot_median_on_axis_as_line, plot_median_on_axis_as_pdf

from typing import Dict, List, Tuple, Union
from numpy.typing import NDArray
from swiftsimio import SWIFTDataset

# solar metallicity value assumed to compute metallicities
Zsolar = 0.0134

# markers used to distinguish different simulations in integrated plots
markers = ["o", "s", "^", "D", "v"]
# labels and line styles for different metallicity cuts
metallicities = [
    ("all", "all", "-", "o"),
    ("Zm1", "$\\log_{10} Z/Z_\\odot = -1$", "--", "s"),
    ("Z0", "$\\log_{10} Z/Z_\\odot = 0$", ":", "^"),
    ("Zp1", "$\\log_{10} Z/Z_\\odot = 1$", (0, (3, 2, 1, 2)), "D"),
]

# debugging: plot individual surface density images
make_plots = False


def calculate_surface_densities_grid(
    data: SWIFTDataset,
    face_on_rmatrix: NDArray[float],
    gas_mask: NDArray[bool],
    stars_mask: NDArray[bool],
    index: int,
    resolution: int = 128,
) -> Dict:
    """
    Generate a set of face-on surface density projections that is needed for the
    various KS quantities.

    Parameters:
     - data: SWIFTDataset
       Galaxy data, already recentred on the galaxy centre.
     - face_on_rmatrix: NDArray[float]
       Face-on rotation matrix.
     - gas_mask: NDArray[bool]
       30 kpc aperture mask for gas particles.
     - stars_mask: NDArray[bool]
       30 kpc aperture mask for star particles.
     - index: int
       Unique index for this galaxy, used to guarantee unique names for debugging
       images.
     - resolution: int
       Pixel resolution of the maps. Note that the physical size of the maps is
       fixed to a 60 kpc square centred on the galaxy (with 60 kpc of depth in the
       projection direction).

    Returns a dictionary with all the maps.
    """

    R = sw.objects.cosmo_array(
        30.0 * unyt.kpc, comoving=False, cosmo_factor=data.gas.coordinates.cosmo_factor
    )

    images = {}

    for q in [
        "HI_mass",
        "H2_mass",
        "H_neutral_mass",
        "star_formation_rates",
        "metal_mass",
        "masses",
    ]:
        images[q] = sw.visualisation.project_gas(
            data=data,
            project=q,
            resolution=resolution,
            mask=gas_mask,
            rotation_center=unyt.unyt_array(
                [0.0, 0.0, 0.0], units=data.gas.coordinates.units
            ),
            rotation_matrix=face_on_rmatrix,
            region=[-R, R, -R, R, -R, R],
        )

    stars_pixel_grid = sw.visualisation.projection.project_pixel_grid(
        data.stars,
        project="masses",
        resolution=resolution,
        mask=stars_mask,
        rotation_center=unyt.unyt_array(
            [0.0, 0.0, 0.0], units=data.gas.coordinates.units
        ),
        rotation_matrix=face_on_rmatrix,
        region=[-R, R, -R, R, -R, R],
        boxsize=data.metadata.boxsize,
    )
    units = 1.0 / (4.0 * R ** 2)
    units.convert_to_units(1.0 / R.units ** 2)
    units *= data.stars.masses.units
    images["star_mass"] = unyt.unyt_array(stars_pixel_grid, units=units)
    images["star_mass"].convert_to_units("Msun/pc**2")

    images["metallicity"] = unyt.unyt_array(
        np.zeros_like(images["masses"]), "dimensionless"
    )
    mask = images["masses"] > 0.0
    images["metallicity"][mask] = (
        images["metal_mass"][mask] / images["masses"][mask] / Zsolar
    )

    for q in ["HI_mass", "H2_mass", "H_neutral_mass"]:
        images[f"tgas_{q}"] = unyt.unyt_array(np.zeros_like(images[q]), "yr")
        mask = images["star_formation_rates"] > 0.0
        images[f"tgas_{q}"][mask] = (
            images[q][mask] / images["star_formation_rates"][mask]
        ).to("yr")
        images[q].convert_to_units("Msun/pc**2")

    images["star_formation_rates"].convert_to_units("Msun/yr/kpc**2")

    if make_plots:
        for q in images:
            pl.imshow(
                images[q], norm=matplotlib.colors.LogNorm(vmin=1.0e-6, vmax=1.0e4)
            )
            pl.savefig(f"test_surfdens_{q}_{index:03d}.png", dpi=300)
            pl.close()
        pl.loglog(images["H_neutral_mass"], images["star_formation_rates"], ".")
        pl.xlim(1.0e-1, 1.0e4)
        pl.ylim(1.0e-6, 10.0)
        pl.savefig(f"test_sigma_neutral_SFR_{index:03d}.png", dpi=300)
        pl.close()
        pl.loglog(images["H2_mass"], images["star_formation_rates"], ".")
        pl.xlim(1.0e-1, 1.0e4)
        pl.ylim(1.0e-6, 10.0)
        pl.savefig(f"test_sigma_H2_SFR_{index:03d}.png", dpi=300)
        pl.close()

    return images


def calculate_spatially_resolved_KS(
    data: SWIFTDataset,
    face_on_rmatrix: NDArray[float],
    gas_mask: NDArray[bool],
    stars_mask: NDArray[bool],
    index: int,
) -> Tuple[NDArray]:
    """
    Compute the KS surface density pixel values.
    Computes an appropriate image resolution and then delegates the
    actual computation to calculate_surface_densities_grid().

    Parameters:
     - data: SWIFTDataset
       Galaxy data, already recentred on the galaxy centre.
     - face_on_rmatrix: NDArray[float]
       Face-on rotation matrix.
     - gas_mask: NDArray[bool]
       30 kpc aperture mask for gas particles.
     - stars_mask: NDArray[bool]
       30 kpc aperture mask for star particles.
     - index: int
       Unique index for this galaxy, used to guarantee unique names for debugging
       images.

    Returns a tuple with all the pixel values that are actually of interest.
    """
    image_diameter = 60.0 * unyt.kpc
    pixel_size = 0.75 * unyt.kpc
    resolution = int((image_diameter / pixel_size).value) + 1
    images = calculate_surface_densities_grid(
        data, face_on_rmatrix, gas_mask, stars_mask, index, resolution
    )

    return (
        images["H_neutral_mass"].value.flatten(),
        images["HI_mass"].value.flatten(),
        images["H2_mass"].value.flatten(),
        images["star_formation_rates"].value.flatten(),
        images["tgas_H_neutral_mass"].value.flatten(),
        images["tgas_HI_mass"].value.flatten(),
        images["tgas_H2_mass"].value.flatten(),
        images["metallicity"].value.flatten(),
        images["star_mass"].value.flatten(),
    )


def calculate_azimuthally_averaged_KS(
    data: SWIFTDataset,
    face_on_rmatrix: NDArray[float],
    gas_mask: NDArray[float],
    index: int,
) -> Tuple[Union[NDArray, None]]:
    """
    Compute azimuthally averaged surface densities relevant for KS plots.

    Parameters:
     - data: SWIFTDataset
       Galaxy data, already recentred on the galaxy centre.
     - face_on_rmatrix: NDArray[float]
       Face-on rotation matrix.
     - gas_mask: NDArray[bool]
       30 kpc aperture mask for gas particles.
     - index: int
       Unique index for this galaxy, used to guarantee unique names for debugging
       images.

    Returns a tuple with all the azimuthal bin values that are actually of interest, or
    None if there is no gas.
    """
    if gas_mask.sum() == 0:
        return None, None, None, None, None

    bin_size = 0.75 * unyt.kpc
    x, y, _ = np.matmul(face_on_rmatrix, data.gas.coordinates[gas_mask].T)
    r = np.sqrt(x ** 2 + y ** 2).to("kpc")
    rbins = np.arange(0.0 * unyt.kpc, 30.0 * unyt.kpc, bin_size)
    mhist = {}
    for q, unit, output_unit in [
        ("HI_mass", "Msun", "Msun/pc**2"),
        ("H2_mass", "Msun", "Msun/pc**2"),
        ("H_neutral_mass", "Msun", "Msun/pc**2"),
        ("star_formation_rates", "Msun/yr", "Msun/yr/kpc**2"),
        ("masses", "Msun", "Msun/pc**2"),
        ("metal_mass", "Msun", "Msun/pc**2"),
    ]:
        qdata = getattr(data.gas, q)[gas_mask].to(unit)
        qbin, _, _ = stats.binned_statistic(r, qdata, statistic="sum", bins=rbins)
        mhist[q] = unyt.unyt_array(
            qbin / (np.pi * (rbins[1:] ** 2 - rbins[:-1] ** 2)), units=f"{unit}/kpc**2"
        )
        mhist[q].convert_to_units(output_unit)

        if make_plots:
            img = np.zeros((400, 400))
            xr = np.linspace(-30.0, 30.0, 400)
            xg, yg = np.meshgrid(xr, xr)
            rg = np.sqrt(xg ** 2 + yg ** 2)
            ig = np.clip(np.floor(rg / bin_size.value), 0, rbins.shape[0] - 2).astype(
                np.uint32
            )
            img = mhist[q][ig]
            pl.imshow(img, norm=matplotlib.colors.LogNorm(vmin=1.0e-6, vmax=1.0e4))
            pl.savefig(f"test_azimuthal_{q}_{index:03d}.png", dpi=300)
            pl.close()

    mhist["metallicity"] = unyt.unyt_array(
        np.zeros_like(mhist["masses"]), "dimensionless"
    )
    mask = mhist["masses"] > 0.0
    mhist["metallicity"][mask] = (
        mhist["metal_mass"][mask] / mhist["masses"][mask] / Zsolar
    )

    if make_plots:
        pl.loglog(mhist["H_neutral_mass"], mhist["star_formation_rates"], ".")
        pl.xlim(1.0e-1, 1.0e4)
        pl.ylim(1.0e-6, 10.0)
        pl.savefig(f"test_azimuth_gas_SFR_{index:03d}.png", dpi=300)
        pl.close()
        pl.loglog(mhist["H2_mass"], mhist["star_formation_rates"], ".")
        pl.xlim(1.0e-1, 1.0e4)
        pl.ylim(1.0e-6, 10.0)
        pl.savefig(f"test_azimuth_H2_SFR_{index:03d}.png", dpi=300)
        pl.close()

    return (
        mhist["H_neutral_mass"],
        mhist["HI_mass"],
        mhist["H2_mass"],
        mhist["star_formation_rates"],
        mhist["metallicity"],
    )


def calculate_integrated_surface_densities(
    data: SWIFTDataset,
    face_on_rmatrix: NDArray[float],
    gas_mask: NDArray[bool],
    radius: unyt.unyt_quantity,
) -> Tuple[Union[float, None]]:
    """
    Compute integrated surface densities relevant for KS plots.

    Parameters:
     - data: SWIFTDataset
       Galaxy data, already recentred on the galaxy centre.
     - face_on_rmatrix: NDArray[float]
       Face-on rotation matrix.
     - gas_mask: NDArray[bool]
       30 kpc aperture mask for gas particles.
     - radius: unyt.unyt_quantity
       Radius of the integration circle.

    Returns a tuple with all the integrated values that are actually of interest, or
    None if the integration radius is 0.
    """

    if radius == 0.0:
        return 0.0, 0.0, 0.0, 0.0

    surface = np.pi * radius ** 2

    x, y, _ = np.matmul(face_on_rmatrix, data.gas.coordinates[gas_mask].T)
    r = np.sqrt(x ** 2 + y ** 2).to("kpc").value
    select = gas_mask.copy()
    select[gas_mask] = r < radius.to("kpc").value

    Sigma_HI = data.gas.HI_mass[select].sum() / surface
    Sigma_H2 = data.gas.H2_mass[select].sum() / surface
    Sigma_neutral = data.gas.H_neutral_mass[select].sum() / surface
    select &= data.gas.star_formation_rates > 0.0
    Sigma_SFR = data.gas.star_formation_rates[select].sum() / surface

    return (
        Sigma_HI.to("Msun/pc**2"),
        Sigma_H2.to("Msun/pc**2"),
        Sigma_neutral.to("Msun/pc**2"),
        Sigma_SFR.to("Msun/yr/kpc**2"),
    )


def plot_KS_relations(
    output_path: str,
    observational_data_path: str,
    name_list: List[str],
    all_galaxies_list: Union[List["AllGalaxyData"], List["GalaxyData"]],
    prefix: str = "",
    always_plot_scatter: bool = False,
    plot_integrated_quantities: bool = True,
) -> Dict:
    """
    Create KS related plots.

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
     - prefix: str
       Unique prefix that is prepended to images file names. Useful to distinguish
       individual galaxy images.
     - always_plot_scatter: bool
       Wheter or not to always plot individual median histrogram bins as well as median lines.
     - plot_integrated_quantities: bool
       Whether or not to plot integrated KS plots. Set to False for individual galaxy plots,
       since a galaxy is only a single point on those plots.

    Returns an image data dictionary compatible with MorphologyConfig.add_images().
    """

    plots = {}

    if plot_integrated_quantities:
        fig_neut, ax_neut = pl.subplots(1, 1)
        fig_at, ax_at = pl.subplots(1, 1)
        fig_mol, ax_mol = pl.subplots(1, 1)

        if len(name_list) > len(markers):
            raise RuntimeError(
                f"Not enough markers to plot {len(name_list)} different simulations!"
            )
        cs_neut = None
        cs_at = None
        cs_mol = None
        for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
            marker = markers[i]
            Mstar = unyt.unyt_array(data["stellar_mass"], unyt.Msun).in_base("galactic")
            Mstar.name = f"Stellar Mass"
            sigma_neutral = unyt.unyt_array(data["sigma_neutral"], "Msun/pc**2")
            sigma_neutral.name = "Neutral Gas Surface Density"
            sigma_HI = unyt.unyt_array(data["sigma_HI"], "Msun/pc**2")
            sigma_HI.name = "Atomic Gas Surface Density"
            sigma_H2 = unyt.unyt_array(data["sigma_H2"], "Msun/pc**2")
            sigma_H2.name = "Molecular Gas Surface Density"
            sigma_SFR = unyt.unyt_array(data["sigma_SFR"], "Msun/yr/kpc**2")
            sigma_SFR.name = "Star Formation Rate Surface Density"

            with unyt.matplotlib_support:
                cs_neut = ax_neut.scatter(
                    sigma_neutral,
                    sigma_SFR,
                    c=Mstar,
                    norm=matplotlib.colors.LogNorm(vmin=1.0e6, vmax=1.0e12),
                    marker=marker,
                )
                cs_at = ax_at.scatter(
                    sigma_HI,
                    sigma_SFR,
                    c=Mstar,
                    norm=matplotlib.colors.LogNorm(vmin=1.0e6, vmax=1.0e12),
                    marker=marker,
                )
                cs_mol = ax_mol.scatter(
                    sigma_H2,
                    sigma_SFR,
                    c=Mstar,
                    norm=matplotlib.colors.LogNorm(vmin=1.0e6, vmax=1.0e12),
                    marker=marker,
                )

        fig_neut.colorbar(cs_neut, label=f"{Mstar.name} [${Mstar.units.latex_repr}$]")
        fig_at.colorbar(cs_at, label=f"{Mstar.name} [${Mstar.units.latex_repr}$]")
        fig_mol.colorbar(cs_mol, label=f"{Mstar.name} [${Mstar.units.latex_repr}$]")

        for dataname, ax in zip(
            [
                "IntegratedNeutralKSRelation",
                "IntegratedAtomicKSRelation",
                "IntegratedMolecularKSRelation",
            ],
            [ax_neut, ax_at, ax_mol],
        ):

            sim_lines = []
            sim_labels = []
            for i, name in enumerate(name_list):
                marker = markers[i]
                sim_lines.append(
                    ax.plot([], [], marker=marker, color="k", linestyle="None")[0]
                )
                sim_labels.append(name)

            if dataname is not None:
                observational_data = load_observations(
                    sorted(glob.glob(f"{observational_data_path}/{dataname}/*.hdf5"))
                )
                with unyt.matplotlib_support:
                    for obs_data in observational_data:
                        obs_data.plot_on_axes(ax)

            ax.grid(True)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1.0e-1, 1.0e4)
            ax.set_ylim(1.0e-6, 1.0e1)
            ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
            sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
            ax.legend(loc="lower right")
            ax.add_artist(sim_legend)

        neut_filename = f"{prefix}integrated_KS_neutral.png"
        fig_neut.savefig(f"{output_path}/{neut_filename}", dpi=300)
        pl.close(fig_neut)

        at_filename = f"{prefix}integrated_KS_atomic.png"
        fig_at.savefig(f"{output_path}/{at_filename}", dpi=300)
        pl.close(fig_at)

        mol_filename = f"{prefix}integrated_KS_molecular.png"
        fig_mol.savefig(f"{output_path}/{mol_filename}", dpi=300)
        pl.close(fig_mol)

        plots["Integrated surface densities"] = {
            neut_filename: {
                "title": "Integrated surface densities / Neutral Gas",
                "caption": (
                    "Integrated surface densities of H2+HI gas and star-forming gas"
                    " for each individual galaxy. Quantities are calculated summing"
                    " up all gas (and SFR) within the galaxies' stellar half mass radius."
                ),
            },
            at_filename: {
                "title": "Integrated surface densities / Atomic Gas",
                "caption": (
                    "Integrated surface densities of HI gas and star-forming gas for"
                    " each individual galaxy. Quantities are calculated summing up"
                    " all gas (and SFR) within the galaxies' stellar half mass radius."
                ),
            },
            mol_filename: {
                "title": "Integrated surface densities / Molecular Gas",
                "caption": (
                    "Integrated surface densities of H2 gas and star-forming gas for"
                    " each individual galaxy. Quantities are calculated summing up"
                    " all gas (and SFR) within the galaxies' stellar half mass radius."
                ),
            },
        }

    fig_neut, ax_neut = pl.subplots(1, 1)
    fig_at, ax_at = pl.subplots(1, 1)
    fig_mol, ax_mol = pl.subplots(1, 1)

    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        for Zmask, _, linestyle, marker in metallicities:
            if Zmask == "all" and always_plot_scatter:
                plot_median_on_axis_as_pdf(
                    ax_neut, data.medians[f"sigma_neutral_SFR_azimuthal_{Zmask}"]
                )
                plot_median_on_axis_as_pdf(
                    ax_at, data.medians[f"sigma_HI_SFR_azimuthal_{Zmask}"]
                )
                plot_median_on_axis_as_pdf(
                    ax_mol, data.medians[f"sigma_H2_SFR_azimuthal_{Zmask}"]
                )

            plot_median_on_axis_as_line(
                ax_neut,
                data.medians[f"sigma_neutral_SFR_azimuthal_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )
            plot_median_on_axis_as_line(
                ax_at,
                data.medians[f"sigma_HI_SFR_azimuthal_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )
            plot_median_on_axis_as_line(
                ax_mol,
                data.medians[f"sigma_H2_SFR_azimuthal_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )

    for dataname, ax in zip(
        [
            "AzimuthallyAveragedNeutralKSRelation",
            None,
            "AzimuthallyAveragedMolecularKSRelation",
        ],
        [ax_neut, ax_at, ax_mol],
    ):

        sim_lines = []
        sim_labels = []
        for name in name_list:
            sim_lines.append(ax.plot([], [], "-")[0])
            sim_labels.append(name)
        for _, Zlabel, linestyle, marker in metallicities:
            sim_lines.append(
                ax.plot([], [], linestyle=linestyle, marker=marker, color="k")[0]
            )
            sim_labels.append(Zlabel)

        if dataname is not None:
            observational_data = load_observations(
                sorted(glob.glob(f"{observational_data_path}/{dataname}/*.hdf5"))
            )
            with unyt.matplotlib_support:
                for obs_data in observational_data:
                    obs_data.plot_on_axes(ax)

        ax.grid(True)
        ax.set_xlim(1.0e-1, 1.0e4)
        ax.set_ylim(1.0e-6, 1.0e1)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
        ax.legend(loc="lower right")
        ax.add_artist(sim_legend)

    neut_filename = f"{prefix}azimuthal_KS_neutral.png"
    fig_neut.savefig(f"{output_path}/{neut_filename}", dpi=300)
    pl.close(fig_neut)

    at_filename = f"{prefix}azimuthal_KS_atomic.png"
    fig_at.savefig(f"{output_path}/{at_filename}", dpi=300)
    pl.close(fig_at)

    mol_filename = f"{prefix}azimuthal_KS_molecular.png"
    fig_mol.savefig(f"{output_path}/{mol_filename}", dpi=300)
    pl.close(fig_mol)

    plots["Combined surface densities"] = {
        neut_filename: {
            "title": "Neutral KS relation (azimuthally averaged)",
            "caption": (
                "Combined spatially resolved measurements from N most massive"
                " individual galaxies, The X axis shows the surface density of"
                " neutral gas and the Y axis shows the star formation rate surface"
                " density. The surface densities were calculated using the"
                " azimuthally-averaging method with a radial width of 750pc."
                " Full lines show the median relation for all cells, while"
                " different line styles only include cells with a fixed"
                " metallicity (as indicated in the legend)."
            ),
        },
        at_filename: {
            "title": "Atomic KS relation (azimuthally averaged)",
            "caption": (
                "Combined spatially resolved measurements from N most massive"
                " individual galaxies, The X axis shows the surface density of"
                " atomic gas and the Y axis shows the star formation rate"
                " surface density. The surface densities were calculated using"
                " the azimuthally-averaging method with a radial width of 750pc."
                " Full lines show the median relation for all cells, while"
                " different line styles only include cells with a fixed"
                " metallicity (as indicated in the legend)."
            ),
        },
        mol_filename: {
            "title": "Molecular KS relation (azimuthally averaged)",
            "caption": (
                "Combined spatially resolved measurements from N most massive"
                " individual galaxies, The X axis shows the surface density of"
                " molecular gas and the Y axis shows the star formation rate"
                " surface density. The surface densities were calculated using"
                " the azimuthally-averaging method with a radial width of 750pc."
                " Full lines show the median relation for all cells, while"
                " different line styles only include cells with a fixed"
                " metallicity (as indicated in the legend)."
            ),
        },
    }

    fig_neut, ax_neut = pl.subplots(1, 1)
    fig_at, ax_at = pl.subplots(1, 1)
    fig_mol, ax_mol = pl.subplots(1, 1)

    for i, (_, data) in enumerate(zip(name_list, all_galaxies_list)):
        for Zmask, _, linestyle, marker in metallicities:
            if Zmask == "all" and always_plot_scatter:
                plot_median_on_axis_as_pdf(
                    ax_neut, data.medians[f"sigma_neutral_SFR_spatial_{Zmask}"]
                )
                plot_median_on_axis_as_pdf(
                    ax_at, data.medians[f"sigma_HI_SFR_spatial_{Zmask}"]
                )
                plot_median_on_axis_as_pdf(
                    ax_mol, data.medians[f"sigma_H2_SFR_spatial_{Zmask}"]
                )
            plot_median_on_axis_as_line(
                ax_neut,
                data.medians[f"sigma_neutral_SFR_spatial_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )
            plot_median_on_axis_as_line(
                ax_at,
                data.medians[f"sigma_HI_SFR_spatial_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )
            plot_median_on_axis_as_line(
                ax_mol,
                data.medians[f"sigma_H2_SFR_spatial_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )

    for dataname, ax in zip(
        [
            "SpatiallyResolvedNeutralKSRelation",
            None,
            "SpatiallyResolvedMolecularKSRelation",
        ],
        [ax_neut, ax_at, ax_mol],
    ):

        sim_lines = []
        sim_labels = []
        for name in name_list:
            sim_lines.append(ax.plot([], [], "-")[0])
            sim_labels.append(name)
        for _, Zlabel, linestyle, marker in metallicities:
            sim_lines.append(
                ax.plot([], [], linestyle=linestyle, marker=marker, color="k")[0]
            )
            sim_labels.append(Zlabel)

        if dataname is not None:
            observational_data = load_observations(
                sorted(glob.glob(f"{observational_data_path}/{dataname}/*.hdf5"))
            )
            with unyt.matplotlib_support:
                for obs_data in observational_data:
                    obs_data.plot_on_axes(ax)

        ax.grid(True)
        ax.set_xlim(1.0e-1, 1.0e4)
        ax.set_ylim(1.0e-6, 1.0e1)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
        ax.legend(loc="lower right")
        ax.add_artist(sim_legend)

    neut_filename = f"{prefix}spatial_KS_neutral.png"
    fig_neut.savefig(f"{output_path}/{neut_filename}", dpi=300)
    pl.close(fig_neut)

    at_filename = f"{prefix}spatial_KS_atomic.png"
    fig_at.savefig(f"{output_path}/{at_filename}", dpi=300)
    pl.close(fig_at)

    mol_filename = f"{prefix}spatial_KS_molecular.png"
    fig_mol.savefig(f"{output_path}/{mol_filename}", dpi=300)
    pl.close(fig_mol)

    plots["Combined surface densities"].update(
        {
            neut_filename: {
                "title": "Neutral KS relation (spatially averaged)",
                "caption": (
                    "Combined spatially resolved measurements from N most massive"
                    " individual galaxies. The X axis shows the surface density of neutral"
                    " gas and the Y axis shows the star formation rate surface density."
                    " The surface densities were calculated using the grid method with"
                    " a pixel size of 750pc."
                    " Full lines show the median relation for all cells, while"
                    " different line styles only include cells with a fixed"
                    " metallicity (as indicated in the legend)."
                ),
            },
            at_filename: {
                "title": "Atomic KS relation (spatially averaged)",
                "caption": (
                    "Combined spatially resolved measurements from N most massive"
                    " individual galaxies."
                    " The X axis shows the surface density of"
                    " atomic gas and the Y axis shows the star formation rate"
                    " surface density. The surface densities were calculated using"
                    " the grid method with a pixel size of 750pc."
                    " Full lines show the median relation for all cells, while"
                    " different line styles only include cells with a fixed"
                    " metallicity (as indicated in the legend)."
                ),
            },
            mol_filename: {
                "title": "Molecular KS relation (spatially averaged)",
                "caption": (
                    "Combined spatially resolved measurements from N most massive"
                    " individual galaxies."
                    " The X axis shows the surface density of"
                    " molecular gas and the Y axis shows the star formation rate"
                    " surface density. The surface densities were calculated using"
                    " the grid method with a pixel size of 750pc."
                    " Full lines show the median relation for all cells, while"
                    " different line styles only include cells with a fixed"
                    " metallicity (as indicated in the legend)."
                ),
            },
        }
    )

    fig_neut, ax_neut = pl.subplots(1, 1)
    fig_at, ax_at = pl.subplots(1, 1)
    fig_mol, ax_mol = pl.subplots(1, 1)

    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        for Zmask, _, linestyle, marker in metallicities:
            if Zmask == "all" and always_plot_scatter:
                plot_median_on_axis_as_pdf(
                    ax_neut, data.medians[f"sigma_neutral_tgas_spatial_{Zmask}"]
                )
                plot_median_on_axis_as_pdf(
                    ax_at, data.medians[f"sigma_HI_tgas_spatial_{Zmask}"]
                )
                plot_median_on_axis_as_pdf(
                    ax_mol, data.medians[f"sigma_H2_tgas_spatial_{Zmask}"]
                )
            plot_median_on_axis_as_line(
                ax_neut,
                data.medians[f"sigma_neutral_tgas_spatial_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )
            plot_median_on_axis_as_line(
                ax_at,
                data.medians[f"sigma_HI_tgas_spatial_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )
            plot_median_on_axis_as_line(
                ax_mol,
                data.medians[f"sigma_H2_tgas_spatial_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )

    for ax in [ax_neut, ax_at, ax_mol]:
        sim_lines = []
        sim_labels = []
        for name in name_list:
            sim_lines.append(ax.plot([], [], "-")[0])
            sim_labels.append(name)
        for _, Zlabel, linestyle, marker in metallicities:
            sim_lines.append(
                ax.plot([], [], linestyle=linestyle, marker=marker, color="k")[0]
            )
            sim_labels.append(Zlabel)
        ax.grid(True)
        ax.set_xlim(1.0e-1, 1.0e4)
        ax.set_ylim(1.0e7, 1.0e12)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
        ax.legend(loc="lower right")
        ax.add_artist(sim_legend)

    neut_filename = f"{prefix}spatial_tgas_neutral.png"
    fig_neut.savefig(f"{output_path}/{neut_filename}", dpi=300)
    pl.close(fig_neut)

    at_filename = f"{prefix}spatial_tgas_atomic.png"
    fig_at.savefig(f"{output_path}/{at_filename}", dpi=300)
    pl.close(fig_at)

    mol_filename = f"{prefix}spatial_tgas_molecular.png"
    fig_mol.savefig(f"{output_path}/{mol_filename}", dpi=300)
    pl.close(fig_mol)

    plots["Combined surface densities"].update(
        {
            neut_filename: {
                "title": "Neutral gas depletion time (spatially averaged)",
                "caption": (
                    "Depletion time of neutral gas vs. neutral gas surface density"
                    " from N most massive individual galaxies."
                    " The surface densities were"
                    " calculated using a grid with pixel size of 750 pc."
                    " Full lines show the median relation for all cells, while"
                    " different line styles only include cells with a fixed"
                    " metallicity (as indicated in the legend)."
                ),
            },
            at_filename: {
                "title": "Atomic gas depletion time (spatially averaged)",
                "caption": (
                    "Depletion time of atomic gas vs. atomic gas surface density"
                    " from N most massive individual galaxies."
                    " The surface densities were"
                    " calculated using a grid with pixel size of 750 pc."
                    " Full lines show the median relation for all cells, while"
                    " different line styles only include cells with a fixed"
                    " metallicity (as indicated in the legend)."
                ),
            },
            mol_filename: {
                "title": "Molecular gas depletion time (spatially averaged)",
                "caption": (
                    "Depletion time of molecular gas vs. molecular gas surface density"
                    " from N most massive individual galaxies."
                    " The surface densities were"
                    " calculated using a grid with pixel size of 750 pc."
                    " Full lines show the median relation for all cells, while"
                    " different line styles only include cells with a fixed"
                    " metallicity (as indicated in the legend)."
                ),
            },
        }
    )

    fig_neut, ax_neut = pl.subplots(1, 1)
    fig_at, ax_at = pl.subplots(1, 1)

    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        for Zmask, _, linestyle, marker in metallicities:
            if Zmask == "all" and len(name_list) == 1:
                plot_median_on_axis_as_pdf(
                    ax_neut, data.medians[f"H2_to_neutral_vs_neutral_spatial_{Zmask}"]
                )
                plot_median_on_axis_as_pdf(
                    ax_at, data.medians[f"H2_to_HI_vs_neutral_spatial_{Zmask}"]
                )
            plot_median_on_axis_as_line(
                ax_neut,
                data.medians[f"H2_to_neutral_vs_neutral_spatial_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )
            plot_median_on_axis_as_line(
                ax_at,
                data.medians[f"H2_to_HI_vs_neutral_spatial_{Zmask}"],
                color=f"C{i}",
                linestyle=linestyle,
                marker=marker,
            )

    for dataname, ax in zip(
        ["NeutralGasSurfaceDensityMolecularToAtomicRatio", None], [ax_neut, ax_at]
    ):
        sim_lines = []
        sim_labels = []
        for name in name_list:
            sim_lines.append(ax.plot([], [], "-")[0])
            sim_labels.append(name)
        for _, Zlabel, linestyle, marker in metallicities:
            sim_lines.append(
                ax.plot([], [], linestyle=linestyle, marker=marker, color="k")[0]
            )
            sim_labels.append(Zlabel)

        if dataname is not None:
            observational_data = load_observations(
                sorted(glob.glob(f"{observational_data_path}/{dataname}/*.hdf5"))
            )
            with unyt.matplotlib_support:
                for obs_data in observational_data:
                    obs_data.plot_on_axes(ax)

        ax.grid(True)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
        ax.legend(loc="lower right")
        ax.add_artist(sim_legend)

    ax_neut.set_xlim(1.0e-1, 1.0e4)
    ax_neut.set_ylim(1.0e-8, 1.0)
    ax_at.set_xlim(1.0e-1, 1.0e4)
    ax_at.set_ylim(1.0e-2, 1.0e3)

    neut_filename = f"{prefix}H2_to_neutral_vs_neutral_spatial.png"
    fig_neut.savefig(f"{output_path}/{neut_filename}", dpi=300)
    pl.close(fig_neut)

    at_filename = f"{prefix}H2_to_HI_vs_neutral_spatial.png"
    fig_at.savefig(f"{output_path}/{at_filename}", dpi=300)
    pl.close(fig_at)

    plots["Combined surface densities"].update(
        {
            neut_filename: {
                "title": "H2 to Neutral as a function of Neutral Surface Density (spatially averaged)",
                "caption": (
                    "Ratio of the molecular and neutral surface density as a"
                    " function of the neutral surface density"
                    " for individual galaxies."
                    " The surface densities were"
                    " calculated using a grid with pixel size of 750 pc."
                    " Full lines show the median relation for all cells, while"
                    " different line styles only include cells with a fixed"
                    " metallicity (as indicated in the legend)."
                ),
            },
            at_filename: {
                "title": "H2 to HI as a function of Neutral Surface Density (spatially averaged)",
                "caption": (
                    "Ratio of the molecular and atomic surface density as a"
                    " function of the neutral surface density"
                    " for individual galaxies."
                    " The surface densities were"
                    " calculated using a grid with pixel size of 750 pc."
                    " Full lines show the median relation for all cells, while"
                    " different line styles only include cells with a fixed"
                    " metallicity (as indicated in the legend)."
                ),
            },
        }
    )

    fig, ax = pl.subplots(1, 1)

    sim_lines = []
    sim_labels = []
    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        if len(name_list) == 1:
            plot_median_on_axis_as_pdf(ax, data.medians["H2_to_star_vs_star_spatial"])
        line = plot_median_on_axis_as_line(
            ax, data.medians["H2_to_star_vs_star_spatial"], color=f"C{i}"
        )
        sim_lines.append(line)
        sim_labels.append(name)

    ax.grid(True)
    ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
    sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
    ax.legend(loc="lower right")
    ax.add_artist(sim_legend)
    ax.set_xlim(1.0e-1, 1.0e4)
    ax.set_ylim(1.0e-3, 1.0e1)

    filename = f"{prefix}H2_to_star_vs_star_spatial.png"
    fig.savefig(f"{output_path}/{filename}", dpi=300)
    pl.close(fig)

    plots["Combined surface densities"].update(
        {
            filename: {
                "title": "Molecular to Stellar density vs Stellar Density",
                "caption": (
                    "Ratio of the molecular gas surface density and the stellar surface"
                    " density as a function of the stellar surface density. Computed"
                    " using a grid with a pixel size of 750 pc."
                ),
            }
        }
    )

    fig, ax = pl.subplots(1, 1)

    sim_lines = []
    sim_labels = []
    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        if len(name_list) == 1:
            plot_median_on_axis_as_pdf(ax, data.medians["SFR_to_H2_vs_H2_spatial"])
        line = plot_median_on_axis_as_line(
            ax, data.medians["SFR_to_H2_vs_H2_spatial"], color=f"C{i}"
        )
        sim_lines.append(line)
        sim_labels.append(name)

    ax.grid(True)
    ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
    sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
    ax.legend(loc="lower right")
    ax.add_artist(sim_legend)
    ax.set_xlim(1.0e-2, 1.0e3)
    ax.set_ylim(1.0e-11, 1.0e-7)

    filename = f"{prefix}SFR_to_H2_vs_H2_spatial.png"
    fig.savefig(f"{output_path}/{filename}", dpi=300)
    pl.close(fig)

    plots["Combined surface densities"].update(
        {
            filename: {
                "title": "SFR to Molecular density vs Molecular Density",
                "caption": (
                    "Ratio of the SFR surface density and the molecular gas surface"
                    " density as a function of the molecular gas surface density."
                    " Computed using a grid with a pixel size of 750 pc."
                ),
            }
        }
    )

    fig, ax = pl.subplots(1, 1)

    sim_lines = []
    sim_labels = []
    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        if len(name_list) == 1:
            plot_median_on_axis_as_pdf(ax, data.medians["SFR_to_star_vs_star_spatial"])
        line = plot_median_on_axis_as_line(
            ax, data.medians["SFR_to_star_vs_star_spatial"], color=f"C{i}"
        )
        sim_lines.append(line)
        sim_labels.append(name)

    ax.grid(True)
    ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
    sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
    ax.legend(loc="lower right")
    ax.add_artist(sim_legend)
    ax.set_xlim(1.0e-1, 1.0e4)
    ax.set_ylim(1.0e-13, 1.0e-7)

    filename = f"{prefix}SFR_to_star_vs_star_spatial.png"
    fig.savefig(f"{output_path}/{filename}", dpi=300)
    pl.close(fig)

    plots["Combined surface densities"].update(
        {
            filename: {
                "title": "SFR to Stellar density vs Stellar Density",
                "caption": (
                    "Ratio of the SFR surface density and the stellar surface"
                    " density as a function of the stellar surface density."
                    " Computed using a grid with a pixel size of 750 pc."
                ),
            }
        }
    )

    return plots
