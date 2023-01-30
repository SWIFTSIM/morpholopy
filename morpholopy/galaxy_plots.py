#!/usr/bin/env python3

"""
galaxy_plots.py

Code to generate images for individual galaxies.
Mostly copied over from the old morphology pipeline.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl
from matplotlib import gridspec
import unyt
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.slice import kernel_gamma
from swiftsimio.visualisation.smoothing_length_generation import (
    generate_smoothing_lengths,
)
from swiftsimio import swift_cosmology_to_astropy
from astropy.visualization import make_lupton_rgb

from scipy.optimize import curve_fit

cmRdBu = pl.get_cmap("RdYlBu_r")
Sigma_min = 0.0
Sigma_max = 99.0
H2_over_HI_min = 10.0 ** (-1.9)
H2_over_HI_max = 10.0 ** (+2.4)

twelve_plus_logOH_solar = 8.69
twelve_plus_logOH_min = twelve_plus_logOH_solar - 0.5
twelve_plus_logOH_max = twelve_plus_logOH_solar + 0.5
Zsun = 0.0134
twelve_plus_log_OH_solar = 8.69

#########################
# nr of map plots
#########################

nr_text_boxes = 1
nr_map_plots = 6
nr_scatter_plots = 1
nr_total_plots = nr_text_boxes + nr_map_plots + nr_scatter_plots
#################################
# Miscellaneous
#################################

npix = int(512 / 2)

vmin = -0.98
vmax = 2.98

r_img_kpc = 30.0 * unyt.kpc
lbar_kpc = 15.0 * unyt.kpc
ypos_bar = 20.0 * unyt.kpc

size = 2.0 * r_img_kpc
radialbinsizes = [0.22, 0.8, 1.8]  # in kpc
pixsize_kpc = r_img_kpc.value / npix

bingrid = 0.8 * unyt.kpc
npix_coarse = int(2.0 * r_img_kpc / bingrid)

cmap = pl.get_cmap("Greens")
usergreen = cmap(0.7)
cmap = pl.get_cmap("Blues")
userblue = cmap(0.7)


def exponential(x, Sigma0, H, offset):
    return Sigma0 * np.exp(-np.abs(x + offset) / H)

def calculate_scaleheight_pointmasses(zcoords, weights, zrange, resolution):

    S_1D, bin_edges = np.histogram(zcoords, bins = resolution, range = (-zrange, zrange),
                        weights = weights, density = False)

    z_1D = (bin_edges[1:] + bin_edges[:-1])/2.
    p0 = (hist.max(), 1., 0.)

    try:
        popt, pcov = curve_fit(
                exponential, z_1D[np.isfinite(S_1D)], S_1D[np.isfinite(S_1D)]
        )
    except:
        return np.nan

    return popt[1]
        

def calculate_scaleheight_fit(mass_map, r_img_kpc, r_abs_max_kpc):
    xx = np.linspace(
        -r_img_kpc.value, r_img_kpc.value, len(mass_map[:, 0]), endpoint=True
    )
    z = (np.tile(xx, (len(xx), 1))).T
    z_1D = np.ravel(z[:, (np.abs(xx) < r_abs_max_kpc)])
    S_1D = np.ravel(mass_map[:, (np.abs(xx) < r_abs_max_kpc)])

    p0 = (mass_map.max(), 1., 0.)

    try:
        popt, pcov = curve_fit(
                exponential, z_1D[np.isfinite(S_1D)], S_1D[np.isfinite(S_1D)],
                bounds = ( (0., 1.e-5, -5.), (np.inf, 100., +5.) )
        )
    except:
        return np.nan

    return popt[1]


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """

    if num == 0:
        return "0"
    else:
        if exponent is None:
            exponent = int(np.floor(np.log10(np.abs(num))))
        coeff = np.round(num / float(10 ** exponent), decimal_digits)
        if precision is None:
            precision = decimal_digits

        return r"${0:.{2}f}\times10^{{{1:d}}}$".format(coeff, exponent, precision)

def get_stars_surface_brightness_map(
    catalogue,
    halo_id,
    data,
    size,
    npix,
    r_img_kpc,
    face_on_rotation_matrix,
    edge_on_rotation_matrix,
):

    # needs to be in comoving coordinates for the mask
    visualise_region = [-0.5 * size, 0.5 * size, -0.5 * size, 0.5 * size]

    if not hasattr(data.stars, "smoothing_lenghts"):
        data.stars.smoothing_lengths = generate_smoothing_lengths(
            coordinates=data.stars.coordinates + 0.5 * data.metadata.boxsize[None, :],
            boxsize=data.metadata.boxsize,
            kernel_gamma=kernel_gamma,
            neighbours=11,
            speedup_fac=1,
            dimension=3,
        )

    luminosities = [
        data.stars.luminosities.GAMA_i,
        data.stars.luminosities.GAMA_r,
        data.stars.luminosities.GAMA_g,
    ]
    rgb_image_face = np.zeros((npix, npix, len(luminosities)))

    for ilum in range(len(luminosities)):
        # Face on projection
        data.stars.usermass = luminosities[ilum]
        pixel_grid = project_pixel_grid(
            data.stars,
            resolution=npix,
            project="usermass",
            parallel=True,
            region=visualise_region,
            rotation_center=unyt.unyt_array(
                [0.0, 0.0, 0.0], units=data.metadata.boxsize.units
            ),
            rotation_matrix=face_on_rotation_matrix,
            boxsize=data.metadata.boxsize,
            backend="subsampled",
        )
        x_range = visualise_region[1] - visualise_region[0]
        y_range = visualise_region[3] - visualise_region[2]
        units = 1.0 / (x_range * y_range)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / (x_range.units * y_range.units))

        mass_map_face = unyt.unyt_array(pixel_grid, units=units)
        mass_map_face.convert_to_units(1.0 / unyt.pc ** 2)
        try:
            mass_map_face[mass_map_face == 0.0] = mass_map_face[
                mass_map_face > 0.0
            ].min()
        except:
            mass_map_face[mass_map_face == 0.0] = 1.0e-10

        rgb_image_face[:, :, ilum] = mass_map_face.T
    image_face = make_lupton_rgb(
        rgb_image_face[:, :, 0],
        rgb_image_face[:, :, 1],
        rgb_image_face[:, :, 2],
        Q=10,
        stretch=0.5,
    )

    H_kpc_gri = np.zeros(len(luminosities))
    rgb_image_edge = np.zeros((npix, npix, len(luminosities)))
    for ilum in range(len(luminosities)):
        # Face on projection
        data.stars.usermass = luminosities[ilum]
        pixel_grid = project_pixel_grid(
            data.stars,
            resolution=npix,
            project="usermass",
            parallel=True,
            region=visualise_region,
            rotation_center=unyt.unyt_array(
                [0.0, 0.0, 0.0], units=data.metadata.boxsize.units
            ),
            rotation_matrix=edge_on_rotation_matrix,
            boxsize=data.metadata.boxsize,
            backend="subsampled",
        )

        x_range = visualise_region[1] - visualise_region[0]
        y_range = visualise_region[3] - visualise_region[2]
        units = 1.0 / (x_range * y_range)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / (x_range.units * y_range.units))

        mass_map_edge = unyt.unyt_array(pixel_grid, units=units)
        mass_map_edge.convert_to_units(1.0 / unyt.pc ** 2)
        try:
            mass_map_edge[mass_map_edge == 0.0] = mass_map_edge[
                mass_map_edge > 0.0
            ].min()
        except:
            mass_map_edge[mass_map_edge == 0.0] = 1.0e-10

        try:
            H_kpc_gri[ilum] = calculate_scaleheight_fit(mass_map_edge.T, r_img_kpc, 7.5)
        except:
            H_kpc_gri[ilum] = -1.0
        rgb_image_edge[:, :, ilum] = mass_map_edge.T

    image_edge = make_lupton_rgb(
        rgb_image_edge[:, :, 0],
        rgb_image_edge[:, :, 1],
        rgb_image_edge[:, :, 2],
        Q=10,
        stretch=0.5,
    )

    return image_face, image_edge, visualise_region, 0.0, 0.0, -1.0, H_kpc_gri


def get_stars_surface_density_map(
    catalogue,
    halo_id,
    plottype,
    data,
    size,
    npixloc,
    face_on_rotation_matrix,
    edge_on_rotation_matrix,
):
    visualise_region = [-0.5 * size, 0.5 * size, -0.5 * size, 0.5 * size]

    if not hasattr(data.stars, "smoothing_lenghts"):
        data.stars.smoothing_lengths = generate_smoothing_lengths(
            coordinates=data.stars.coordinates + 0.5 * data.metadata.boxsize[None, :],
            boxsize=data.metadata.boxsize,
            kernel_gamma=kernel_gamma,
            neighbours=11,
            speedup_fac=1,
            dimension=3,
        )

    # Face on projection
    pixel_grid = project_pixel_grid(
        data.stars,
        resolution=int(npixloc),
        project="masses",
        parallel=True,
        region=visualise_region,
        rotation_center=unyt.unyt_array(
            [0.0, 0.0, 0.0], units=data.metadata.boxsize.units
        ),
        rotation_matrix=face_on_rotation_matrix,
        boxsize=data.metadata.boxsize,
        backend="subsampled",
    )

    x_range = visualise_region[1] - visualise_region[0]
    y_range = visualise_region[3] - visualise_region[2]
    units = 1.0 / (x_range * y_range)
    # Unfortunately this is required to prevent us from {over,under}flowing
    # the units...
    units.convert_to_units(1.0 / (x_range.units * y_range.units))
    units *= getattr(data.stars, "masses").units

    mass_map_face = unyt.unyt_array(pixel_grid, units=units)
    mass_map_face.convert_to_units(unyt.Msun / unyt.pc ** 2)
    mass_map_face[mass_map_face == 0.0] = 1.0e-20

    pixelsize = (visualise_region[1] - visualise_region[0]) / float(npixloc)
    totalmass = np.sum(mass_map_face) * pixelsize * pixelsize
    totalmass.convert_to_units("Msun")

    # Edge on projection
    pixel_grid = project_pixel_grid(
        data.stars,
        resolution=int(npixloc),
        project="masses",
        parallel=True,
        region=visualise_region,
        rotation_center=unyt.unyt_array(
            [0.0, 0.0, 0.0], units=data.metadata.boxsize.units
        ),
        rotation_matrix=edge_on_rotation_matrix,
        boxsize=data.metadata.boxsize,
        backend="subsampled",
    )

    x_range = visualise_region[1] - visualise_region[0]
    y_range = visualise_region[3] - visualise_region[2]
    units = 1.0 / (x_range * y_range)
    # Unfortunately this is required to prevent us from {over,under}flowing
    # the units...
    units.convert_to_units(1.0 / (x_range.units * y_range.units))
    units *= getattr(data.stars, "masses").units

    mass_map_edge = unyt.unyt_array(pixel_grid, units=units)
    mass_map_edge.convert_to_units(unyt.Msun / unyt.pc ** 2)
    try:
        mass_map_edge[mass_map_edge == 0.0] = mass_map_edge[mass_map_edge > 0.0].min()
    except:
        mass_map_edge[mass_map_edge == 0.0] = 1.0e-10

    return mass_map_face.T, mass_map_edge.T, visualise_region, 0.0, 0.0, totalmass


def get_gas_surface_density_map(
    catalogue,
    halo_id,
    plottype,
    data,
    size,
    npixlocal,
    face_on_rotation_matrix,
    edge_on_rotation_matrix,
):
    visualise_region = [-0.5 * size, 0.5 * size, -0.5 * size, 0.5 * size]

    if plottype == "HI":
        data.gas.usermass = (
            data.gas.masses
            * data.gas.species_fractions.HI
            * data.gas.element_mass_fractions.hydrogen
        )
    elif plottype == "H2":
        data.gas.usermass = (
            data.gas.masses
            * 2.0
            * data.gas.species_fractions.H2
            * data.gas.element_mass_fractions.hydrogen
        )
    elif "GK11" in plottype:
        data.gas.usermass = (
            get_GK11fractions(data, plottype)
            * data.gas.masses
            * data.gas.element_mass_fractions.hydrogen
        )
    elif "hydrogen" in plottype:
        data.gas.usermass = data.gas.masses * data.gas.element_mass_fractions.hydrogen
    elif "helium" in plottype:
        data.gas.usermass = data.gas.masses * data.gas.element_mass_fractions.helium
    elif "totaloxygen" in plottype:
        data.gas.usermass = data.gas.masses * data.gas.element_mass_fractions.oxygen
    elif "diffuseoxygen" in plottype:
        data.gas.usermass = data.gas.diffuse_oxygen_masses_from_table
    elif "sfr" in plottype:
        data.gas.star_formation_rates.convert_to_units(unyt.Msun / unyt.yr)
        data.gas.star_formation_rates[data.gas.star_formation_rates < 0.0] = 0.0
        data.gas.usermass = data.gas.star_formation_rates
    else:
        raise RuntimeError(f"Unknown plot type: {plottype}!")

    if not "sfr" in plottype:
        data.gas.usermass.convert_to_units("Msun")

    # Face on projection
    mass_map_face = project_gas(
        data,
        resolution=int(npixlocal),
        project="usermass",
        parallel=True,
        region=visualise_region,
        rotation_center=unyt.unyt_array(
            [0.0, 0.0, 0.0], units=data.metadata.boxsize.units
        ),
        rotation_matrix=face_on_rotation_matrix,
        backend="subsampled",
    )

    if "sfr" in plottype:
        mass_map_face.convert_to_units(unyt.Msun / unyt.yr / unyt.pc ** 2)
    else:
        mass_map_face.convert_to_units(unyt.Msun / unyt.pc ** 2)

    pixelsize = (visualise_region[1] - visualise_region[0]) / float(npixlocal)
    totalmass = np.sum(mass_map_face) * pixelsize * pixelsize
    if "sfr" in plottype:
        totalmass.convert_to_units("Msun/yr")
    else:
        totalmass.convert_to_units("Msun")

    # Edge on projection
    mass_map_edge = project_gas(
        data,
        resolution=int(npixlocal),
        project="usermass",
        parallel=True,
        region=visualise_region,
        rotation_center=unyt.unyt_array(
            [0.0, 0.0, 0.0], units=data.metadata.boxsize.units
        ),
        rotation_matrix=edge_on_rotation_matrix,
        backend="subsampled",
    )

    if "sfr" in plottype:
        mass_map_edge.convert_to_units(unyt.Msun / unyt.yr / unyt.pc ** 2)
    else:
        mass_map_edge.convert_to_units(unyt.Msun / unyt.pc ** 2)

    return mass_map_face.T, mass_map_edge.T, visualise_region, 0.0, 0.0, totalmass


def get_radial_profile(mass_map, radialbin_kpc, pixsize_kpc, r_img_kpc):
    nbins_radial = int(r_img_kpc.value / radialbin_kpc)

    y, x = np.indices((mass_map.shape))
    npix = len(mass_map[0, :])
    pcenter = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - pcenter[0], y - pcenter[1])
    r_1D = r.ravel() * pixsize_kpc

    # from mass per pc-2 to mass per pixel
    mass_map_per_pixel = mass_map * pixsize_kpc * pixsize_kpc * unyt.kpc ** 2
    mass_map_per_pixel.convert_to_units(unyt.Msun)
    mass_map_1D = (mass_map_per_pixel.ravel()).value

    rmin = 0.0
    rmax = nbins_radial * radialbin_kpc

    M_hist, r_hist = np.histogram(
        r_1D, weights=mass_map_1D, bins=nbins_radial, range=(rmin, rmax), density=False
    )

    # calculate the area of each radial bin
    A = np.zeros(len(r_hist) - 1)
    for i in range(len(r_hist) - 1):
        A[i] = np.pi * (r_hist[i + 1] * r_hist[i + 1] - r_hist[i] * r_hist[i])

    A_unyt = A * unyt.kpc ** 2
    Sigma = M_hist * unyt.Msun / A_unyt
    Sigma.convert_to_units(unyt.Msun / unyt.pc ** 2)

    r_central = np.zeros(len(r_hist) - 1)
    for i in range(len(r_hist) - 1):
        r_central[i] = 0.5 * (r_hist[i + 1] + r_hist[i])

    r_central = r_central * unyt.kpc

    return r_central, Sigma


def plot_galaxy(
    catalogue,
    halo_id,
    index,
    data,
    face_on_rotation_matrix,
    edge_on_rotation_matrix,
    output_path,
):

    FLTMIN = np.nextafter(np.float32(0), np.float32(1))
    if FLTMIN == 0.:
        #if shared objects are compiled with e.g. fast math
        FLTMIN = 1.e-30

    image_info =  "Image size: (%.1f x %.1f) kpc, "%(2. * r_img_kpc.value, 2. * r_img_kpc.value)
    image_info += " resolution: (%i x %i) pixel."%(npix, npix) 

    galaxy_coords_text = (
           f"[ {catalogue.positions.xcmbp[halo_id].to('Mpc').value:.02f}, "
         + f"{catalogue.positions.ycmbp[halo_id].to('Mpc').value:.02f}, "
         + f"{catalogue.positions.zcmbp[halo_id].to('Mpc').value:.02f} ] cMpc"
    )

    galaxy_info_title = (
        f"Galaxy {halo_id:08d} " + galaxy_coords_text
    )

    galaxy_info_short = (
        r"M$_{\mathrm{200,crit}}$ = "
        + sci_notation(catalogue.masses.mass_200crit[halo_id].to("Msun").value)
        + r" M$_{\odot}$, "
        + r"M$_{\mathrm{*,30kpc}}$ = "
        + sci_notation(catalogue.masses.mass_star_30kpc[halo_id].to("Msun").value)
        + r" M$_{\odot}$, "
    )

    galaxy_info = (
        " Coordinates (x,y,z) = " + galaxy_coords_text + ", "
    )
    galaxy_info += (
        r"M$_{\mathrm{200,crit}}$ = "
        + sci_notation(catalogue.masses.mass_200crit[halo_id].to("Msun").value)
        + r" M$_{\odot}$, "
    )
    galaxy_info += (
        r"M$_{\mathrm{*,30kpc}}$ = "
        + sci_notation(catalogue.masses.mass_star_30kpc[halo_id].to("Msun").value)
        + r" M$_{\odot}$, "
    )
    galaxy_info += (
        r"M$_{\mathrm{gas,30kpc}}$ = "
        + sci_notation(catalogue.masses.mass_gas_30kpc[halo_id].to("Msun").value)
        + r" M$_{\odot}$, "
    )
    galaxy_info += (
        r"M$_{\mathrm{HI,30kpc}}$ = "
        + sci_notation(
            catalogue.gas_hydrogen_species_masses.HI_mass_30_kpc[halo_id]
            .to("Msun")
            .value
        )
        + r" M$_{\odot}$, "
    )
    galaxy_info += (
        r"M$_{\mathrm{H2,30kpc}}$ = "
        + sci_notation(
            catalogue.gas_hydrogen_species_masses.H2_mass_30_kpc[halo_id]
            .to("Msun")
            .value
        )
        + r" M$_{\odot}$, "
    )
    sSFR = (
        catalogue.apertures.sfr_gas_100_kpc[halo_id]
        / catalogue.apertures.mass_star_100_kpc[halo_id]
    )
    if np.isfinite(sSFR):
        galaxy_info += (
            r"sSFR$_{\mathrm{100}}$ = "
            + sci_notation(sSFR.to("Gyr**(-1)").value)
            + r" Gyr$^{-1}$"
        )

    stars_faceon_filename = "galaxy_%3.3i_map_stars_faceon.png"%(index)
    stars_edgeon_filename = "galaxy_%3.3i_map_stars_edgeon.png"%(index)
    HI_faceon_filename    = "galaxy_%3.3i_map_HI_faceon.png"%(index) 
    HI_edgeon_filename    = "galaxy_%3.3i_map_HI_edgeon.png"%(index) 
    H2_faceon_filename    = "galaxy_%3.3i_map_H2_faceon.png"%(index) 
    H2_edgeon_filename    = "galaxy_%3.3i_map_H2_edgeon.png"%(index) 

    plots = {}

    fig_stars_faceon, ax_stars_faceon = pl.subplots(1, 1)
    fig_stars_edgeon, ax_stars_edgeon = pl.subplots(1, 1)
    fig_HI_faceon, ax_HI_faceon = pl.subplots(1, 1)
    fig_HI_edgeon, ax_HI_edgeon = pl.subplots(1, 1)
    fig_H2_faceon, ax_H2_faceon = pl.subplots(1, 1)
    fig_H2_edgeon, ax_H2_edgeon = pl.subplots(1, 1)

    # Stars
    ax_stars_faceon.set_title("Stars (gri) - face")
    ax_stars_edgeon.set_title("Stars (gri) - edge")
    (
        mass_map_face,
        mass_map_edge,
        visualise_region,
        x,
        y,
        totalmass,
        H_kpc_gri,
    ) = get_stars_surface_brightness_map(
        catalogue,
        halo_id,
        data,
        size,
        npix,
        r_img_kpc,
        face_on_rotation_matrix,
        edge_on_rotation_matrix,
    )
    im = ax_stars_faceon.imshow(mass_map_face, extent=visualise_region)
    im = ax_stars_edgeon.imshow(mass_map_edge, extent=visualise_region)

    # HI
    ax_HI_faceon.set_title("Gas (HI) - face")
    ax_HI_edgeon.set_title("Gas (HI) - edge")

    (   
        mass_map_face,
        mass_map_edge,
        visualise_region,
        x,
        y,
        totalmass,
    ) = get_gas_surface_density_map(
        catalogue,
        halo_id,
        "HI",
        data,
        size,
        npix,
        face_on_rotation_matrix,
        edge_on_rotation_matrix,
    )
    mass_map_face.convert_to_units("Msun / pc**2")
    mass_map_edge.convert_to_units("Msun / pc**2")
    mass_map_face_plot_HI = mass_map_face  # save for H2 ratio plot
    mass_map_face_plot = np.log10(mass_map_face.value)
    mass_map_edge_plot = np.log10(mass_map_edge.value)

    im = ax_HI_faceon.imshow(
        mass_map_face_plot, cmap="Blues", extent=visualise_region, vmin=vmin, vmax=vmax
    )
    im = ax_HI_edgeon.imshow(
        mass_map_edge_plot, cmap="Blues", extent=visualise_region, vmin=vmin, vmax=vmax
    )

    # H2
    ax_H2_faceon.set_title("Gas (H2) - face")
    ax_H2_edgeon.set_title("Gas (H2) - edge")

    (   
        mass_map_face,
        mass_map_edge,
        visualise_region,
        x,
        y,
        totalmass,
    ) = get_gas_surface_density_map(
        catalogue,
        halo_id,
        "H2",
        data,
        size,
        npix,
        face_on_rotation_matrix,
        edge_on_rotation_matrix,
    )
    mass_map_face.convert_to_units("Msun / pc**2")
    mass_map_edge.convert_to_units("Msun / pc**2")
    mass_map_face_plot_H2 = mass_map_face  # save for H2 ratio plot
    mass_map_face_plot = np.log10(np.maximum(mass_map_face.value, FLTMIN))
    mass_map_edge_plot = np.log10(np.maximum(mass_map_edge.value, FLTMIN))

    im = ax_H2_faceon.imshow(
        mass_map_face_plot, cmap="Greens", extent=visualise_region, vmin=vmin, vmax=vmax
    )
    im = ax_H2_edgeon.imshow(
        mass_map_edge_plot, cmap="Greens", extent=visualise_region, vmin=vmin, vmax=vmax
    )

    for ax in [ax_stars_faceon, ax_stars_edgeon, 
               ax_HI_faceon, ax_HI_edgeon, 
               ax_H2_faceon, ax_H2_edgeon]:
        ax.set_aspect("equal")
        ax.tick_params(labelleft=False, labelbottom=False)
        ax.set_xticks([])
        ax.set_yticks([])

    fig_stars_faceon.savefig(f"{output_path}/{stars_faceon_filename}", dpi = 300)
    fig_stars_edgeon.savefig(f"{output_path}/{stars_edgeon_filename}", dpi = 300)
    fig_HI_faceon.savefig(f"{output_path}/{HI_faceon_filename}", dpi = 300)
    fig_HI_edgeon.savefig(f"{output_path}/{HI_edgeon_filename}", dpi = 300)
    fig_H2_faceon.savefig(f"{output_path}/{H2_faceon_filename}", dpi = 300)
    fig_H2_edgeon.savefig(f"{output_path}/{H2_edgeon_filename}", dpi = 300)

    pl.close(fig_stars_faceon)
    pl.close(fig_stars_edgeon)
    pl.close(fig_HI_faceon)
    pl.close(fig_HI_edgeon)
    pl.close(fig_H2_faceon)
    pl.close(fig_H2_edgeon)

    plots["Gallery"] = {
        stars_faceon_filename: {
            "title": galaxy_info_title,
            "caption": (
                            "Unattenuated gri image in face-on projection. "
                            + galaxy_info_short
            ),
        },
        stars_edgeon_filename: {
            "title": galaxy_info_title,
            "caption": (
                            "Unattenuated gri image in edge-on projection. "
                            + galaxy_info_short
            ),
        },
    }

    plots["Visualisation"] = {
        stars_faceon_filename: {
            "title": "Stars (face-on)",
            "caption": (
                            "Unattenuated gri image in face-on projection. "
                            + image_info
                            + galaxy_info
            ),
        },
        stars_edgeon_filename: {
            "title": "Stars (edge-on)",
            "caption": (
                            "Unattenuated gri image in edge-on projection. "
                            + image_info
                            + galaxy_info
            ),
        },
        HI_faceon_filename: {
            "title": "HI surface density (face-on)",
            "caption": (
                            r"HI surface density in units of log$_{\mathrm{10}}$ M$_{\odot} \,\mathrm{pc}^{-2}$,"
                            f" colormap range: [ {vmin:.01f}, {vmax:.01f} ], "
                             "in face-on projection."
                            + image_info
                            + galaxy_info
            ),
        },
        HI_edgeon_filename: {
            "title": "HI surface density (edge-on)",
            "caption": (
                            r"HI surface density in units of log$_{\mathrm{10}}$ M$_{\odot} \,\mathrm{pc}^{-2}$, "
                            f" colormap range: [ {vmin:.01f}, {vmax:.01f} ], "
                             "in edge-on projection."
                            + image_info
                            + galaxy_info
            ),
        },
        H2_faceon_filename: {
            "title": "H2 surface density (face-on)",
            "caption": (
                            r"H2 surface density in units of log$_{\mathrm{10}}$ M$_{\odot} \,\mathrm{pc}^{-2}$, "
                            f" colormap range: [ {vmin:.01f}, {vmax:.01f} ], "
                             "in face-on projection."
                            + image_info
                            + galaxy_info
            ),
        },
        H2_edgeon_filename: {
            "title": "H2 surface density (edge-on)",
            "caption": (
                            r"H2 surface density in units of log$_{\mathrm{10}}$ M$_{\odot} \,\mathrm{pc}^{-2}$, "
                            f" colormap range: [ {vmin:.01f}, {vmax:.01f} ], "
                             "in edge-on projection."
                            + image_info
                            + galaxy_info
            ),
        },
    }

    return plots
