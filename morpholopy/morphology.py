#!/usr/bin/env python3

"""
morphology.py

Morphology related plots: axis lengths and angular momenta.
"""

import numpy as np
import unyt
import swiftsimio as sw
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio import SWIFTDataset

from .orientation import get_orientation_mask
from .plot import plot_data_on_axis
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl

from typing import Dict, List, Tuple, Union
from numpy.typing import NDArray
from .logging import GalaxyLog
from .medians import plot_median_on_axis_as_line

from scipy.optimize import curve_fit
from scipy import stats

from unyt import proton_mass_cgs as mH

def exponential(x, Sigma0, H, offset):
    return Sigma0 * np.exp(-np.abs(x + offset) / H)

###################################################################
# see e.g. Eq. 4 of Smith et al. (https://arxiv.org/pdf/2301.08265.pdf)
# but added the option for an offset
###################################################################
def exponential_density(z, n0, H, offset):
    return n0 * np.exp(- np.power(z + offset, 2) / (2. * np.power(H, 2)))


###################################################################
# Calculate the scaleheight of the gas volume density 
# input:
#   zcoords: the coordinates of the particle positions perpendicular to the disk;
#            this routine assumes that the particle selection, the rotation of the
#            galaxy into an edge-on view and the translation so that the center of the 
#            galaxy is at (0,0,0) - or close to it - is already taken care of
#   densities: gas densities in particles per volume; can be e.g. n_H, n_HI, n_H2, ...
#   img_size: size of the selection region
#   resolution: number of vertical bins
#   index: halo id for debugging
###################################################################
def calculate_scaleheight_from_densities(zcoords, densities, img_size, resolution, index):
    # bounds for fitting:
    # scale height H needs to be between H_bound_min_kpc and H_bound_max_kpc kpc;
    # a non-zero H_bound_min_kpc is particularly important to avoid division by 0
    # The maximum bound should be set generously because the fit will be discarded
    # if the fitted scale height equals H_bound_max_kpc
    H_bound_min_kpc = 0.01
    H_bound_max_kpc = 10.

    # offset needs be within [-absoffset_bound_kpc, absoffset_bound_kpc]
    # this is also used to validate the fit. If the fitted offset lies exactly
    # at the bound, something went wrong
    absoffset_bound_kpc = 5.

    # no gas - no scaleheight
    if len(densities) == 0:
        return np.nan

    # make sure everything is in the right units
    img_size.convert_to_units('kpc')
    zcoords.convert_to_units('kpc')
    densities.convert_to_units('cm**-3')

    # no ISM gas - no scaleheight
    if densities.value.max() < 0.1:
        return np.nan

    vert = np.asarray(zcoords, dtype = np.float64)
    dens = np.asarray(densities, dtype = np.float64)

    # bin the data in bins of zcoords and get both the mean density and 
    # the standard deviation of each bin
    mean_density, bin_edges, binnumber = stats.binned_statistic(vert, dens, statistic = "mean",
                                          bins = resolution, 
                                          range = (-img_size.value/2., img_size.value/2.))

    # convert arrays to float64 as mentioned in the notes of scipy.optimize.curve_fit
    z_1D = np.asarray((bin_edges[1:] + bin_edges[:-1])/2., dtype = np.float64)
    mean = np.asarray(mean_density, dtype = np.float64)

    if np.sum(np.isfinite(mean)) == 0: 
        scaleheight = np.nan
    else:
        meanmax = mean[np.isfinite(mean)].max()
        maskbins = np.isfinite(mean) 
        p0 = (meanmax, 1., 0.)

        try:
            popt, pcov = curve_fit(
                    exponential_density, z_1D[maskbins], mean[maskbins],
                    bounds = ( (meanmax/10., H_bound_min_kpc, -1. * absoffset_bound_kpc), 
                               (meanmax*10., H_bound_max_kpc, +1. * absoffset_bound_kpc) ) )
        except:
            scaleheight =  np.nan
        else:
            scaleheight = popt[1]
            # It is not a good sign if the 
            # found fitting parameter is right at the boundary; 
            # assume that the fit is bad in this case
            if np.abs(1. - np.abs(popt[2]) / absoffset_bound_kpc) < 0.01:
                scaleheight = np.nan
            elif np.abs(1. - popt[1] / H_bound_max_kpc) < 0.01:
                scaleheight = np.nan
            else:
                scaleheight = popt[1]

    return scaleheight

def calculate_scaleheight_from_points(zcoords, weights, img_size, resolution):
    img_size.convert_to_units('kpc')
    zcoords.convert_to_units('kpc')

    S_1D, bin_edges = np.histogram(zcoords.value, bins = resolution, 
                        range = (-img_size.value/2., img_size.value/2.),
                        weights = weights.value, density = False)

    z_1D = (bin_edges[1:] + bin_edges[:-1])/2.
    p0 = (S_1D.max(), 1., 0.)

    try:
        popt, pcov = curve_fit(
                exponential, z_1D[np.isfinite(S_1D)], S_1D[np.isfinite(S_1D)],
                bounds = ( (S_1D.max()/10., 1.e-5, -5.), (S_1D.max()*10., 100., +5.) )
        )
    except:
        return np.nan

    return popt[1]

def calculate_scaleheight_from_map(mass_map, img_size):
    img_size.convert_to_units('kpc')

    xx = np.linspace(
            -img_size.value/2., img_size.value/2., len(mass_map[:,0]), endpoint=True
         )

    z  = (np.tile(xx, (len(xx), 1))).T
    z_1D = np.ravel(z)
    S_1D = np.ravel(mass_map)
    
    p0 = (mass_map.max(), 1., 0.)

    try:
        popt, pcov = curve_fit(
                exponential, z_1D[np.isfinite(S_1D)], S_1D[np.isfinite(S_1D)],
                bounds = ( (0., 1.e-5, -5.), (np.inf, 100., +5.) )
        )
    except:
        return np.nan

    return popt[1]

def get_scaleheight(
    galaxy_log: GalaxyLog,
    data: SWIFTDataset,
    half_mass_radius: unyt.unyt_quantity,
    edge_on_rmatrix: NDArray[float],
    gas_mask: NDArray[bool],
    stars_mask: NDArray[bool],
    index: int,
    scaleheight_binsize_kpc: float,
    scaleheight_lower_mass_limit_in_Msun: float,
) -> Tuple[unyt.unyt_quantity, unyt.unyt_quantity, unyt.unyt_quantity]: 

    # Image size is 4 * half_mass_radius
    # but at maximum 60 kpc (limited by 30 kpc aperture)
    img_size =  sw.objects.cosmo_array( 
        np.minimum(4. * half_mass_radius, 60. * unyt.kpc),
        comoving=False,
        cosmo_factor=data.gas.coordinates.cosmo_factor,
    )    

    maxbinsize = sw.objects.cosmo_array(
        scaleheight_binsize_kpc, 
        comoving=False,
        cosmo_factor=data.gas.coordinates.cosmo_factor,
    )

    minimum_mass = sw.objects.cosmo_array(
        scaleheight_lower_mass_limit_in_Msun,
        comoving=False,
        cosmo_factor=data.gas.masses.cosmo_factor,
    )

    resolution = int(img_size / maxbinsize) + 1

    rot_center = unyt.unyt_array(
                    [0.0, 0.0, 0.0], units=data.gas.coordinates.units
                 )

    #########
    # Gas
    #########
    # rotate gas coordinates
    x, y, z = np.matmul(edge_on_rmatrix, (data.gas.coordinates[gas_mask] - rot_center).T)
    x += rot_center[0]
    y += rot_center[1]
    z += rot_center[2]

    x.convert_to_units('kpc')
    y.convert_to_units('kpc')
    z.convert_to_units('kpc')

    # get gas species densities; do in steps to prevent overflow
    nHI = data.gas.densities[gas_mask] * data.gas.species_fractions.HI[gas_mask]
    nHI.convert_to_cgs()
    nHI /= mH

    MHI = data.gas.masses[gas_mask] * data.gas.species_fractions.HI[gas_mask]
    MHI.convert_to_units("msun")

    nH2 = data.gas.densities[gas_mask] * data.gas.species_fractions.H2[gas_mask]
    nH2.convert_to_cgs()
    nH2 /= mH

    MH2 = data.gas.masses[gas_mask] * data.gas.species_fractions.H2[gas_mask] * 2.
    MH2.convert_to_units("msun")

    mask = (np.abs(x) <= img_size/2.)  & (np.abs(y) <= img_size/2.) & (np.abs(z) <= img_size/2.)

    # Only calculate scale heights if the HI mass within the analysed cube is
    # above the minimum mass set in the config file 
    if np.sum(MHI[mask]) >= minimum_mass:
        H_HI_density_kpc = calculate_scaleheight_from_densities(y[mask], nHI[mask],
                                                                img_size, resolution, index)
    else:
        H_HI_density_kpc = np.nan
    
    # Only calculate scale heights if the H2 mass within the analysed cube is
    # above the minimum mass set in the config file 
    if np.sum(MH2[mask]) >= minimum_mass:
        H_H2_density_kpc = calculate_scaleheight_from_densities(y[mask], nH2[mask],
                                                                img_size, resolution, index)
    else:
        H_H2_density_kpc = np.nan

    #########
    # Stars
    #########
    # rotate stars coordinates
    x, y, z = np.matmul(edge_on_rmatrix, (data.stars.coordinates[stars_mask] - rot_center).T)
    x += rot_center[0]
    y += rot_center[1]
    z += rot_center[2]    

    x.convert_to_units('kpc')
    y.convert_to_units('kpc')
    z.convert_to_units('kpc')
    M = data.stars.masses[stars_mask]
    M.convert_to_units('msun')

    mask = (np.abs(x) <= img_size/2.)  & (np.abs(y) <= img_size/2.)

    H_stars_kpc = calculate_scaleheight_from_points(y[mask], M[mask],
                                                    img_size, resolution)

    galaxy_log.debug("Scaleheights: H_HI = %.2f kpc, H_H2 = %.2f kpc, H_stars = %.2f kpc"
                    %(H_HI_density_kpc, H_H2_density_kpc, H_stars_kpc))
    

    return [H_HI_density_kpc, H_H2_density_kpc, H_stars_kpc]

def get_axis_lengths_tensor(
    galaxy_log: GalaxyLog,
    partdata: "SWIFTParticleDataset",
    half_mass_radius: unyt.unyt_quantity,
    mass_variable: str = "masses",
    reduced: bool = True,
    iterative: bool = True,
) -> Tuple[unyt.unyt_array, NDArray[float]]:
    """
    Get the axis lengths and z axis vector by diagonalising the
    (reduced) moment of inertia tensor of the given particles,
    using the given mass variable.

    This either uses the normal or the reduced moment of inertia tensor
    depending on the value of the 'reduced' parameter.

    When 'iterative=True', the initial axis ratios are used to define an
    ellipsoid with the same volume as the original aperture sphere. The
    calculation is then repeated for all the particles within this ellipsoid.
    The procedure is repeated until the axis ratios converge.
    """

    all_position = partdata.coordinates
    all_mass = getattr(partdata, mass_variable)
    all_radius = np.sqrt((all_position ** 2).sum(axis=1))

    mask = all_radius <= 0.5 * half_mass_radius
    position = all_position[mask]
    mass = all_mass[mask]
    radius = all_radius[mask]

    weight = None
    if reduced:
        weight = unyt.unyt_array(np.zeros(mass.shape), mass.units / radius.units ** 2)
        weight[radius > 0.0] = (mass[radius > 0.0] / radius[radius > 0.0] ** 2).to(
            weight.units
        )
    else:
        weight = unyt.unyt_array(mass)

    if weight.sum() == 0.0:
        galaxy_log.debug("Total weight of 0, so not calculating axis lengths.")
        return unyt.unyt_array(np.zeros(3), position.units), np.zeros(3)

    Itensor = (weight[:, None, None] / weight.sum()) * np.ones((weight.shape[0], 3, 3))
    # Note: unyt currently ignores the position units in the *=
    # i.e. Itensor is dimensionless throughout (even though it should not be)
    for i in range(3):
        for j in range(3):
            Itensor[:, i, j] *= position[:, i] * position[:, j]

    Itensor = Itensor.sum(axis=0)

    # linalg.eigenvals cannot deal with units anyway, so we have to add them
    # back in
    axes, basis = np.linalg.eig(Itensor)
    axes = axes.real
    axes = np.clip(axes, 0.0, None)
    axes = np.sqrt(axes) * position.units

    # sort the axes from long to short
    isort = np.argsort(axes)[::-1]
    axes = axes[isort]
    basis = basis[isort]

    if not iterative:
        # we got the result we wanted, return
        return axes, basis[2].real

    # we need to continue iterating
    R2 = (0.5 * half_mass_radius) ** 2

    if (axes[0] == 0.0) or (axes[1] == 0.0) or (axes[2] == 0.0):
        galaxy_log.debug(f"Zero axis ratio! Giving up on this galaxy.")
        return unyt.unyt_array(np.zeros(3), position.units), np.zeros(3)

    c_a = axes[2] / axes[0]
    b_a = axes[1] / axes[0]
    old_c_a = 2.0 * c_a
    old_b_a = 2.0 * b_a
    loop = 0
    while (abs((c_a - old_c_a) / (c_a + old_c_a)) > 0.01) or (
        abs((b_a - old_b_a) / (b_a + old_b_a)) > 0.01
    ):

        if (c_a == 0.0) or (b_a == 0.0):
            galaxy_log.debug(f"Zero axis ratio! Giving up on this galaxy.")
            return unyt.unyt_array(np.zeros(3), position.units), np.zeros(3)

        loop += 1
        if loop == 100:
            galaxy_log.debug(
                f"Too many iterations (c_a: {old_c_a} - {c_a}, b_a: {old_b_a} - {b_a})!"
            )
            break
        old_c_a = c_a
        old_b_a = b_a

        ra = (all_position * basis[0]).sum(axis=1)
        rb = (all_position * basis[1]).sum(axis=1)
        rc = (all_position * basis[2]).sum(axis=1)
        mask = (ra ** 2 + rb ** 2 / b_a ** 2 + rc ** 2 / c_a ** 2) <= R2 / (
            b_a * c_a
        ) ** (2.0 / 3.0)

        mass = all_mass[mask]
        position = all_position[mask]

        radius = np.sqrt((position ** 2).sum(axis=1))

        if reduced:
            weight = unyt.unyt_array(
                np.zeros(mass.shape), mass.units / radius.units ** 2
            )
            weight[radius > 0.0] = (mass[radius > 0.0] / radius[radius > 0.0] ** 2).to(
                weight.units
            )
        else:
            weight = unyt.unyt_array(mass)

        if weight.sum() == 0.0:
            galaxy_log.debug(
                "Total weight of 0, so not calculating axis lengths."
                " Using last value that worked."
            )
            break

        Itensor = (weight[:, None, None] / weight.sum()) * np.ones(
            (weight.shape[0], 3, 3)
        )
        # Note: unyt currently ignores the position units in the *=
        # i.e. Itensor is dimensionless throughout (even though it should not be)
        for i in range(3):
            for j in range(3):
                Itensor[:, i, j] *= position[:, i] * position[:, j]

        Itensor = Itensor.sum(axis=0)

        # linalg.eigenvals cannot deal with units anyway, so we have to add them
        # back in
        axes, basis = np.linalg.eig(Itensor)
        axes = axes.real
        axes = np.clip(axes, 0.0, None)
        axes = np.sqrt(axes) * position.units

        # sort the axes from long to short
        isort = np.argsort(axes)[::-1]
        axes = axes[isort]
        basis = basis[isort]

        c_a = axes[2] / axes[0]
        b_a = axes[1] / axes[0]

    return axes, basis[2].real


def get_axis_lengths_reduced_tensor(
    galaxy_log: GalaxyLog,
    partdata: "SWIFTParticleDataset",
    half_mass_radius: unyt.unyt_quantity,
    mass_variable: str = "masses",
) -> Tuple[unyt.unyt_array, NDArray[float]]:
    """
    Call get_axis_lengths_tensor() with reduced=True.
    """
    return get_axis_lengths_tensor(
        galaxy_log,
        partdata,
        half_mass_radius,
        mass_variable,
        reduced=True,
        iterative=True,
    )


def get_axis_lengths_normal_tensor(
    galaxy_log: GalaxyLog,
    partdata: "SWIFTParticleDataset",
    half_mass_radius: unyt.unyt_quantity,
    mass_variable: str = "masses",
) -> Tuple[unyt.unyt_array, NDArray[float]]:
    """
    Call get_axis_lengths_tensor() with reduced=False.
    """
    return get_axis_lengths_tensor(
        galaxy_log,
        partdata,
        half_mass_radius,
        mass_variable,
        reduced=False,
        iterative=True,
    )


def get_kappa_corot(
    partdata: "SWIFTParticleDataset",
    half_mass_radius: unyt.unyt_quantity,
    R200crit: unyt.unyt_quantity,
    Rvir: unyt.unyt_quantity,
    orientation_type: str,
    orientation_vector: NDArray[float],
    mass_variable: str = "masses",
) -> Tuple[unyt.unyt_quantity, NDArray[float]]:
    """
    Calculate kappa corot for the given particles and mass variable, using
    the given masking strategy (which might depend on any of the
    radii passed on as arguments), and respecting the orientation
    vector determined earlier.

    Returns the total specific angular momentum (vector norm) and kappa corot,
    the ratio of the kinetic energy in ordered rotation and the total kinetic
    energy in the same component.
    """

    _, inner_aperture, outer_aperture, clipping = orientation_type.split("_")

    position = partdata.coordinates
    velocity = partdata.velocities
    mass = getattr(partdata, mass_variable)

    radius = np.sqrt((position ** 2).sum(axis=1))
    mask = get_orientation_mask(
        radius, half_mass_radius, R200crit, Rvir, inner_aperture, outer_aperture
    )

    position = position[mask]
    velocity = velocity[mask]
    mass = mass[mask]
    radius = radius[mask]

    K = 0.5 * (mass[:, None] * velocity ** 2).sum()
    if K == 0.0:
        return 0.0, 0.0

    # np.cross does not preserve units, so we need to multiply them back in
    angular_momentum = (
        (mass[:, None] * np.cross(position, velocity)) * position.units * velocity.units
    )
    Lz = (angular_momentum * orientation_vector[None, :]).sum(axis=1)
    rdotL = (position * orientation_vector[None, :]).sum(axis=1)
    R2 = radius ** 2 - rdotL ** 2
    mask = (Lz > 0.0) & (R2 > 0.0)
    Kcorot = 0.5 * (Lz[mask] ** 2 / (mass[mask] * R2[mask])).sum()

    j = angular_momentum.sum(axis=0) / mass.sum()
    j.convert_to_units("kpc*km/s")
    j = np.sqrt((j ** 2).sum())

    return j, Kcorot / K


def plot_scaleheights(
    output_path: str,
    observational_data_path: str,
    name_list: List[str],
    all_galaxies_list: Union[List["AllGalaxyData"], List["GalaxyData"]],
) -> Dict:

    HI_scaleheight_filename = "HI_scaleheight.png"
    H2_scaleheight_filename = "H2_scaleheight.png"
    Stars_scaleheight_filename = "Stars_scaleheight.png"

    HI_scaleheight_active_filename = "HI_scaleheight_active.png"
    H2_scaleheight_active_filename = "H2_scaleheight_active.png"
    Stars_scaleheight_active_filename = "Stars_scaleheight_active.png"

    HI_scaleheight_passive_filename = "HI_scaleheight_passive.png"
    H2_scaleheight_passive_filename = "H2_scaleheight_passive.png"
    Stars_scaleheight_passive_filename = "Stars_scaleheight_passive.png"

    plots = {}

    fig_HI, ax_HI = pl.subplots(1, 1)
    fig_H2, ax_H2 = pl.subplots(1, 1)
    fig_stars, ax_stars = pl.subplots(1, 1)

    fig_HI_a, ax_HI_a = pl.subplots(1, 1)
    fig_H2_a, ax_H2_a = pl.subplots(1, 1)
    fig_stars_a, ax_stars_a = pl.subplots(1, 1)

    fig_HI_p, ax_HI_p = pl.subplots(1, 1)
    fig_H2_p, ax_H2_p = pl.subplots(1, 1)
    fig_stars_p, ax_stars_p = pl.subplots(1, 1)

    sim_lines = []
    sim_labels = []
    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        Mstar = unyt.unyt_array(data["stellar_mass"], unyt.Msun).in_base("galactic")
        Mstar.name = "Stellar Mass"
        H_HI = unyt.unyt_array(data["HI_scaleheight"], unyt.kpc)
        H_HI.name = "HI disk scale height"
        H_H2 = unyt.unyt_array(data["H2_scaleheight"], unyt.kpc)
        H_H2.name = "H2 disk scale height"
        H_star = unyt.unyt_array(data["stars_scaleheight"], unyt.kpc)
        H_star.name = "Stellar disk scale height (mass)"

        mask_HI = (np.isfinite(H_HI) & (H_HI.value > 0.))
        mask_H2 = (np.isfinite(H_H2) & (H_H2.value > 0.))
        mask_star = (np.isfinite(H_star) & (H_star.value > 0.))

        mask_HI_a = (np.isfinite(H_HI) & (H_HI.value > 0.)) & (data["is_active"] == True)
        mask_H2_a = (np.isfinite(H_H2) & (H_H2.value > 0.)) & (data["is_active"] == True)
        mask_star_a = (np.isfinite(H_star) & (H_star.value > 0.)) & (data["is_active"] == True)

        mask_HI_p = (np.isfinite(H_HI) & (H_HI.value > 0.)) & (data["is_active"] == False)
        mask_H2_p = (np.isfinite(H_H2) & (H_H2.value > 0.)) & (data["is_active"] == False)
        mask_star_p = (np.isfinite(H_star) & (H_star.value > 0.)) & (data["is_active"] == False)

        line = plot_data_on_axis(
            ax_HI, Mstar[mask_HI], H_HI[mask_HI], color=f"C{i}", plot_scatter=(len(name_list) == 1)
        )
        sim_lines.append(line)
        sim_labels.append(name)
        line = plot_data_on_axis(
            ax_H2, Mstar[mask_H2], H_H2[mask_H2], color=f"C{i}", plot_scatter=(len(name_list) == 1)
        )
        line = plot_data_on_axis(
            ax_stars, Mstar[mask_star], H_star[mask_star], color=f"C{i}", plot_scatter=(len(name_list) == 1)
        )

        # only add points if there is data
        if np.sum(mask_HI_a) > 0:
            line = plot_data_on_axis(
                ax_HI_a, Mstar[mask_HI_a], H_HI[mask_HI_a], color=f"C{i}", plot_scatter=(len(name_list) == 1)
            )
        if np.sum(mask_H2_a) > 0:
            line = plot_data_on_axis(
                ax_H2_a, Mstar[mask_H2_a], H_H2[mask_H2_a], color=f"C{i}", plot_scatter=(len(name_list) == 1)
            )
        if np.sum(mask_star_a) > 0:
            line = plot_data_on_axis(
                ax_stars_a, Mstar[mask_star_a], H_star[mask_star_a], color=f"C{i}", plot_scatter=(len(name_list) == 1)
            )

        if np.sum(mask_HI_p) > 0:
            line = plot_data_on_axis(
                ax_HI_p, Mstar[mask_HI_p], H_HI[mask_HI_p], color=f"C{i}", plot_scatter=(len(name_list) == 1)
            )
        if np.sum(mask_H2_p) > 0:
            line = plot_data_on_axis(
                ax_H2_p, Mstar[mask_H2_p], H_H2[mask_H2_p], color=f"C{i}", plot_scatter=(len(name_list) == 1)
            )
        if np.sum(mask_star_p) > 0:
            line = plot_data_on_axis(
                ax_stars_p, Mstar[mask_star_p], H_star[mask_star_p], color=f"C{i}", plot_scatter=(len(name_list) == 1)
            )


    ax_HI.set_title("HI disk thickness")
    ax_H2.set_title("H2 disk thickness")
    ax_stars.set_title("Stellar disk thickness (mass)")

    ax_HI_a.set_title("HI disk thickness (active)")
    ax_H2_a.set_title("H2 disk thickness (active)")
    ax_stars_a.set_title("Stellar disk thickness (mass) (active)")

    ax_HI_p.set_title("HI disk thickness (passive)")
    ax_H2_p.set_title("H2 disk thickness (passive)")
    ax_stars_p.set_title("Stellar disk thickness (mass) (passive)")

    for ax in [ax_HI, ax_H2, ax_stars, 
               ax_HI_a, ax_H2_a, ax_stars_a, 
               ax_HI_p, ax_H2_p, ax_stars_p ]:
        ax.grid(True)
        ax.set_xlim(1.0e6, 1.0e12)
        ax.set_ylim(0.01, 1.0e2)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
        # uncomment this if observational data is added
        # ax.legend(loc="lower right")
        ax.add_artist(sim_legend)

    fig_HI.savefig(f"{output_path}/{HI_scaleheight_filename}", dpi=300)
    pl.close(fig_HI)

    fig_H2.savefig(f"{output_path}/{H2_scaleheight_filename}", dpi=300)
    pl.close(fig_H2)

    fig_stars.savefig(f"{output_path}/{Stars_scaleheight_filename}", dpi=300)
    pl.close(fig_stars)

    fig_HI_a.savefig(f"{output_path}/{HI_scaleheight_active_filename}", dpi=300)
    pl.close(fig_HI_a)

    fig_H2_a.savefig(f"{output_path}/{H2_scaleheight_active_filename}", dpi=300)
    pl.close(fig_H2_a)

    fig_stars_a.savefig(f"{output_path}/{Stars_scaleheight_active_filename}", dpi=300)
    pl.close(fig_stars_a)

    fig_HI_p.savefig(f"{output_path}/{HI_scaleheight_passive_filename}", dpi=300)
    pl.close(fig_HI_p)

    fig_H2_p.savefig(f"{output_path}/{H2_scaleheight_passive_filename}", dpi=300)
    pl.close(fig_H2_p)

    fig_stars_p.savefig(f"{output_path}/{Stars_scaleheight_passive_filename}", dpi=300)
    pl.close(fig_stars_p)

    plots["Disk scale height"] = {
        HI_scaleheight_filename: {
            "title": "HI disk scale height",
            "caption": (
                "Scale height of the HI disk from fit to exponential profile"
                " of edge-on image within 2 times the 3D stellar half mass radius."
                " Half mass radius calculated from 50 kpc aperture."
                " All galaxies with a successful fit are shown."
            ),
        },
        H2_scaleheight_filename: {
            "title": "H2 disk scale height",
            "caption": (
                "Scale height of the H2 disk from fit to exponential profile"
                " of edge-on image within 2 times the 3D stellar half mass radius."
                " Half mass radius calculated from 50 kpc aperture."
                " All galaxies with a successful fit are shown."
            ),
        },
        Stars_scaleheight_filename: {
            "title": "Stellar disk scale height (mass)",
            "caption": (
                "Scale height of the H2 disk from fit to exponential profile"
                " of binned particle positions" 
                " within 2 times the 3D stellar half mass radius."
                " Half mass radius calculated from 50 kpc aperture."
                " All galaxies with a successful fit are shown."
            ),
        },
        HI_scaleheight_active_filename: {
            "title": "HI disk scale height (active)",
            "caption": (
                "Scale height of the HI disk from fit to exponential profile"
                " of edge-on image within 2 times the 3D stellar half mass radius."
                " Half mass radius calculated from 50 kpc aperture."
                " Only active galaxies with a successful fit are shown."
            ),
        },
        H2_scaleheight_active_filename: {
            "title": "H2 disk scale height (active)",
            "caption": (
                "Scale height of the H2 disk from fit to exponential profile"
                " of edge-on image within 2 times the 3D stellar half mass radius."
                " Half mass radius calculated from 50 kpc aperture."
                " Only active galaxies with a successful fit are shown."
            ),
        },
        Stars_scaleheight_active_filename: {
            "title": "Stellar disk scale height (mass) (active)",
            "caption": (
                "Scale height of the H2 disk from fit to exponential profile"
                " of binned particle positions"
                " within 2 times the 3D stellar half mass radius."
                " Half mass radius calculated from 50 kpc aperture."
                " Only active galaxies with a successful fit are shown."
            ),
        },
        HI_scaleheight_passive_filename: {
            "title": "HI disk scale height (passive)",
            "caption": (
                "Scale height of the HI disk from fit to exponential profile"
                " of edge-on image within 2 times the 3D stellar half mass radius."
                " Half mass radius calculated from 50 kpc aperture."
                " Only passive galaxies with a successful fit are shown."
            ),
        },
        H2_scaleheight_passive_filename: {
            "title": "H2 disk scale height (passive)",
            "caption": (
                "Scale height of the H2 disk from fit to exponential profile"
                " of edge-on image within 2 times the 3D stellar half mass radius."
                " Half mass radius calculated from 50 kpc aperture."
                " Only passive galaxies with a successful fit are shown."
            ),
        },
        Stars_scaleheight_passive_filename: {
            "title": "Stellar disk scale height (mass) (passive)",
            "caption": (
                "Scale height of the H2 disk from fit to exponential profile"
                " of binned particle positions"
                " within 2 times the 3D stellar half mass radius."
                " Half mass radius calculated from 50 kpc aperture."
                " Only passive galaxies with a successful fit are shown."
            ),  
        },  
    }

    return plots


def plot_morphology(
    output_path: str,
    observational_data_path: str,
    name_list: List[str],
    all_galaxies_list: Union[List["AllGalaxyData"], List["GalaxyData"]],
) -> Dict:
    """
    Create morphology related plots.

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

    plots = {}

    fig_gas, ax_gas = pl.subplots(1, 1)
    fig_star, ax_star = pl.subplots(1, 1)

    sim_lines = []
    sim_labels = []
    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        Mstar = unyt.unyt_array(data["stellar_mass"], unyt.Msun).in_base("galactic")
        Mstar.name = "Stellar Mass"
        jstar = unyt.unyt_array(data["stars_momentum"], unyt.kpc * unyt.km / unyt.s)
        jstar.name = "Stellar Specific Angular Momentum"
        jgas = unyt.unyt_array(data["gas_momentum"], unyt.kpc * unyt.km / unyt.s)
        jgas.name = "Gas Specific Angular Momentum"

        line = plot_data_on_axis(
            ax_star, Mstar, jstar, color=f"C{i}", plot_scatter=(len(name_list) == 1)
        )
        sim_lines.append(line)
        sim_labels.append(name)
        line = plot_data_on_axis(
            ax_gas, Mstar, jgas, color=f"C{i}", plot_scatter=(len(name_list) == 1)
        )

    ax_star.set_title("Stellar component")
    ax_gas.set_title("HI+H2 gas")
    for ax in [ax_gas, ax_star]:
        ax.grid(True)
        ax.set_xlim(1.0e6, 1.0e12)
        ax.set_ylim(0.1, 1.0e4)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
        # uncomment this if observational data is added
        # ax.legend(loc="lower right")
        ax.add_artist(sim_legend)

    jgas_filename = "specific_angular_momentum_gas.png"
    fig_gas.savefig(f"{output_path}/{jgas_filename}", dpi=300)
    pl.close(fig_gas)

    jstar_filename = "specific_angular_momentum_stars.png"
    fig_star.savefig(f"{output_path}/{jstar_filename}", dpi=300)
    pl.close(fig_star)

    plots["Specific angular momentum"] = {
        jstar_filename: {
            "title": "Specific angular momentum / Stars",
            "caption": (
                "Ratio between the total angular momentum of stars within 30 kpc"
                " of aperture divided by the total mass in stars."
            ),
        },
        jgas_filename: {
            "title": "Specific angular momentum / HI+H2 gas",
            "caption": (
                "Ratio between the total angular momentum of gas within 30 kpc"
                " of aperture divided by the total mass in gas."
            ),
        },
    }

    fig_gas, ax_gas = pl.subplots(1, 1)
    fig_star, ax_star = pl.subplots(1, 1)

    sim_lines = []
    sim_labels = []
    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        Mstar = unyt.unyt_array(data["stellar_mass"], unyt.Msun).in_base("galactic")
        Mstar.name = "Stellar Mass"
        kappa_star = unyt.unyt_array(data["stars_kappa_co"], unyt.dimensionless)
        kappa_star.name = "Stellar Kappa Corot"
        kappa_gas = unyt.unyt_array(data["gas_kappa_co"], unyt.dimensionless)
        kappa_gas.name = "Gas Kappa Corot"

        line = plot_data_on_axis(
            ax_star,
            Mstar,
            kappa_star,
            color=f"C{i}",
            plot_scatter=(len(name_list) == 1),
            log_y=False,
        )
        sim_lines.append(line)
        sim_labels.append(name)
        line = plot_data_on_axis(
            ax_gas,
            Mstar,
            kappa_gas,
            color=f"C{i}",
            plot_scatter=(len(name_list) == 1),
            log_y=False,
        )

    ax_star.set_title("Stellar component")
    ax_gas.set_title("HI+H2 gas")
    for ax in [ax_gas, ax_star]:
        ax.grid(True)
        ax.set_xlim(1.0e6, 1.0e12)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
        # uncomment this if observational data is added
        # ax.legend(loc="lower right")
        ax.add_artist(sim_legend)

    kappa_gas_filename = "kappa_corot_gas.png"
    fig_gas.savefig(f"{output_path}/{kappa_gas_filename}", dpi=300)
    pl.close(fig_gas)

    kappa_star_filename = "kappa_corot_stars.png"
    fig_star.savefig(f"{output_path}/{kappa_star_filename}", dpi=300)
    pl.close(fig_star)

    plots["Kappa corotation"] = {
        kappa_star_filename: {
            "title": "Kappa corotation / Stars",
            "caption": (
                "Kappa corotation is defined as the fraction of kinetic energy in"
                " a galaxy that is in ordered rotation. Note that the rotating"
                " contribution is calculated only for prograde rotation."
            ),
        },
        kappa_gas_filename: {
            "title": "Kappa corotation / HI+H2 gas",
            "caption": (
                "Kappa corotation is defined as the fraction of kinetic energy in"
                " a galaxy that is in ordered rotation. Note that the rotating"
                " contribution is calculated only for prograde rotation."
            ),
        },
    }

    fig_gas, ax_gas = pl.subplots(2, 3, figsize=(11.0, 7.0))
    fig_star, ax_star = pl.subplots(2, 3, figsize=(11.0, 7.0))

    sim_lines = []
    sim_labels = []
    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        Mstar = unyt.unyt_array(data["stellar_mass"], unyt.Msun).in_base("galactic")
        Mstar.name = "Stellar Mass"
        for j, (dname, dlabel) in enumerate(
            [("axis_ca", "c/a"), ("axis_cb", "c/b"), ("axis_ba", "b/a")]
        ):
            for k, type in enumerate(["reduced", "normal"]):
                ratio_star = unyt.unyt_array(
                    data[f"stars_{dname}_{type}"], unyt.dimensionless
                )
                ratio_star.name = f"Stellar {dlabel} ({type} tensor)"
                ratio_gas = unyt.unyt_array(
                    data[f"gas_{dname}_{type}"], unyt.dimensionless
                )
                ratio_gas.name = f"Gas {dlabel} ({type} tensor)"

                line = plot_data_on_axis(
                    ax_star[k][j],
                    Mstar,
                    ratio_star,
                    color=f"C{i}",
                    plot_scatter=(len(name_list) == 1),
                    log_y=False,
                )
                if j == 0 and k == 0:
                    sim_lines.append(line)
                    sim_labels.append(name)
                line = plot_data_on_axis(
                    ax_gas[k][j],
                    Mstar,
                    ratio_gas,
                    color=f"C{i}",
                    plot_scatter=(len(name_list) == 1),
                    log_y=False,
                )

    ax_star[0][1].set_title("Stellar component")
    ax_gas[0][1].set_title("HI+H2 gas")
    for ax in [*ax_gas.flatten(), *ax_star.flatten()]:
        ax.grid(True)
        ax.set_xlim(1.0e6, 1.0e12)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
        # uncomment this if observational data is added
        # ax.legend(loc="lower right")
        ax.add_artist(sim_legend)

    ratio_gas_filename = "axis_ratios_gas.png"
    fig_gas.savefig(f"{output_path}/{ratio_gas_filename}", dpi=300)
    pl.close(fig_gas)

    ratio_star_filename = "axis_ratios_stars.png"
    fig_star.savefig(f"{output_path}/{ratio_star_filename}", dpi=300)
    pl.close(fig_star)

    plots["Axis ratios"] = {
        ratio_star_filename: {
            "title": "Axis ratios / Stars",
            "caption": (
                "Axial ratios of galaxies, based on the stars. a, b and c"
                " (a >= b >= c) represent the lengths of the primary axes."
                " The axis lengths have been computed from the reduced moment"
                " of inertia tensor using the iterative scheme of Thob et al. (2018)."
            ),
        },
        ratio_gas_filename: {
            "title": "Axis ratios / HI+H2 gas",
            "caption": (
                "Axial ratios of galaxies, based on the neutral gas. a, b and c"
                " (a >= b >= c) represent the lengths of the primary axes. The axis"
                " lengths have been computed from the reduced moment of inertia tensor"
                " using the iterative scheme of Thob et al. (2018)."
            ),
        },
    }

    return plots
