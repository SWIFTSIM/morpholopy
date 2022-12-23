#!/usr/bin/env python3

"""
orientation.py

Functions related to the orientation of galaxies.

The orientation is determined exactly once for each galaxy,
based on a strategy encoded in a string with the following format:
  <component type>_<inner mask radius>_<outer mask radius>_<sigma clipping>
where:
 - component type can be stars, gas, ISM, HI, baryons
 - inner mask radius can be 0xR0.5, 0.5R0.5
 - outer mask radius can be R0.5, 2xR0.5, 4xR0.5, 50kpc, R200crit, 0.1Rvir
 - sigma clipping can be 0sigma (no clipping), 1sigma, 2sigma

This file contains a number of functions that can be called in other
parts of the pipeline to make particle selections that are consistent
with this string.

The orientation itself is given by the total angular momentum vector of
the selected component, after applying all the masks.
"""

import numpy as np
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.objects import cosmo_array
import unyt

from typing import Tuple
from numpy.typing import NDArray
from swiftsimio import SWIFTDataset


def get_orientation_mask(
    radius: unyt.unyt_array,
    half_mass_radius: unyt.unyt_quantity,
    R200crit: unyt.unyt_quantity,
    Rvir: unyt.unyt_quantity,
    inner_aperture: str,
    outer_aperture: str,
) -> NDArray[bool]:
    """
    Mask out the given particle radii by applying a mask consistent
    with the orientation method.

    Returns the resulting mask.
    """

    mask = None
    if inner_aperture == "0xR0.5":
        mask = np.ones(radius.shape, dtype=bool)
    elif inner_aperture == "0.5xR0.5":
        mask = radius >= 0.5 * half_mass_radius
    else:
        raise RuntimeError(f"Unknown inner aperture: {inner_aperture}!")

    if outer_aperture == "R0.5":
        mask &= radius < half_mass_radius
    elif outer_aperture == "2xR0.5":
        mask &= radius < 2.0 * half_mass_radius
    elif outer_aperture == "4xR0.5":
        mask &= radius < 4.0 * half_mass_radius
    elif outer_aperture == "50kpc":
        mask &= radius < cosmo_array(
            50.0 * unyt.kpc, comoving=False, cosmo_factor=radius.cosmo_factor
        )
    elif outer_aperture == "R200crit":
        mask &= radius < R200crit
    elif outer_aperture == "0.1Rvir":
        mask &= radius < 0.1 * Rvir
    else:
        raise RuntimeError(f"Unknown outer aperture: {outer_aperture}!")

    return mask


def get_mass_position_velocity_nomask(
    data: SWIFTDataset, Ltype: str
) -> Tuple[unyt.unyt_array, unyt.unyt_array, unyt.unyt_array]:
    """
    Get the masses, positions and velocities of all particles
    that should be used for the orientation calculation.

    Depends on the first part of the orientation method string.
    Does not apply any spatial masking.
    """

    position = None
    velocity = None
    mass = None
    if Ltype == "stars":
        position = data.stars.coordinates
        velocity = data.stars.velocities
        mass = data.stars.masses
    elif Ltype == "gas":
        position = data.gas.coordinates
        velocity = data.gas.velocities
        mass = data.gas.masses
    elif Ltype == "ISM":
        nH = data.gas.densities / unyt.mp
        refnH = cosmo_array(
            0.1 * unyt.cm ** (-3), comoving=False, cosmo_factor=nH.cosmo_factor
        )
        refT = cosmo_array(
            10 ** 4.5 * unyt.K,
            comoving=False,
            cosmo_factor=data.gas.temperatures.cosmo_factor,
        )
        mask = (nH > refnH) & (data.gas.temperatures < refT)
        position = data.gas.coordinates[mask]
        velocity = data.gas.velocities[mask]
        mass = data.gas.masses[mask]
    elif Ltype == "HI":
        position = data.gas.coordinates
        velocity = data.gas.velocities
        mass = (
            data.gas.masses
            * data.gas.species_fractions.HI
            * data.gas.element_mass_fractions.hydrogen
        )
    elif Ltype == "baryons":
        position = cosmo_array(
            np.concatenate([data.gas.coordinates, data.stars.coordinates]),
            comoving=False,
            cosmo_factor=data.gas.coordinates.cosmo_factor,
        )
        velocity = cosmo_array(
            np.concatenate([data.gas.velocities, data.stars.velocities]),
            comoving=False,
            cosmo_factor=data.gas.velocities.cosmo_factor,
        )
        mass = cosmo_array(
            np.concatenate([data.gas.masses, data.stars.masses]),
            comoving=False,
            cosmo_factor=data.gas.masses.cosmo_factor,
        )
    else:
        raise RuntimeError(f"Unknown Ltype: {Ltype}!")

    return mass, position, velocity


def get_mass_position_velocity(
    data: SWIFTDataset,
    half_mass_radius: unyt.unyt_quantity,
    R200crit: unyt.unyt_quantity,
    Rvir: unyt.unyt_quantity,
    orientation_type: str,
) -> Tuple[unyt.unyt_array, unyt.unyt_array, unyt.unyt_array]:
    """
    Get the masses, positions and velocities of all particles
    that should be used for the orientation calculation.

    Also applies all spatial masking and sigma clipping.
    """

    Ltype, inner_aperture, outer_aperture, clipping = orientation_type.split("_")

    mass, position, velocity = get_mass_position_velocity_nomask(data, Ltype)

    radius = np.sqrt((position ** 2).sum(axis=1))
    mask = get_orientation_mask(
        radius, half_mass_radius, R200crit, Rvir, inner_aperture, outer_aperture
    )

    position = position[mask]
    velocity = velocity[mask]
    mass = mass[mask]

    vcom = ((mass[:, None] / mass.sum()) * velocity).sum(axis=0)
    velocity -= vcom[None, :]

    if clipping == "0sigma":
        pass
    elif clipping == "1sigma" or clipping == "2sigma":
        angular_momentum = mass[:, None] * np.cross(position, velocity)
        Lnorm = np.sqrt((angular_momentum ** 2).sum(axis=1))
        Lmean = Lnorm.mean()
        Lstd = Lnorm.std()
        mask = None
        if clipping == "1sigma":
            mask = Lnorm <= Lmean + Lstd
        elif clipping == "2sigma":
            mask = Lnorm <= Lmean + 2.0 * Lstd
        mass = mass[mask]
        position = position[mask]
        velocity = velocity[mask]
    else:
        raise RuntimeError(f"Unknown clipping: {clipping}!")

    return mass, position, velocity


def get_orientation_matrices(
    data: SWIFTDataset,
    half_mass_radius: unyt.unyt_quantity,
    R200crit: unyt.unyt_quantity,
    Rvir: unyt.unyt_quantity,
    orientation_type: str,
) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
    """
    Get the orientation of a galaxy based on the given particles and
    orientation method string.

    Returns the direction of the z axis (direction of total angular
    momentum) and rotation matrices that can be used to produce
    face-on and edge-on projections using swiftsimio projection
    functions.
    """

    mass, position, velocity = get_mass_position_velocity(
        data, half_mass_radius, R200crit, Rvir, orientation_type
    )

    angular_momentum = (mass[:, None] * np.cross(position, velocity)).sum(axis=0)
    angular_momentum_norm = np.sqrt((angular_momentum ** 2).sum())
    if angular_momentum_norm > 0.0:
        angular_momentum /= angular_momentum_norm
        angular_momentum = angular_momentum.value
    else:
        # choose a random direction, i.e. the x axis
        # note that we cannot choose the y or z axis, since that would cause
        # problems in rotation_matrix_from_vector()
        # (the z axis is the default rotation axis, the y axis is our chosen
        # axis of rotation for the edge on view)
        angular_momentum = np.array([1.0, 0.0, 0.0])

    face_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum)
    edge_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum, axis="y")

    return angular_momentum, face_on_rotation_matrix, edge_on_rotation_matrix
