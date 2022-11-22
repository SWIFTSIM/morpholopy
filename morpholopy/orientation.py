import numpy as np
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.objects import cosmo_array
import unyt


def get_orientation_mask(radius, half_mass_radius, R200crit, aperture):
    mask = None
    if aperture == "R0.5":
        mask = radius < half_mass_radius
    elif aperture == "2xR0.5":
        mask = radius < 2.0 * half_mass_radius
    elif aperture == "4xR0.5":
        mask = radius < 4.0 * half_mass_radius
    elif aperture == "50kpc":
        mask = radius < cosmo_array(
            50.0 * unyt.kpc, comoving=False, cosmo_factor=radius.cosmo_factor
        )
    elif aperture == "R200crit":
        mask = radius < R200crit
    else:
        raise RuntimeError(f"Unknown aperture: {aperture}!")
    return mask


def get_orientation_matrices(
    data, galaxy_position, galaxy_velocity, half_mass_radius, R200crit, orientation_type
):

    Ltype, aperture, clipping = orientation_type.split("_")

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

    radius = np.sqrt((position ** 2).sum(axis=1))
    mask = get_orientation_mask(radius, half_mass_radius, R200crit, aperture)

    position = position[mask]
    velocity = velocity[mask]
    mass = mass[mask]

    if clipping == "0sigma":
        pass

    angular_momentum = (mass[:, None] * np.cross(position, velocity)).sum(axis=0)
    angular_momentum /= np.sqrt((angular_momentum ** 2).sum())

    angular_momentum = angular_momentum.value

    face_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum)
    edge_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum, axis="y")

    return angular_momentum, face_on_rotation_matrix, edge_on_rotation_matrix
