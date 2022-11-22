import numpy as np
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.objects import cosmo_array


def get_orientation_matrices(
    data, galaxy_position, galaxy_velocity, half_mass_radius, orientation_type
):

    Ltype, aperture, clipping = orientation_type.split("_")

    position = None
    velocity = None
    mass = None
    if Ltype == "stars":
        position = data.stars.coordinates
        velocity = data.stars.velocities
        mass = data.stars.masses
    else:
        raise RuntimeError(f"Unknown Ltype: {Ltype}!")

    radius = np.sqrt((position ** 2).sum(axis=1))
    if aperture == "R0.5":
        mask = radius < half_mass_radius

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
