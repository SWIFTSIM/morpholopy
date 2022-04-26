import numpy as np
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector


def calculate_integrated_quantities(
    gas_coordinates, mass_variable, SFR, ang_momentum, radius
):

    face_on_rotation_matrix = rotation_matrix_from_vector(ang_momentum)

    x, y, _ = np.matmul(face_on_rotation_matrix, gas_coordinates.T)
    r = np.sqrt(x ** 2 + y ** 2)
    select = r <= radius

    surface = np.pi * radius ** 2

    # If we have gas within rhalfMs
    if len(mass_variable) > 0:
        Sigma_gas = np.log10(np.sum(mass_variable) / surface) - 6.0  # Msun / pc^2

        sfr = SFR[SFR > 0.0]
        Sigma_SFR = np.log10(np.sum(sfr) / surface)  # Msun / yr / kpc^2

    else:
        Sigma_gas = -6.5
        Sigma_SFR = -6.5

    return Sigma_gas, Sigma_SFR


def calculate_surface_densities(
    gas_coordinates, gas_HI, gas_H2, gas_sfr, angular_momentum, half_mass_radius_star
):

    Sigma_H2, Sigma_SFR_H2 = calculate_integrated_quantities(
        gas_coordinates, gas_H2, gas_sfr, angular_momentum, half_mass_radius_star
    )
    Sigma_gas, Sigma_SFR = calculate_integrated_quantities(
        gas_coordinates,
        gas_H2 + gas_HI,
        gas_sfr,
        angular_momentum,
        half_mass_radius_star,
    )

    return Sigma_H2, Sigma_gas, Sigma_SFR
