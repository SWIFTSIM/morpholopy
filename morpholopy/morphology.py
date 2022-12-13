import numpy as np
import unyt
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from .orientation import (
    get_orientation_mask,
    get_mass_position_velocity,
    get_mass_position_velocity_nomask,
    get_orientation_mask_radius,
)


def get_new_axis_lengths(data, half_mass_radius, R200crit, Rvir, orientation_type):

    mass, position, velocity = get_mass_position_velocity(
        data, half_mass_radius, R200crit, Rvir, orientation_type
    )
    radius = np.sqrt((position ** 2).sum(axis=1))

    weight = unyt.unyt_array(np.zeros(mass.shape), mass.units / radius.units ** 2)
    weight[radius > 0.0] = (mass[radius > 0.0] / radius[radius > 0.0] ** 2).to(
        weight.units
    )

    if weight.sum() == 0.0:
        print("Total weight of 0, so not calculating axis lengths.")
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

    Ltype, _, outer_aperture, _ = orientation_type.split("_")
    all_mass, all_position, all_velocity = get_mass_position_velocity_nomask(
        data, Ltype
    )
    R2 = (
        get_orientation_mask_radius(half_mass_radius, R200crit, Rvir, outer_aperture)
        ** 2
    )

    c_a = axes[2] / axes[0]
    b_a = axes[1] / axes[0]
    old_c_a = 2.0 * c_a
    old_b_a = 2.0 * b_a
    loop = 0
    while (abs((c_a - old_c_a) / (c_a + old_c_a)) > 0.01) or (
        abs((b_a - old_b_a) / (b_a + old_b_a)) > 0.01
    ):
        loop += 1
        if loop == 100:
            print(
                f"Too many iterations (c_a: {old_c_a} - {c_a}, b_a: {old_b_a} - {b_a} ({Ltype}_{outer_aperture})!"
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
        velocity = all_velocity[mask]

        radius = np.sqrt((position ** 2).sum(axis=1))

        weight = unyt.unyt_array(np.zeros(mass.shape), mass.units / radius.units ** 2)
        weight[radius > 0.0] = (mass[radius > 0.0] / radius[radius > 0.0] ** 2).to(
            weight.units
        )

        if weight.sum() == 0.0:
            print(
                "Total weight of 0, so not calculating axis lengths. Using last value that worked."
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

    return axes, basis[2]


def get_kappa_corot(
    partdata, half_mass_radius, R200crit, Rvir, orientation_type, orientation_vector
):

    _, inner_aperture, outer_aperture, clipping = orientation_type.split("_")

    position = partdata.coordinates
    velocity = partdata.velocities
    mass = partdata.masses

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
        return 0.0

    # np.cross does not preserve units, so we need to multiply them back in
    angular_momentum = (
        (mass[:, None] * np.cross(position, velocity)) * position.units * velocity.units
    )
    Lz = (angular_momentum * orientation_vector[None, :]).sum(axis=1)
    rdotL = (position * orientation_vector[None, :]).sum(axis=1)
    R2 = radius ** 2 - rdotL ** 2
    mask = (Lz > 0.0) & (R2 > 0.0)
    Kcorot = 0.5 * (Lz[mask] ** 2 / (mass[mask] * R2[mask])).sum()

    return angular_momentum.sum(axis=0), Kcorot / K
