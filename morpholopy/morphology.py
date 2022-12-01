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


def get_axis_lengths(partdata, half_mass_radius, R200crit, Rvir, orientation_type):

    _, aperture, clipping = orientation_type.split("_")

    position = partdata.coordinates
    velocity = partdata.velocities
    mass = partdata.masses

    radius = np.sqrt((position ** 2).sum(axis=1))
    mask = get_orientation_mask(radius, half_mass_radius, R200crit, Rvir, aperture)

    position = position[mask]
    velocity = velocity[mask]
    mass = mass[mask]
    radius = radius[mask]

    if clipping == "0sigma":
        pass

    weight = mass / radius ** 2

    Itensor = (weight[:, None, None] / weight.sum()) * np.ones((weight.shape[0], 3, 3))
    # Note: unyt currently ignores the position units in the *=
    # i.e. Itensor is dimensionless throughout (even though it should not be)
    for i in range(3):
        for j in range(3):
            Itensor[:, i, j] *= position[:, i] * position[:, j]
    Itensor = Itensor.sum(axis=0)

    # linalg.eigenvals cannot deal with units anyway, so we have to add them
    # back in
    axes = np.linalg.eigvals(Itensor).real
    axes = np.clip(axes, 0.0, None)
    axes = np.sqrt(axes) * position.units

    # sort the axes from long to short
    axes.sort()
    axes = axes[::-1]

    return axes


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

    if clipping == "0sigma":
        pass

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

    return Kcorot / K


def get_angular_momentum_vector(rparticles, vparticles, mparticles, rgalaxy, vgalaxy):

    r = rparticles - rgalaxy
    v = vparticles - vgalaxy
    m = mparticles.copy()

    d = np.linalg.norm(r, axis=1)

    # mask out the innermost x% of star particles
    dmin = np.quantile(d, 0.3, axis=0)
    dmax = np.quantile(d, 0.5, axis=0)
    m[d < dmin] = 0.0
    m[d > dmax] = 0.0

    L = np.cross(r, v)
    L[:, 0] *= m
    L[:, 1] *= m
    L[:, 2] *= m

    Ltotal = np.sum(L, axis=0)

    Ltotal = Ltotal / np.linalg.norm(Ltotal)

    face_on_rotation_matrix = rotation_matrix_from_vector(Ltotal)
    edge_on_rotation_matrix = rotation_matrix_from_vector(Ltotal, axis="y")

    return face_on_rotation_matrix, edge_on_rotation_matrix


def AxialRatios(rs, ms):
    """
    rs - CoM subtracted positions of *selected* particles in galactic units
    ms - *selected* particle masses in galactic units
    """
    radius = np.linalg.norm(rs[:, :3], axis=1)
    rs = rs[radius > 0, :]
    ms = ms[radius > 0]
    rs2 = rs ** 2

    # construct MoI tensor
    I_xx = (
        rs2[:, [1, 2]].sum(axis=-1) / abs((rs2[:, [1, 2]].sum(axis=-1)) ** 0.5)
    ) * ms
    I_xx = I_xx[np.isnan(I_xx) == False]  # remove nans
    I_xx = I_xx.sum()
    I_yy = (
        rs2[:, [0, 2]].sum(axis=-1) / abs((rs2[:, [0, 2]].sum(axis=-1)) ** 0.5)
    ) * ms
    I_yy = I_yy[np.isnan(I_yy) == False]
    I_yy = I_yy.sum()
    I_zz = (
        rs2[:, [0, 1]].sum(axis=-1) / abs((rs2[:, [0, 1]].sum(axis=-1)) ** 0.5)
    ) * ms
    I_zz = I_zz[np.isnan(I_zz) == False]
    I_zz = I_zz.sum()
    I_xy = -((rs[:, 0] * rs[:, 1] / abs(rs[:, 0] * rs[:, 1]) ** 0.5) * ms)
    I_xy = I_xy[np.isnan(I_xy) == False]
    I_xy = I_xy.sum()
    I_xz = -((rs[:, 0] * rs[:, 2] / abs(rs[:, 0] * rs[:, 2]) ** 0.5) * ms)
    I_xz = I_xz[np.isnan(I_xz) == False]
    I_xz = I_xz.sum()
    I_yz = -((rs[:, 1] * rs[:, 2] / abs(rs[:, 1] * rs[:, 2]) ** 0.5) * ms)
    I_yz = I_yz[np.isnan(I_yz) == False]
    I_yz = I_yz.sum()
    I = np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])

    # Get and order eigenvalues
    W, V = np.linalg.eig(I)
    W1, W2, W3 = np.sort(W)[::-1]

    # compute axes (unnormalised as we don't need absolute values)
    a = np.sqrt(np.abs(W1 + W2 - W3))
    b = np.sqrt(np.abs(W1 + W3 - W2))
    c = np.sqrt(np.abs(W2 + W3 - W1))

    return c / a, c / b, b / a


def calculate_morphology(
    coordinates, velocities, mass, box, galaxy_center, galaxy_velocity
):

    coordinates = coordinates.copy()
    velocities = velocities.copy()

    # recenter galaxy
    coordinates[:, :] -= galaxy_center[None, :]
    coordinates[:, :] += 0.5 * box[None, :]
    coordinates[:, :] %= box[None, :]
    coordinates[:, :] -= 0.5 * box[None, :]

    velocities[:, :] -= galaxy_velocity[None, :]

    # we cannot use np.linalg.norm, since that does not conserve units
    radius = unyt.array.unorm(coordinates, axis=1)
    aperture_mask = radius < 30.0 * unyt.kpc

    Mstar = mass[aperture_mask].sum()
    dvVmass = (mass[aperture_mask, None] * velocities[aperture_mask, :]).sum(
        axis=0
    ) / Mstar
    velocities[:, :] -= dvVmass

    part_momentum = unyt.array.ucross(
        coordinates[aperture_mask], velocities[aperture_mask]
    )
    momentum = (mass[aperture_mask, None] * part_momentum).sum(axis=0)
    specific_momentum = unyt.array.unorm(momentum / Mstar)

    momentum /= unyt.array.unorm(momentum)

    momentum_z = (momentum * part_momentum).sum(axis=1)
    cyldistances = np.sqrt(
        np.abs(
            radius[aperture_mask] ** 2
            - (momentum * coordinates[aperture_mask]).sum(axis=1) ** 2
        )
    )
    if (cyldistances > 0.0).sum() > 0.0:
        cylmin = cyldistances[cyldistances > 0.0].min()
        cyldistances[cyldistances == 0.0] = cylmin
        vrots = momentum_z / cyldistances
    else:
        vrots = momentum_z

    Mvrot2 = (mass[aperture_mask] * vrots ** 2)[vrots > 0.0].sum()
    kappa_co = (
        Mvrot2
        / (mass[aperture_mask] * (velocities[aperture_mask] ** 2).sum(axis=1)).sum()
    )
    axis_1, axis_2, axis_3 = AxialRatios(
        coordinates[aperture_mask], mass[aperture_mask]
    )

    specific_momentum.convert_to_units("kpc*km/s")

    return kappa_co, specific_momentum, axis_1, axis_2, axis_3
