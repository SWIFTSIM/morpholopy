import numpy as np
import unyt
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from .orientation import (
    get_orientation_mask,
    get_mass_position_velocity,
    get_mass_position_velocity_nomask,
    get_orientation_mask_radius,
)
from .plot import plot_data_on_axis
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl


def get_axis_lengths_tensor(
    galaxy_log,
    partdata,
    half_mass_radius,
    mass_variable="masses",
    reduced=True,
    iterative=True,
):

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
    galaxy_log, partdata, half_mass_radius, mass_variable="masses"
):
    return get_axis_lengths_tensor(
        galaxy_log,
        partdata,
        half_mass_radius,
        mass_variable,
        reduced=True,
        iterative=True,
    )


def get_axis_lengths_normal_tensor(
    galaxy_log, partdata, half_mass_radius, mass_variable="masses"
):
    return get_axis_lengths_tensor(
        galaxy_log,
        partdata,
        half_mass_radius,
        mass_variable,
        reduced=False,
        iterative=True,
    )


def get_kappa_corot(
    partdata,
    half_mass_radius,
    R200crit,
    Rvir,
    orientation_type,
    orientation_vector,
    mass_variable="masses",
):

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


def plot_morphology(output_path, observational_data_path, name_list, all_galaxies_list):

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
