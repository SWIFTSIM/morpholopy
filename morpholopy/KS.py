import numpy as np
import swiftsimio as sw
import unyt
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl
from velociraptor.observations import load_observations
import glob

make_plots = False
Zsolar = 0.0134


def calculate_surface_densities_grid(
    data, face_on_rmatrix, gas_mask, index, resolution=128
):

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


def calculate_spatially_resolved_KS(data, face_on_rmatrix, gas_mask, index):
    image_diameter = 60.0 * unyt.kpc
    pixel_size = 0.75 * unyt.kpc
    resolution = int((image_diameter / pixel_size).value) + 1
    images = calculate_surface_densities_grid(
        data, face_on_rmatrix, gas_mask, index, resolution
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
    )


def calculate_azimuthally_averaged_KS(data, face_on_rmatrix, gas_mask, index):
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


def calculate_integrated_surface_densities(data, face_on_rmatrix, gas_mask, radius):

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
    output_path, observational_data_path, name_list, all_galaxies_list
):

    plots = {}

    fig_neut, ax_neut = pl.subplots(1, 1)
    fig_at, ax_at = pl.subplots(1, 1)
    fig_mol, ax_mol = pl.subplots(1, 1)

    markers = ["o", "s", "^", "D", "v"]
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
                label=name,
            )
            cs_at = ax_at.scatter(
                sigma_HI,
                sigma_SFR,
                c=Mstar,
                norm=matplotlib.colors.LogNorm(vmin=1.0e6, vmax=1.0e12),
                marker=marker,
                label=name,
            )
            cs_mol = ax_mol.scatter(
                sigma_H2,
                sigma_SFR,
                c=Mstar,
                norm=matplotlib.colors.LogNorm(vmin=1.0e6, vmax=1.0e12),
                marker=marker,
                label=name,
            )

    fig_neut.colorbar(cs_neut, label=f"{Mstar.name} [${Mstar.units.latex_repr}$]")
    fig_at.colorbar(cs_at, label=f"{Mstar.name} [${Mstar.units.latex_repr}$]")
    fig_mol.colorbar(cs_mol, label=f"{Mstar.name} [${Mstar.units.latex_repr}$]")

    for ax in [ax_neut, ax_at, ax_mol]:
        ax.grid(True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0e-1, 1.0e4)
        ax.set_ylim(1.0e-6, 1.0e1)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        ax.legend(loc="best")

    neut_filename = "integrated_KS_neutral.png"
    fig_neut.savefig(f"{output_path}/{neut_filename}", dpi=300)
    pl.close(fig_neut)

    at_filename = "integrated_KS_atomic.png"
    fig_at.savefig(f"{output_path}/{at_filename}", dpi=300)
    pl.close(fig_at)

    mol_filename = "integrated_KS_molecular.png"
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
        for Zmask, linestyle in [
            ("all", "-"),
            ("Zm1", "--"),
            ("Z0", ":"),
            ("Zp1", "-."),
        ]:
            med_neut = data.medians[f"sigma_neutral_SFR_azimuthal_{Zmask}"]
            med_at = data.medians[f"sigma_HI_SFR_azimuthal_{Zmask}"]
            med_mol = data.medians[f"sigma_H2_SFR_azimuthal_{Zmask}"]

            x_neut = unyt.unyt_array(med_neut["x centers"], med_neut["x units"])
            x_neut.name = "Neutral Gas Surface Density"
            y_neut = unyt.unyt_array(med_neut["y values"], med_neut["y units"])
            y_neut.name = "Star Formation Rate Surface Density"
            x_at = unyt.unyt_array(med_at["x centers"], med_at["x units"])
            x_at.name = "Atomic Gas Surface Density"
            y_at = unyt.unyt_array(med_at["y values"], med_at["y units"])
            y_at.name = "Star Formation Rate Surface Density"
            x_mol = unyt.unyt_array(med_mol["x centers"], med_mol["x units"])
            x_mol.name = "Molecular Gas Surface Density"
            y_mol = unyt.unyt_array(med_mol["y values"], med_mol["y units"])
            y_mol.name = "Star Formation Rate Surface Density"

            with unyt.matplotlib_support:
                ax_neut.loglog(x_neut, y_neut, linestyle, color=f"C{i}")
                ax_at.loglog(x_at, y_at, linestyle, color=f"C{i}")
                ax_mol.loglog(x_mol, y_mol, linestyle, color=f"C{i}")

    for dataname, ax in [
        ("SpatiallyResolvedNeutralKSRelation", ax_neut),
        ("SpatiallyResolvedMolecularKSRelation", ax_mol),
    ]:
        observational_data = load_observations(
            sorted(glob.glob(f"{observational_data_path}/{dataname}/*.hdf5"))
        )
        with unyt.matplotlib_support:
            for obs_data in observational_data:
                obs_data.plot_on_axes(ax)

    for ax in [ax_neut, ax_at, ax_mol]:
        sim_lines = []
        sim_labels = []
        for i, (name, _) in enumerate(zip(name_list, all_galaxies_list)):
            sim_lines.append(ax.plot([], [], "-", color=f"C{i}")[0])
            sim_labels.append(name)
        for Zlabel, linestyle in [
            ("all", "-"),
            ("$\\log_{10} Z/Z_\\odot = -1$", "--"),
            ("$\\log_{10} Z/Z_\\odot = 0$", ":"),
            ("$\\log_{10} Z/Z_\\odot = 1$", "-."),
        ]:
            sim_lines.append(ax.plot([], [], linestyle, color="k")[0])
            sim_labels.append(Zlabel)
        ax.grid(True)
        ax.set_xlim(1.0e-1, 1.0e4)
        ax.set_ylim(1.0e-6, 1.0e1)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
        ax.legend(loc="lower right")
        ax.add_artist(sim_legend)

    neut_filename = "azimuthal_KS_neutral.png"
    fig_neut.savefig(f"{output_path}/{neut_filename}", dpi=300)
    pl.close(fig_neut)

    at_filename = "azimuthal_KS_atomic.png"
    fig_at.savefig(f"{output_path}/{at_filename}", dpi=300)
    pl.close(fig_at)

    mol_filename = "azimuthal_KS_molecular.png"
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
                " Coloured lines show in the median relations considering only"
                " cells with fixed metallicity (as indicated in the legends)."
                " The grey solid line shows the median relation for all pixels,"
                " whereas the black solid line shows the relation only for pixels"
                " that have SFR surface density >0."
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
                " Coloured lines show in the median relations considering only"
                " cells with fixed metallicity (as indicated in the legends)."
                " The grey solid line shows the median relation for all pixels,"
                " whereas the black solid line shows the relation only for pixels"
                " that have SFR surface density >0."
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
                " Coloured lines show in the median relations considering only"
                " cells with fixed metallicity (as indicated in the legends)."
                " The grey solid line shows the median relation for all pixels,"
                " whereas the black solid line shows the relation only for pixels"
                " that have SFR surface density >0."
            ),
        },
    }

    fig_neut, ax_neut = pl.subplots(1, 1)
    fig_at, ax_at = pl.subplots(1, 1)
    fig_mol, ax_mol = pl.subplots(1, 1)

    for i, (_, data) in enumerate(zip(name_list, all_galaxies_list)):
        for Zmask, linestyle in [
            ("all", "-"),
            ("Zm1", "--"),
            ("Z0", ":"),
            ("Zp1", "-."),
        ]:
            med_neut = data.medians[f"sigma_neutral_SFR_spatial_{Zmask}"]
            med_at = data.medians[f"sigma_HI_SFR_spatial_{Zmask}"]
            med_mol = data.medians[f"sigma_H2_SFR_spatial_{Zmask}"]

            x_neut = unyt.unyt_array(med_neut["x centers"], med_neut["x units"])
            x_neut.name = "Neutral Gas Surface Density"
            y_neut = unyt.unyt_array(med_neut["y values"], med_neut["y units"])
            y_neut.name = "Star Formation Rate Surface Density"
            x_at = unyt.unyt_array(med_at["x centers"], med_at["x units"])
            x_at.name = "Atomic Gas Surface Density"
            y_at = unyt.unyt_array(med_at["y values"], med_at["y units"])
            y_at.name = "Star Formation Rate Surface Density"
            x_mol = unyt.unyt_array(med_mol["x centers"], med_mol["x units"])
            x_mol.name = "Molecular Gas Surface Density"
            y_mol = unyt.unyt_array(med_mol["y values"], med_mol["y units"])
            y_mol.name = "Star Formation Rate Surface Density"

            with unyt.matplotlib_support:
                ax_neut.loglog(x_neut, y_neut, linestyle, color=f"C{i}")
                ax_at.loglog(x_at, y_at, linestyle, color=f"C{i}")
                ax_mol.loglog(x_mol, y_mol, linestyle, color=f"C{i}")

    for dataname, ax in [
        ("SpatiallyResolvedNeutralKSRelation", ax_neut),
        ("SpatiallyResolvedMolecularKSRelation", ax_mol),
    ]:
        observational_data = load_observations(
            sorted(glob.glob(f"{observational_data_path}/{dataname}/*.hdf5"))
        )
        with unyt.matplotlib_support:
            for obs_data in observational_data:
                obs_data.plot_on_axes(ax)

    for ax in [ax_neut, ax_at, ax_mol]:
        sim_lines = []
        sim_labels = []
        for i, (name, _) in enumerate(zip(name_list, all_galaxies_list)):
            sim_lines.append(ax.plot([], [], "-", color=f"C{i}")[0])
            sim_labels.append(name)
        for Zlabel, linestyle in [
            ("all", "-"),
            ("$\\log_{10} Z/Z_\\odot = -1$", "--"),
            ("$\\log_{10} Z/Z_\\odot = 0$", ":"),
            ("$\\log_{10} Z/Z_\\odot = 1$", "-."),
        ]:
            sim_lines.append(ax.plot([], [], linestyle, color="k")[0])
            sim_labels.append(Zlabel)
        ax.grid(True)
        ax.set_xlim(1.0e-1, 1.0e4)
        ax.set_ylim(1.0e-6, 1.0e1)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        sim_legend = ax.legend(sim_lines, sim_labels, loc="upper left")
        ax.legend(loc="lower right")
        ax.add_artist(sim_legend)

    neut_filename = "spatial_KS_neutral.png"
    fig_neut.savefig(f"{output_path}/{neut_filename}", dpi=300)
    pl.close(fig_neut)

    at_filename = "spatial_KS_atomic.png"
    fig_at.savefig(f"{output_path}/{at_filename}", dpi=300)
    pl.close(fig_at)

    mol_filename = "spatial_KS_molecular.png"
    fig_mol.savefig(f"{output_path}/{mol_filename}", dpi=300)
    pl.close(fig_mol)

    plots["Combined surface densities"].update(
        {
            neut_filename: {
                "title": "Neutral KS relation (spatially averaged)",
                "caption": (
                    "Combined spatially resolved measurements from N most massive"
                    " individual galaxies, coloured by the mean metallicity of the"
                    " resolved pixel. The X axis shows the surface density of neutral"
                    " gas and the Y axis shows the star formation rate surface density."
                    " The surface densities were calculated using the grid method with"
                    " a pixel size of 750pc. Coloured lines show in the median relations"
                    " considering only cells with fixed metallicity (as indicated in"
                    " the legends). The grey solid line shows the median relation for"
                    " all pixels, whereas the black solid line shows the relation only"
                    " for pixels that have SFR surface density >0."
                ),
            },
            at_filename: {
                "title": "Atomic KS relation (spatially averaged)",
                "caption": (
                    "Combined spatially resolved measurements from N most massive"
                    " individual galaxies, coloured by the mean metallicity of the"
                    " resolved pixel. The X axis shows the surface density of"
                    " atomic gas and the Y axis shows the star formation rate"
                    " surface density. The surface densities were calculated using"
                    " the grid method with a pixel size of 750pc. Coloured lines"
                    " show in the median relations considering only cells with fixed"
                    " metallicity (as indicated in the legends). The grey solid line"
                    " shows the median relation for all pixels, whereas the black"
                    " solid line shows the relation only for pixels that have SFR"
                    " surface density >0."
                ),
            },
            mol_filename: {
                "title": "Molecular KS relation (spatially averaged)",
                "caption": (
                    "Combined spatially resolved measurements from N most massive"
                    " individual galaxies, coloured by the mean metallicity of the"
                    " resolved pixel. The X axis shows the surface density of"
                    " molecular gas and the Y axis shows the star formation rate"
                    " surface density. The surface densities were calculated using"
                    " the grid method with a pixel size of 750pc. Coloured lines"
                    " show in the median relations considering only cells with fixed"
                    " metallicity (as indicated in the legends). The grey solid line"
                    " shows the median relation for all pixels, whereas the black"
                    " solid line shows the relation only for pixels that have SFR"
                    " surface density >0."
                ),
            },
        }
    )

    fig_neut, ax_neut = pl.subplots(1, 1)
    fig_at, ax_at = pl.subplots(1, 1)
    fig_mol, ax_mol = pl.subplots(1, 1)

    for i, (name, data) in enumerate(zip(name_list, all_galaxies_list)):
        for Zmask, linestyle in [
            ("all", "-"),
            ("Zm1", "--"),
            ("Z0", ":"),
            ("Zp1", "-."),
        ]:
            med_neut = data.medians[f"sigma_neutral_tgas_spatial_{Zmask}"]
            med_at = data.medians[f"sigma_HI_tgas_spatial_{Zmask}"]
            med_mol = data.medians[f"sigma_H2_tgas_spatial_{Zmask}"]

            x_neut = unyt.unyt_array(med_neut["x centers"], med_neut["x units"])
            x_neut.name = "Neutral Gas Surface Density"
            y_neut = unyt.unyt_array(med_neut["y values"], med_neut["y units"])
            y_neut.name = "Neutral Gas Depletion Time"
            x_at = unyt.unyt_array(med_at["x centers"], med_at["x units"])
            x_at.name = "Atomic Gas Surface Density"
            y_at = unyt.unyt_array(med_at["y values"], med_at["y units"])
            y_at.name = "Atomic Gas Depletion Time"
            x_mol = unyt.unyt_array(med_mol["x centers"], med_mol["x units"])
            x_mol.name = "Molecular Gas Surface Density"
            y_mol = unyt.unyt_array(med_mol["y values"], med_mol["y units"])
            y_mol.name = "Molecular Gas Depletion Time"

            label = name if Zmask == "all" else None
            with unyt.matplotlib_support:
                ax_neut.loglog(x_neut, y_neut, linestyle, label=label, color=f"C{i}")
                ax_at.loglog(x_at, y_at, linestyle, label=label, color=f"C{i}")
                ax_mol.loglog(x_mol, y_mol, linestyle, label=label, color=f"C{i}")

    for ax in [ax_neut, ax_at, ax_mol]:
        for Zlabel, linestyle in [
            ("all", "-"),
            ("$\\log_{10} Z/Z_\\odot = -1$", "--"),
            ("$\\log_{10} Z/Z_\\odot = 0$", ":"),
            ("$\\log_{10} Z/Z_\\odot = 1$", "-."),
        ]:
            ax.plot([], [], linestyle, color="k", label=Zlabel)
        ax.grid(True)
        ax.set_xlim(1.0e-1, 1.0e4)
        ax.set_ylim(1.0e7, 1.0e12)
        ax.tick_params(direction="in", axis="both", which="both", pad=4.5)
        ax.legend(loc="best")

    neut_filename = "spatial_tgas_neutral.png"
    fig_neut.savefig(f"{output_path}/{neut_filename}", dpi=300)
    pl.close(fig_neut)

    at_filename = "spatial_tgas_atomic.png"
    fig_at.savefig(f"{output_path}/{at_filename}", dpi=300)
    pl.close(fig_at)

    mol_filename = "spatial_tgas_molecular.png"
    fig_mol.savefig(f"{output_path}/{mol_filename}", dpi=300)
    pl.close(fig_mol)

    plots["Combined surface densities"].update(
        {
            neut_filename: {
                "title": "Neutral gas depletion time (spatially averaged)",
                "caption": (
                    "Depletion time of neutral gas vs. neutral gas surface density"
                    " from N most massive individual galaxies, coloured by the mean"
                    " metallicity of the resolved pixel. The surface densities were"
                    " calculated using a grid with pixel size of 750 pc. Coloured"
                    " lines show in the median relations considering only cells with"
                    " fixed metallicity (as indicated in the legends). The grey solid"
                    " line shows the median relation for all pixels, whereas the black"
                    " solid line shows the relation only for pixels that have SFR"
                    " surface density >0."
                ),
            },
            at_filename: {
                "title": "Atomic gas depletion time (spatially averaged)",
                "caption": (
                    "Depletion time of atomic gas vs. atomic gas surface density"
                    " from N most massive individual galaxies, coloured by the mean"
                    " metallicity of the resolved pixel. The surface densities were"
                    " calculated using a grid with pixel size of 750 pc. Coloured"
                    " lines show in the median relations considering only cells with"
                    " fixed metallicity (as indicated in the legends). The grey solid"
                    " line shows the median relation for all pixels, whereas the black"
                    " solid line shows the relation only for pixels that have SFR"
                    " surface density >0."
                ),
            },
            mol_filename: {
                "title": "Molecular gas depletion time (spatially averaged)",
                "caption": (
                    "Depletion time of molecular gas vs. molecular gas surface density"
                    " from N most massive individual galaxies, coloured by the mean"
                    " metallicity of the resolved pixel. The surface densities were"
                    " calculated using a grid with pixel size of 750 pc. Coloured"
                    " lines show in the median relations considering only cells with"
                    " fixed metallicity (as indicated in the legends). The grey solid"
                    " line shows the median relation for all pixels, whereas the black"
                    " solid line shows the relation only for pixels that have SFR"
                    " surface density >0."
                ),
            },
        }
    )

    return plots
