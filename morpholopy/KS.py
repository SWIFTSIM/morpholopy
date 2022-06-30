import numpy as np
import swiftsimio as sw
import unyt
import scipy.stats as stats

make_plots = False
if make_plots:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pl


def calculate_surface_densities_grid(
    data, face_on_rmatrix, gas_mask, index, resolution=128
):

    R = 30.0 * unyt.kpc

    images = {}

    for q in ["HI_mass", "H2_mass", "H_neutral_mass", "star_formation_rates"]:
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

    for q in ["HI_mass", "H2_mass", "H_neutral_mass"]:
        images[f"tgas_{q}"] = images[q] / images["star_formation_rates"]
        images[f"tgas_{q}"].convert_to_units("yr")
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
        pl.savefig(f"test_sigma_gas_SFR_{index:03d}.png", dpi=300)
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
        images["H2_mass"].value.flatten(),
        images["star_formation_rates"].value.flatten(),
    )


def calculate_azimuthally_averaged_KS(data, face_on_rmatrix, gas_mask, index):
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

    return mhist["H_neutral_mass"], mhist["H2_mass"], mhist["star_formation_rates"]


def calculate_integrated_surface_densities(data, face_on_rmatrix, gas_mask, radius):

    surface = np.pi * radius ** 2

    x, y, _ = np.matmul(face_on_rmatrix, data.gas.coordinates[gas_mask].T)
    r = np.sqrt(x ** 2 + y ** 2)
    select = gas_mask.copy()
    select[gas_mask] = r < radius

    Sigma_H2 = data.gas.H2_mass[select].sum() / surface
    Sigma_gas = data.gas.H_neutral_mass[select].sum() / surface
    select &= data.gas.star_formation_rates > 0.0
    Sigma_SFR = data.gas.star_formation_rates[select].sum() / surface

    return (
        Sigma_H2.to("Msun/pc**2"),
        Sigma_gas.to("Msun/pc**2"),
        Sigma_SFR.to("Msun/yr/kpc**2"),
    )
