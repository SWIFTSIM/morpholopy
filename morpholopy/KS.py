import numpy as np
import swiftsimio as sw
import unyt
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
        print(q, images[q].units)

    for q in ["HI_mass", "H2_mass", "H_neutral_mass"]:
        images[f"tgas_{q}"] = images[q] / images["star_formation_rates"]
        images[f"tgas_{q}"].convert_to_units("yr")
        images[q].convert_to_units("Msun/kpc**2")

    images["star_formation_rates"].convert_to_units("Msun/yr/kpc**2")

    for q in images:
        pl.imshow(np.log10(images[q]))
        pl.savefig(f"test_surfdens_{q}_{index:03d}.png", dpi=300)
        pl.close()


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
