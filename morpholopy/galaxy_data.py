import numpy as np
from swiftsimio import load as load_snapshot
from swiftsimio.objects import cosmo_array
from velociraptor import load as load_catalogue
from velociraptor.particles import load_groups
from velociraptor.swift.swift import to_swiftsimio_dataset
import unyt
import yaml

from .morphology import get_new_axis_lengths, get_kappa_corot
from .orientation import get_orientation_matrices
from .KS import (
    calculate_integrated_surface_densities,
    calculate_spatially_resolved_KS,
    calculate_azimuthally_averaged_KS,
)
from .HI_size import calculate_HI_size
from .medians import accumulate_median_data, compute_median
from .galaxy_plots import plot_galaxy

from functools import reduce

data_fields = [
    ("stellar_mass", np.float32),
    ("half_mass_radius_star", np.float32),
    ("stars_kappa_co", np.float32),
    ("stars_momentum", np.float32),
    ("orientation_vector", (np.float32, 3)),
    ("stars_axis_ca", np.float32),
    ("stars_axis_cb", np.float32),
    ("stars_axis_ba", np.float32),
    ("stars_z_axis", (np.float32, 3)),
    ("gas_kappa_co", np.float32),
    ("gas_momentum", np.float32),
    ("gas_axis_ca", np.float32),
    ("gas_axis_cb", np.float32),
    ("gas_axis_ba", np.float32),
    ("gas_z_axis", (np.float32, 3)),
    ("sigma_HI", np.float32),
    ("sigma_H2", np.float32),
    ("sigma_neutral", np.float32),
    ("sigma_SFR", np.float32),
    ("HI_size", np.float32),
    ("HI_mass", np.float32),
]

medians = {}
for type in ["neutral", "HI", "H2"]:
    for Zmask in ["all", "Zm1", "Z0", "Zp1"]:
        medians[f"sigma_{type}_tgas_spatial_{Zmask}"] = {
            "number of bins x": 20,
            "log x": True,
            "range in x": [-1.0, 4.0],
            "number of bins y": 100,
            "log y": True,
            "range in y": [7.0, 12.0],
            "x units": "Msun/pc**2",
            "y units": "yr",
        }
        for method in ["spatial", "azimuthal"]:
            medians[f"sigma_{type}_SFR_{method}_{Zmask}"] = {
                "number of bins x": 20,
                "log x": True,
                "range in x": [-1.0, 4.0],
                "number of bins y": 100,
                "log y": True,
                "range in y": [-6.0, 1.0],
                "x units": "Msun/pc**2",
                "y units": "Msun/yr/kpc**2",
            }
for Zmask in ["all", "Zm1", "Z0", "Zp1"]:
    medians[f"H2_to_neutral_vs_neutral_spatial_{Zmask}"] = {
        "number of bins x": 20,
        "log x": True,
        "range in x": [-1.0, 4.0],
        "number of bins y": 100,
        "log y": True,
        "range in y": [-8.0, 1.0],
        "x units": "Msun/pc**2",
        "y units": "dimensionless",
    }
    medians[f"H2_to_HI_vs_neutral_spatial_{Zmask}"] = {
        "number of bins x": 20,
        "log x": True,
        "range in x": [-1.0, 4.0],
        "number of bins y": 100,
        "log y": True,
        "range in y": [-2.0, 3.0],
        "x units": "Msun/pc**2",
        "y units": "dimensionless",
    }

median_data_fields = []
for median in medians:
    median_data_fields.append(
        (
            median,
            np.uint32,
            (medians[median]["number of bins x"], medians[median]["number of bins y"]),
        )
    )


class GalaxyData:
    def __init__(self):
        self.data = np.zeros(1, dtype=data_fields)
        self.median_data = np.zeros(1, dtype=median_data_fields)

    def __getitem__(self, key):
        return self.data[0][key]

    def __setitem__(self, key, value):
        if isinstance(key, list):
            if len(key) != len(value):
                raise RuntimeError(
                    f"Provided different number of keys and values ({key}, {value})!"
                )
            for k, v in zip(key, value):
                self.data[0][k] = v
        else:
            self.data[0][key] = value

    def accumulate_median_data(self, key, values_x, values_y):
        self.median_data[0][key] = accumulate_median_data(
            medians[key], values_x, values_y
        )

    def get_median_data(self, key):
        return self.median_data[0][key]


class AllGalaxyData:
    def __init__(self, number_of_galaxies):
        self.data = np.zeros(number_of_galaxies, dtype=data_fields)
        self.median_data = None
        self.medians = None

    def fromfile(filename):
        with open(filename, "r") as handle:
            datadict = yaml.safe_load(handle)
        number_of_galaxies = datadict["Number of galaxies"]
        all_galaxies = AllGalaxyData(number_of_galaxies)
        for key in all_galaxies.data.dtype.fields:
            all_galaxies.data[key] = np.array(
                datadict["Galaxy properties"][key], dtype=all_galaxies.data[key].dtype
            )
        all_galaxies.medians = datadict["Median lines"]
        return all_galaxies

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        return (
            "Galaxy data object containing the following:\n"
            f" - variables: {list(self.data.dtype.fields)}\n"
            f" - medians: {list(self.medians.keys())}"
        )

    def __setitem__(self, index, galaxy_data):
        self.data[index] = galaxy_data.data[0]
        if self.median_data is None:
            self.median_data = np.zeros(1, dtype=median_data_fields)
        for key in self.median_data.dtype.fields:
            self.median_data[0][key] += galaxy_data.median_data[0][key]

    def compute_medians(self):
        self.medians = {}
        for key in medians:
            xvals, yvals = compute_median(medians[key], self.median_data[0][key])
            self.medians[key] = {
                "x centers": xvals.tolist(),
                "y values": yvals.tolist(),
                **medians[key],
            }

    def output(self, output_name):
        datadict = {"Number of galaxies": self.data.shape[0], "Galaxy properties": {}}
        for key in self.data.dtype.fields:
            datadict["Galaxy properties"][key] = self.data[key].tolist()
        self.compute_medians()
        datadict["Median lines"] = self.medians
        with open(output_name, "w") as handle:
            yaml.safe_dump(datadict, handle)


def process_galaxy(args):

    index, galaxy_index, catalogue_filename, snapshot_filename, output_path, orientation_type, make_plots, main_log = (
        args
    )

    galaxy_log = main_log.get_galaxy_log(galaxy_index)

    galaxy_data = GalaxyData()

    # we need to disregard the units to avoid problems with the SFR unit
    catalogue = load_catalogue(catalogue_filename, disregard_units=True)
    groups = load_groups(
        catalogue_filename.replace(".properties", ".catalog_groups"),
        catalogue=catalogue,
    )

    particles, _ = groups.extract_halo(halo_index=galaxy_index)

    data, mask = to_swiftsimio_dataset(
        particles, snapshot_filename, generate_extra_mask=True
    )
    # mask out all particles that are actually bound to the galaxy
    for parttype in ["gas", "dark_matter", "stars", "black_holes"]:
        fields = reduce(
            getattr, f"metadata.{parttype}_properties.field_names".split("."), data
        )
        partmask = getattr(mask, parttype)
        for field in fields:
            group = getattr(data, parttype)
            dataset = getattr(group, field)
            if hasattr(dataset, "named_columns"):
                columns = dataset.named_columns
                for column in columns:
                    vals = getattr(dataset, column)[partmask].to_physical()
                    setattr(dataset, column, vals)
            else:
                vals = dataset[partmask].to_physical()
                setattr(group, field, vals)

    # get some properties from the catalogue
    galaxy_center = cosmo_array(
        unyt.unyt_array(
            [
                catalogue.positions.xcmbp[galaxy_index],
                catalogue.positions.ycmbp[galaxy_index],
                catalogue.positions.zcmbp[galaxy_index],
            ]
        ),
        comoving=False,
        cosmo_factor=data.gas.coordinates.cosmo_factor,
    )
    galaxy_velocity = cosmo_array(
        unyt.unyt_array(
            [
                catalogue.velocities.vxc[galaxy_index],
                catalogue.velocities.vyc[galaxy_index],
                catalogue.velocities.vzc[galaxy_index],
            ]
        ),
        comoving=False,
        cosmo_factor=data.gas.velocities.cosmo_factor,
    )

    galaxy_data["stellar_mass"] = cosmo_array(
        catalogue.apertures.mass_star_50_kpc[galaxy_index].to("Msun"),
        comoving=False,
        cosmo_factor=data.gas.masses.cosmo_factor,
    )
    r_halfmass_star = catalogue.radii.r_halfmass_star[galaxy_index]
    galaxy_data["half_mass_radius_star"] = cosmo_array(
        r_halfmass_star.to("kpc"),
        comoving=False,
        cosmo_factor=data.gas.coordinates.cosmo_factor,
    )

    galaxy_log.message(
        f"Galaxy {galaxy_index}: stellar mass: {galaxy_data['stellar_mass']:.2e}"
    )

    Rhalf = cosmo_array(
        catalogue.apertures.rhalfmass_star_50_kpc[galaxy_index],
        comoving=False,
        cosmo_factor=data.gas.coordinates.cosmo_factor,
    )
    R200crit = cosmo_array(
        catalogue.spherical_overdensities.r_200_rhocrit[galaxy_index],
        comoving=False,
        cosmo_factor=data.gas.coordinates.cosmo_factor,
    )
    Rvir = cosmo_array(
        catalogue.radii.rvir[galaxy_index],
        comoving=False,
        cosmo_factor=data.gas.coordinates.cosmo_factor,
    )

    # get the box size (for periodic wrapping)
    box = cosmo_array(
        data.metadata.boxsize,
        comoving=True,
        cosmo_factor=data.gas.coordinates.cosmo_factor,
    ).to_physical()

    data.gas.HI_mass = (
        data.gas.species_fractions.HI
        * data.gas.element_mass_fractions.hydrogen
        * data.gas.masses
    )
    data.gas.H2_mass = (
        2.0
        * data.gas.species_fractions.H2
        * data.gas.element_mass_fractions.hydrogen
        * data.gas.masses
    )
    # mask out negative SFR values, which are not SFR at all
    data.gas.star_formation_rates[
        data.gas.star_formation_rates < 0.0 * data.gas.star_formation_rates.units
    ] = (0.0 * data.gas.star_formation_rates.units)
    data.gas.star_formation_rates.convert_to_physical()
    data.gas.H_neutral_mass = data.gas.HI_mass + data.gas.H2_mass
    data.gas.metal_mass = data.gas.metal_mass_fractions * data.gas.masses

    # convert to the galaxy frame: subtract the galaxy centre position and
    # velocity from the particle coordinates and velocities
    # do not forget about the periodic boundary!
    data.gas.coordinates.convert_to_physical()
    data.gas.coordinates[:, :] -= galaxy_center[None, :]
    data.gas.coordinates[:, :] += 0.5 * box[None, :]
    data.gas.coordinates[:, :] %= box[None, :]
    data.gas.coordinates[:, :] -= 0.5 * box[None, :]
    data.gas.smoothing_lengths.convert_to_physical()
    data.gas.velocities.convert_to_physical()
    data.gas.velocities[:, :] -= galaxy_velocity[None, :]
    gas_radius = unyt.array.unorm(data.gas.coordinates[:, :], axis=1)
    gas_mask = gas_radius < 30.0 * unyt.kpc
    data.gas.smoothing_lengths.convert_to_physical()

    data.stars.coordinates.convert_to_physical()
    data.stars.coordinates[:, :] -= galaxy_center[None, :]
    data.stars.coordinates[:, :] += 0.5 * box[None, :]
    data.stars.coordinates[:, :] %= box[None, :]
    data.stars.coordinates[:, :] -= 0.5 * box[None, :]
    data.stars.velocities.convert_to_physical()
    data.stars.velocities[:, :] -= galaxy_velocity[None, :]
    stars_radius = unyt.array.unorm(data.stars.coordinates[:, :], axis=1)
    stars_mask = stars_radius < 30.0 * unyt.kpc

    # determine the angular momentum vector and corresponding face-on and
    # edge-on rotation matrices
    orientation_vector, face_on_rmatrix, edge_on_rmatrix = get_orientation_matrices(
        data, Rhalf, R200crit, Rvir, orientation_type
    )

    (a, b, c), z_axis = get_new_axis_lengths(galaxy_log, data.stars, Rhalf)
    stars_momentum, kappa_corot = get_kappa_corot(
        data.stars, Rhalf, R200crit, Rvir, orientation_type, orientation_vector
    )
    if (a > 0.0) and (b > 0.0) and (c > 0.0):
        galaxy_data["stars_axis_ca"] = c / a
        galaxy_data["stars_axis_cb"] = c / b
        galaxy_data["stars_axis_ba"] = b / a
        galaxy_data["stars_z_axis"] = z_axis
    galaxy_data["stars_kappa_co"] = kappa_corot
    galaxy_data["orientation_vector"] = orientation_vector
    galaxy_data["stars_momentum"] = stars_momentum

    (a, b, c), z_axis = get_new_axis_lengths(
        galaxy_log, data.gas, Rhalf, mass_variable="H_neutral_mass"
    )
    if (a > 0.0) and (b > 0.0) and (c > 0.0):
        galaxy_data["gas_axis_ca"] = c / a
        galaxy_data["gas_axis_cb"] = c / b
        galaxy_data["gas_axis_ba"] = b / a
        galaxy_data["gas_z_axis"] = z_axis
    galaxy_data[["gas_momentum", "gas_kappa_co"]] = get_kappa_corot(
        data.gas,
        Rhalf,
        R200crit,
        Rvir,
        orientation_type,
        orientation_vector,
        mass_variable="H_neutral_mass",
    )

    galaxy_data[
        ["sigma_HI", "sigma_H2", "sigma_neutral", "sigma_SFR"]
    ] = calculate_integrated_surface_densities(
        data, face_on_rmatrix, gas_mask, r_halfmass_star
    )

    (
        sigma_neutral,
        sigma_HI,
        sigma_H2,
        sigma_SFR,
        tgas_neutral,
        tgas_HI,
        tgas_H2,
        metallicity,
    ) = calculate_spatially_resolved_KS(data, face_on_rmatrix, gas_mask, index)
    metallicity[metallicity <= 0.0] = 1.0e-6
    Zgas = np.log10(metallicity)
    for sigma, tgas, name in [
        (sigma_neutral, tgas_neutral, "neutral"),
        (sigma_HI, tgas_HI, "HI"),
        (sigma_H2, tgas_H2, "H2"),
    ]:
        for Zmask in ["all", "Zm1", "Z0", "Zp1"]:
            if Zmask != "all":
                Zmin = {"Zm1": -1.2, "Z0": -0.2, "Zp1": 0.6}[Zmask]
                # we use a Zmax of 1000 as infinite upper limit
                Zmax = {"Zm1": -0.8, "Z0": 0.2, "Zp1": 1000.0}[Zmask]
                mask = (Zgas >= Zmin) & (Zgas < Zmax)
            else:
                mask = np.ones(sigma.shape, dtype=bool)
            vmask = mask & (sigma > 0.0) & (sigma_SFR > 0.0)
            galaxy_data.accumulate_median_data(
                f"sigma_{name}_SFR_spatial_{Zmask}", sigma[vmask], sigma_SFR[vmask]
            )
            vmask = mask & (tgas > 0.0) & (sigma > 0.0)
            galaxy_data.accumulate_median_data(
                f"sigma_{name}_tgas_spatial_{Zmask}", sigma[vmask], tgas[vmask]
            )

    for sigma, name in [(sigma_neutral, "neutral"), (sigma_HI, "HI")]:
        for Zmask in ["all", "Zm1", "Z0", "Zp1"]:
            if Zmask != "all":
                Zmin = {"Zm1": -1.2, "Z0": -0.2, "Zp1": 0.6}[Zmask]
                # we use a Zmax of 1000 as infinite upper limit
                Zmax = {"Zm1": -0.8, "Z0": 0.2, "Zp1": 1000.0}[Zmask]
                mask = (Zgas >= Zmin) & (Zgas < Zmax)
            else:
                mask = np.ones(sigma.shape, dtype=bool)
            vmask = mask & (sigma_neutral > 0.0) & (sigma > 0.0)
            galaxy_data.accumulate_median_data(
                f"H2_to_{name}_vs_neutral_spatial_{Zmask}",
                sigma_neutral[vmask],
                sigma_H2[vmask] / sigma[vmask],
            )

    (
        sigma_neutral,
        sigma_HI,
        sigma_H2,
        sigma_SFR,
        metallicity,
    ) = calculate_azimuthally_averaged_KS(data, face_on_rmatrix, gas_mask, index)
    if sigma_neutral is not None:
        metallicity[metallicity <= 0.0] = 1.0e-6
        Zgas = np.log10(metallicity)
        for sigma, name in [
            (sigma_neutral, "neutral"),
            (sigma_HI, "HI"),
            (sigma_H2, "H2"),
        ]:
            for Zmask in ["all", "Zm1", "Z0", "Zp1"]:
                if Zmask != "all":
                    Zmin = {"Zm1": -1.2, "Z0": -0.2, "Zp1": 0.6}[Zmask]
                    # we use a Zmax of 1000 as infinite upper limit
                    Zmax = {"Zm1": -0.8, "Z0": 0.2, "Zp1": 1000.0}[Zmask]
                    mask = (Zgas >= Zmin) & (Zgas < Zmax)
                else:
                    mask = np.ones(sigma.shape, dtype=bool)
                mask &= (sigma > 0.0) & (sigma_SFR > 0.0)
                galaxy_data.accumulate_median_data(
                    f"sigma_{name}_SFR_azimuthal_{Zmask}", sigma[mask], sigma_SFR[mask]
                )

    galaxy_data[["HI_size", "HI_mass"]] = calculate_HI_size(
        galaxy_log, data, face_on_rmatrix, gas_mask, index
    )

    if make_plots:
        images = plot_galaxy(
            catalogue,
            galaxy_index,
            index,
            data,
            face_on_rmatrix,
            edge_on_rmatrix,
            output_path,
        )
    else:
        images = None

    return index, galaxy_data, images
