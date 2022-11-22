import numpy as np
from swiftsimio import load as load_snapshot
from swiftsimio.objects import cosmo_array
from velociraptor import load as load_catalogue
from velociraptor.particles import load_groups
from velociraptor.swift.swift import to_swiftsimio_dataset
import unyt
import yaml

from .morphology import (
    calculate_morphology,
    get_angular_momentum_vector,
    get_axis_lengths,
    get_kappa_corot,
)
from .orientation import get_orientation_matrices
from .KS import (
    calculate_integrated_surface_densities,
    calculate_spatially_resolved_KS,
    calculate_azimuthally_averaged_KS,
)
from .HI_size import calculate_HI_size
from .medians import accumulate_median_data, compute_median

from functools import reduce

data_fields = [
    ("stellar_mass", np.float32),
    ("half_mass_radius_star", np.float32),
    ("kappa_co", np.float32),
    ("momentum", (np.float32, 3)),
    ("axis_ca", np.float32),
    ("axis_cb", np.float32),
    ("axis_ba", np.float32),
    ("gas_kappa_co", np.float32),
    ("gas_momentum", np.float32),
    ("gas_axis_ca", np.float32),
    ("gas_axis_cb", np.float32),
    ("gas_axis_ba", np.float32),
    ("sigma_H2", np.float32),
    ("sigma_gas", np.float32),
    ("sigma_SFR", np.float32),
    ("HI_size", np.float32),
    ("HI_mass", np.float32),
]

medians = {
    "sigma_gas_SFR_spatial": {
        "number of bins x": 20,
        "log x": True,
        "range in x": [-1.0, 4.0],
        "number of bins y": 100,
        "log y": True,
        "range in y": [-6.0, 1.0],
        "x units": "Msun/pc**2",
        "y units": "Msun/yr/kpc**2",
    },
    "sigma_H2_SFR_spatial": {
        "number of bins x": 20,
        "log x": True,
        "range in x": [-1.0, 4.0],
        "number of bins y": 100,
        "log y": True,
        "range in y": [-6.0, 1.0],
        "x units": "Msun/pc**2",
        "y units": "Msun/yr/kpc**2",
    },
    "sigma_gas_SFR_azimuthal": {
        "number of bins x": 20,
        "log x": True,
        "range in x": [-1.0, 4.0],
        "number of bins y": 100,
        "log y": True,
        "range in y": [-6.0, 1.0],
        "x units": "Msun/pc**2",
        "y units": "Msun/yr/kpc**2",
    },
    "sigma_H2_SFR_azimuthal": {
        "number of bins x": 20,
        "log x": True,
        "range in x": [-1.0, 4.0],
        "number of bins y": 100,
        "log y": True,
        "range in y": [-6.0, 1.0],
        "x units": "Msun/pc**2",
        "y units": "Msun/yr/kpc**2",
    },
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
    output_order = [
        "stellar_mass",
        "momentum",
        "kappa_co",
        "axis_ca",
        "axis_cb",
        "axis_ba",
        "gas_momentum",
        "gas_kappa_co",
        "gas_axis_ca",
        "gas_axis_cb",
        "gas_axis_ba",
        "sigma_H2",
        "sigma_gas",
        "sigma_SFR",
        "HI_size",
        "HI_mass",
    ]

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

    index, galaxy_index, catalogue_filename, snapshot_filename, orientation_type = args

    galaxy_data = GalaxyData()

    catalogue = load_catalogue(catalogue_filename)
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

    # read relevant gas data
    gas_coordinates = data.gas.coordinates.to_physical()
    gas_mass = data.gas.masses.to_physical()
    gas_velocities = data.gas.velocities.to_physical()
    gas_hsml = data.gas.smoothing_lengths.to_physical()
    gas_HI = (
        data.gas.species_fractions.HI
        * data.gas.element_mass_fractions.hydrogen
        * gas_mass
    )
    gas_H2 = (
        2.0
        * data.gas.species_fractions.H2
        * data.gas.element_mass_fractions.hydrogen
        * gas_mass
    )
    gas_sfr = data.gas.star_formation_rates.to_physical()
    gas_rho = data.gas.densities.to_physical()
    gas_Z = data.gas.metal_mass_fractions / 0.0134
    gas_temp = data.gas.temperatures

    # read relevant stars data
    stars_coordinates = data.stars.coordinates.to_physical()
    stars_mass = data.stars.masses.to_physical()
    stars_velocities = data.stars.velocities.to_physical()
    stars_hsml = (
        0.5
        * np.ones(stars_mass.shape)
        * data.metadata.gravity_scheme[
            "Maximal physical baryon softening length  [internal units]"
        ][0]
        * data.metadata.units.length
        * data.metadata.a
    )
    stars_birthz = 1.0 / data.stars.birth_scale_factors - 1.0
    # get age by using cosmology
    stars_age = (
        data.metadata.cosmology.age(data.metadata.z).value
        - np.array(
            [data.metadata.cosmology.age(birthz.value).value for birthz in stars_birthz]
        )
    ) * unyt.Gyr
    stars_Z = data.stars.metal_mass_fractions
    stars_init_mass = data.stars.initial_masses.to_physical()
    stars_density = stars_mass * (1.2348 / stars_hsml) ** 3

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
                catalogue.velocities.vxcmbp[galaxy_index],
                catalogue.velocities.vycmbp[galaxy_index],
                catalogue.velocities.vzcmbp[galaxy_index],
            ]
        ),
        comoving=False,
        cosmo_factor=data.gas.velocities.cosmo_factor,
    )

    galaxy_data["stellar_mass"] = cosmo_array(
        catalogue.apertures.mass_star_30_kpc[galaxy_index].to("Msun"),
        comoving=False,
        cosmo_factor=data.gas.masses.cosmo_factor,
    )
    r_halfmass_star = catalogue.radii.r_halfmass_star[galaxy_index]
    galaxy_data["half_mass_radius_star"] = cosmo_array(
        r_halfmass_star.to("kpc"),
        comoving=False,
        cosmo_factor=data.gas.coordinates.cosmo_factor,
    )

    Rhalf = cosmo_array(
        catalogue.apertures.rhalfmass_star_50_kpc[galaxy_index],
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
    """
    face_on_rmatrix, edge_on_rmatrix = get_angular_momentum_vector(
        stars_coordinates, stars_velocities, stars_mass, galaxy_center, galaxy_velocity
    )
    """
    orientation_vector, face_on_rmatrix, edge_on_rmatrix = get_orientation_matrices(
        data, galaxy_center, galaxy_velocity, Rhalf, orientation_type
    )

    """
    galaxy_data[
        ["kappa_co", "momentum", "axis_ca", "axis_cb", "axis_ba"]
    ] = calculate_morphology(
        stars_coordinates,
        stars_velocities,
        stars_mass,
        box,
        galaxy_center,
        galaxy_velocity,
    )
    galaxy_data[
        ["gas_kappa_co", "gas_momentum", "gas_axis_ca", "gas_axis_cb", "gas_axis_ba"]
    ] = calculate_morphology(
        gas_coordinates, gas_velocities, gas_mass, box, galaxy_center, galaxy_velocity
    )
    """
    a, b, c = get_axis_lengths(data.stars, Rhalf, orientation_type)
    kappa_corot = get_kappa_corot(
        data.stars, Rhalf, orientation_type, orientation_vector
    )
    galaxy_data["axis_ca"] = c / a
    galaxy_data["axis_cb"] = c / b
    galaxy_data["axis_ba"] = b / a
    galaxy_data["kappa_co"] = kappa_corot
    galaxy_data["momentum"] = orientation_vector

    """
    galaxy_data[
        ["sigma_H2", "sigma_gas", "sigma_SFR"]
    ] = calculate_integrated_surface_densities(
        data, face_on_rmatrix, gas_mask, r_halfmass_star
    )

    sigma_gas, sigma_H2, sigma_SFR = calculate_spatially_resolved_KS(
        data, face_on_rmatrix, gas_mask, index
    )
    mask = (sigma_gas > 0.0) & (sigma_SFR > 0.0)
    galaxy_data.accumulate_median_data(
        "sigma_gas_SFR_spatial", sigma_gas[mask], sigma_SFR[mask]
    )
    mask = (sigma_H2 > 0.0) & (sigma_SFR > 0.0)
    galaxy_data.accumulate_median_data(
        "sigma_H2_SFR_spatial", sigma_H2[mask], sigma_SFR[mask]
    )
    sigma_gas, sigma_H2, sigma_SFR = calculate_azimuthally_averaged_KS(
        data, face_on_rmatrix, gas_mask, index
    )
    if not sigma_gas is None:
        galaxy_data.accumulate_median_data(
            "sigma_gas_SFR_azimuthal", sigma_gas, sigma_SFR
        )
        galaxy_data.accumulate_median_data(
            "sigma_H2_SFR_azimuthal", sigma_H2, sigma_SFR
        )

    galaxy_data[["HI_size", "HI_mass"]] = calculate_HI_size(
        data, face_on_rmatrix, gas_mask, index
    )
    """

    return index, galaxy_data
