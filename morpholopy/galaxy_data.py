import numpy as np
from swiftsimio import load as load_snapshot
from velociraptor import load as load_catalogue
from velociraptor.particles import load_groups
from velociraptor.swift.swift import to_swiftsimio_dataset
import unyt

from .morphology import calculate_morphology
from .KS import calculate_surface_densities

data_fields = [
    ("stellar_mass", np.float32),
    ("half_mass_radius_star", np.float32),
    ("kappa_co", np.float32),
    ("momentum", np.float32),
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


class GalaxyData:
    def __init__(self):
        self.data = np.zeros(1, dtype=data_fields)

    def __getitem__(self, key):
        return self.data[0][key]

    def __setitem__(self, key, value):
        if isinstance(key, list):
            for k, v in zip(key, value):
                self.data[0][k] = v
        else:
            self.data[0][key] = value


class AllGalaxyData:
    def __init__(self, number_of_galaxies):
        self.data = np.zeros(number_of_galaxies, dtype=data_fields)

    def __setitem__(self, index, galaxy_data):
        self.data[index] = galaxy_data.data[0]

    def output(self, output_path, simulation_name):

        mask = self.data["momentum"] > 0.0
        np.savetxt(
            f"{output_path}/momentum_parttype_4_{simulation_name}.new.txt",
            self.data[mask][["stellar_mass", "momentum"]],
        )

        mask = self.data["kappa_co"] > 0.0
        np.savetxt(
            f"{output_path}/Kappa_co_parttype_4_{simulation_name}.new.txt",
            self.data[mask][["stellar_mass", "kappa_co"]],
        )

        mask = self.data["axis_ca"] > 0.0
        np.savetxt(
            f"{output_path}/Axis_ratios_parttype_4_{simulation_name}.new.txt",
            self.data[mask][["stellar_mass", "axis_ca", "axis_cb", "axis_ba"]],
        )

        mask = self.data["gas_momentum"] > 0.0
        np.savetxt(
            f"{output_path}/momentum_parttype_0_{simulation_name}.new.txt",
            self.data[mask][["stellar_mass", "gas_momentum"]],
        )

        mask = self.data["gas_kappa_co"] > 0.0
        np.savetxt(
            f"{output_path}/Kappa_co_parttype_0_{simulation_name}.new.txt",
            self.data[mask][["stellar_mass", "gas_kappa_co"]],
        )

        mask = self.data["gas_axis_ca"] > 0.0
        np.savetxt(
            f"{output_path}/Axis_ratios_parttype_0_{simulation_name}.new.txt",
            self.data[mask][
                ["stellar_mass", "gas_axis_ca", "gas_axis_cb", "gas_axis_ba"]
            ],
        )


def process_galaxy(args):

    index, galaxy_index, catalogue_filename, snapshot_filename = args

    galaxy_data = GalaxyData()

    catalogue = load_catalogue(catalogue_filename)
    groups = load_groups(
        catalogue_filename.replace(".properties", ".catalog_groups"),
        catalogue=catalogue,
    )

    galaxy_center = unyt.unyt_array(
        [
            catalogue.positions.xcminpot[galaxy_index],
            catalogue.positions.ycminpot[galaxy_index],
            catalogue.positions.zcminpot[galaxy_index],
        ]
    )
    galaxy_velocity = unyt.unyt_array(
        [
            catalogue.velocities.vxcminpot[galaxy_index],
            catalogue.velocities.vycminpot[galaxy_index],
            catalogue.velocities.vzcminpot[galaxy_index],
        ]
    )

    galaxy_data["stellar_mass"] = catalogue.apertures.mass_star_30_kpc[galaxy_index].to(
        "Msun"
    )
    galaxy_data["half_mass_radius_star"] = catalogue.radii.r_halfmass_star[
        galaxy_index
    ].to("kpc")

    particles, _ = groups.extract_halo(halo_id=galaxy_index)

    data, mask = to_swiftsimio_dataset(
        particles, snapshot_filename, generate_extra_mask=True
    )

    # read relevant gas data
    gas_coordinates = data.gas.coordinates[mask.gas].to_physical()
    gas_mass = data.gas.masses[mask.gas].to_physical()
    gas_velocities = data.gas.velocities[mask.gas].to_physical()
    gas_hsml = data.gas.smoothing_lengths[mask.gas].to_physical()
    gas_HI = (
        data.gas.species_fractions.HI[mask.gas]
        * data.gas.element_mass_fractions.hydrogen[mask.gas]
        * gas_mass
    )
    gas_H2 = (
        2.0
        * data.gas.species_fractions.H2[mask.gas]
        * data.gas.element_mass_fractions.hydrogen[mask.gas]
        * gas_mass
    )
    gas_sfr = data.gas.star_formation_rates[mask.gas].to_physical()
    gas_rho = data.gas.densities[mask.gas].to_physical()
    gas_Z = data.gas.metal_mass_fractions[mask.gas] / 0.0134

    # read relevant stars data
    stars_coordinates = data.stars.coordinates[mask.stars].to_physical()
    stars_mass = data.stars.masses[mask.stars].to_physical()
    stars_velocities = data.stars.velocities[mask.stars].to_physical()
    stars_hsml = (
        0.5
        * np.ones(stars_mass.shape)
        * data.metadata.gravity_scheme[
            "Maximal physical baryon softening length  [internal units]"
        ][0]
        * data.metadata.units.length
        * data.metadata.a
    )
    stars_birthz = 1.0 / data.stars.birth_scale_factors[mask.stars] - 1.0
    # get age by using cosmology
    stars_age = (
        data.metadata.cosmology.age(data.metadata.z).value
        - np.array(
            [data.metadata.cosmology.age(birthz).value for birthz in stars_birthz]
        )
    ) * unyt.Gyr
    stars_Z = data.stars.metal_mass_fractions[mask.stars]
    stars_init_mass = data.stars.initial_masses[mask.stars].to_physical()
    stars_density = stars_mass * (1.2348 / stars_hsml) ** 3

    # get the box size (for periodic wrapping)
    box = data.metadata.boxsize * data.metadata.a

    # Note: these functions will recenter the coordinates and velocities
    stars_angular_momentum, galaxy_data[
        ["kappa_co", "momentum", "axis_ca", "axis_cb", "axis_ba"]
    ] = calculate_morphology(
        stars_coordinates,
        stars_velocities,
        stars_mass,
        box,
        galaxy_center,
        galaxy_velocity,
    )
    gas_angular_momentum, galaxy_data[
        ["gas_kappa_co", "gas_momentum", "gas_axis_ca", "gas_axis_cb", "gas_axis_ba"]
    ] = calculate_morphology(
        gas_coordinates, gas_velocities, gas_mass, box, galaxy_center, galaxy_velocity
    )

    galaxy_data[["sigma_H2", "sigma_gas", "sigma_SFR"]] = calculate_surface_densities(
        gas_coordinates,
        gas_HI,
        gas_H2,
        gas_sfr,
        gas_angular_momentum,
        galaxy_data["half_mass_radius_star"],
    )

    return index, galaxy_data
