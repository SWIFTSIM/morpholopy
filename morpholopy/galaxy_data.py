#!/usr/bin/env python3

"""
galaxy_data.py

This file contains all the machinery needed to process individual
galaxies: it defines the data type list used to construct the
array with galaxy properties, and classes to efficiently handle
this array, both for the full global galaxy array and the individual
galaxy arrays.

The actual computation for an individual galaxy is performed in the
function process_galaxy() defined at the bottom of this file.
"""

import numpy as np
from swiftsimio import load as load_snapshot
from swiftsimio.objects import cosmo_array
from velociraptor import load as load_catalogue
from velociraptor.particles import load_groups
from velociraptor.swift.swift import to_swiftsimio_dataset
import unyt
import yaml

from .morphology import (
    get_axis_lengths_reduced_tensor,
    get_axis_lengths_normal_tensor,
    get_kappa_corot,
    get_scaleheight,
)
from .orientation import get_orientation_matrices
from .KS import (
    calculate_integrated_surface_densities,
    calculate_spatially_resolved_KS,
    calculate_azimuthally_averaged_KS,
    plot_KS_relations,
)
from .HI_size import calculate_HI_size
from .medians import accumulate_median_data, compute_median
from .galaxy_plots import plot_galaxy

from functools import reduce

from typing import Dict, List, Tuple, Union
from numpy.typing import NDArray

# list of integrated galaxy properties, i.e. each galaxy only has one
# of these
# each element has a name that can be used to access the property, and
# a numpy compatible data type, can be a sub-array, in which case a
# (dtype, size) tuple is required
# this list is passed on to numpy.array(dtype=) or numpy.zeros(dtype=)
# to create structured arrays
data_fields = [
    ("stellar_mass", np.float32),
    ("half_mass_radius_star", np.float32),
    ("stars_kappa_co", np.float32),
    ("stars_momentum", np.float32),
    ("orientation_vector", (np.float32, 3)),
    ("stars_axis_ca_reduced", np.float32),
    ("stars_axis_cb_reduced", np.float32),
    ("stars_axis_ba_reduced", np.float32),
    ("stars_z_axis_reduced", (np.float32, 3)),
    ("stars_axis_ca_normal", np.float32),
    ("stars_axis_cb_normal", np.float32),
    ("stars_axis_ba_normal", np.float32),
    ("stars_z_axis_normal", (np.float32, 3)),
    ("gas_kappa_co", np.float32),
    ("gas_momentum", np.float32),
    ("gas_axis_ca_reduced", np.float32),
    ("gas_axis_cb_reduced", np.float32),
    ("gas_axis_ba_reduced", np.float32),
    ("gas_z_axis_reduced", (np.float32, 3)),
    ("gas_axis_ca_normal", np.float32),
    ("gas_axis_cb_normal", np.float32),
    ("gas_axis_ba_normal", np.float32),
    ("gas_z_axis_normal", (np.float32, 3)),
    ("sigma_HI", np.float32),
    ("sigma_H2", np.float32),
    ("sigma_neutral", np.float32),
    ("sigma_SFR", np.float32),
    ("HI_size", np.float32),
    ("HI_mass", np.float32),
    ("is_active", np.int_),
    ("HI_scaleheight", np.float32),
    ("H2_scaleheight", np.float32),
    ("stars_scaleheight", np.float32),
]

# dictionary containing information for combined
# integrated galaxy quantities
# these quantities are binned in a 2D histogram that
# covers a pre-defined plotting space, such that we
# can compute a reasonably accurate approximation to the
# actual median of these quantities without the need to
# store all the individual data points
medians = {}
# for brevity, we append individual plots in logical chunks:
# - spatially resolved and azimuthally averaged KS plots
for type, label in [
    ("neutral", "$\\Sigma{}_{{\\rm{}HI}+{\\rm{}H2}}$"),
    ("HI", "$\\Sigma{}_{\\rm{}HI}$"),
    ("H2", "$\\Sigma{}_{\\rm{}H2}$"),
]:
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
            "x label": label,
            "y label": label + "$/\\Sigma{}_{{\\rm{}SFR}}}$",
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
                "x label": label,
                "y label": "$\\Sigma{}_{{\\rm{}SFR}}}$",
            }
# - Schruba like plots
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
        "x label": "$\\Sigma{}_{{\\rm{}HI}+{\\rm{}H2}}$",
        "y label": "$\\Sigma{}_{\\rm{}H2}/\\Sigma{}_{{\\rm{}HI}+{\\rm{}H2}}$",
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
        "x label": "$\\Sigma{}_{{\\rm{}HI}+{\\rm{}H2}}$",
        "y label": "$\\Sigma{}_{\\rm{}H2}/\\Sigma{}_{\\rm{}HI}$",
    }

# - Sanchez like plots
medians["H2_to_star_vs_star_spatial"] = {
    "number of bins x": 20,
    "log x": True,
    "range in x": [-1.0, 4.0],
    "number of bins y": 100,
    "log y": True,
    "range in y": [-3.0, 1.0],
    "x units": "Msun/pc**2",
    "y units": "dimensionless",
    "x label": "$\\Sigma{}_{\\bigstar{}}$",
    "y label": "$\\Sigma{}_{\\rm{}H2}/\\Sigma{}_{\\bigstar{}}$",
}
medians["SFR_to_H2_vs_H2_spatial"] = {
    "number of bins x": 20,
    "log x": True,
    "range in x": [-2.0, 3.0],
    "number of bins y": 100,
    "log y": True,
    "range in y": [-11.0, -7.0],
    "x units": "Msun/pc**2",
    "y units": "1/yr",
    "x label": "$\\Sigma{}_{\\rm{}H2}$",
    "y label": "$\\Sigma{}_{\\rm{}SFR}/\\Sigma{}_{\\rm{}H2}$",
}
medians["SFR_to_star_vs_star_spatial"] = {
    "number of bins x": 20,
    "log x": True,
    "range in x": [-1.0, 4.0],
    "number of bins y": 100,
    "log y": True,
    "range in y": [-13.0, -7.0],
    "x units": "Msun/pc**2",
    "y units": "1/yr",
    "x label": "$\\Sigma{}_{\\bigstar{}}$",
    "y label": "$\\Sigma{}_{\\rm{}SFR}/\\Sigma{}_{\\bigstar{}}$",
}

# construct a structured array data type for the median arrays
# using the information from the dictionary
# do not edit this code unless you know what you are doing!
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
    """
    Data array for a single galaxy.
    """

    # Normal (integrated) data
    data: NDArray[data_fields]
    # Median (resolved) data
    median_data: NDArray[median_data_fields]
    # Post-processed medians (only used for individual galaxy plots)
    medians: Dict

    def __init__(self):
        """
        Constructor.
        """
        self.data = np.zeros(1, dtype=data_fields)
        self.median_data = np.zeros(1, dtype=median_data_fields)

    def __getitem__(self, key: str) -> Union[float, NDArray[float]]:
        """
        Access the given property for this galaxy.

        The return type is either a single scalar or sub-array for
        vector properties.
        """
        return self.data[0][key]

    def __setitem__(
        self,
        key: Union[str, List[str]],
        value: Union[Union[float, NDArray[float]], List[Union[float, NDArray[float]]]],
    ):
        """
        Set the given property/properties for this galaxy.

        If a list of keys is given, the values should be a list of the same length.
        The value for each key should be compatible with the shape of the underlying
        element, i.e. scalar properties require a scalar value, vector properties can
        be set to either a vector of the same size, or to a single scalar value (which
        is then applied to each element).
        """
        if isinstance(key, list):
            if len(key) != len(value):
                raise RuntimeError(
                    f"Provided different number of keys and values ({key}, {value})!"
                )
            for k, v in zip(key, value):
                self.data[0][k] = v
        else:
            self.data[0][key] = value

    def accumulate_median_data(
        self, key: str, values_x: unyt.unyt_array, values_y: unyt.unyt_array
    ):
        """
        Bin the given x and y values in the 2D histogram for the appropriate median
        indicated by the key.
        """
        self.median_data[0][key] = accumulate_median_data(
            medians[key], values_x, values_y
        )

    def compute_medians(self):
        """
        Post-process the 2D histograms for this galaxy to get approximate
        median lines.
        """
        self.medians = {}
        for key in medians:
            xvals, yvals = compute_median(medians[key], self.median_data[0][key])
            self.medians[key] = {
                "x centers": xvals.tolist(),
                "y values": yvals.tolist(),
                "PDF": self.median_data[0][key].tolist(),
                **medians[key],
            }


class AllGalaxyData:
    """
    Global data array for all galaxies.

    Only one of these objects should exist for a simulation,
    but multiple can exist in comparison mode.

    Objects can be created empty, or be initialised from a
    metadata file.
    """

    # Normal (integrated) data
    data: NDArray[data_fields]
    # Median (resolved) data
    median_data: NDArray[median_data_fields]
    # Post-processed medians
    medians: Dict

    def __init__(self, number_of_galaxies: int):
        """
        Constructor. Requires the number of galaxies.
        """
        self.data = np.zeros(number_of_galaxies, dtype=data_fields)
        self.median_data = None
        self.medians = None

    def fromfile(filename: str) -> "AllGalaxyData":
        """
        File "constructor": creates a class instance based
        on a metadata file.
        """
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

    def __getitem__(self, key: str) -> NDArray[float]:
        """
        Get the full array for all galaxies of the given quantity.
        """
        return self.data[key]

    def __str__(self) -> str:
        """
        Human friendly string displayed when print()ing an object.
        """
        return (
            "Galaxy data object containing the following:\n"
            f" - variables: {list(self.data.dtype.fields)}\n"
            f" - medians: {list(self.medians.keys())}"
        )

    def __setitem__(self, index: int, galaxy_data: NDArray[data_fields]):
        """
        Set the galaxy properties for the galaxy with the given index.
        For integrated quantities, we simply overwrite the corresponding
        row in the global array.
        For median data histograms, we add the counts for this galaxy.
        """
        self.data[index] = galaxy_data.data[0]
        if self.median_data is None:
            self.median_data = np.zeros(1, dtype=median_data_fields)
        for key in self.median_data.dtype.fields:
            self.median_data[0][key] += galaxy_data.median_data[0][key]

    def compute_medians(self):
        """
        Post-process the 2D histograms to get approximate median lines.
        """
        self.medians = {}
        for key in medians:
            xvals, yvals = compute_median(medians[key], self.median_data[0][key])
            self.medians[key] = {
                "x centers": xvals.tolist(),
                "y values": yvals.tolist(),
                "PDF": self.median_data[0][key].tolist(),
                **medians[key],
            }

    def output(self, output_name: str):
        """
        Dump the data to a metadata file with the given name.
        """
        datadict = {"Number of galaxies": self.data.shape[0], "Galaxy properties": {}}
        for key in self.data.dtype.fields:
            datadict["Galaxy properties"][key] = self.data[key].tolist()
        self.compute_medians()
        datadict["Median lines"] = self.medians
        with open(output_name, "w") as handle:
            yaml.safe_dump(datadict, handle)


def process_galaxy(args) -> Tuple[int, NDArray[data_fields], Union[None, Dict]]:
    """
    Main galaxy analysis function.
    Called exactly once for every galaxy in the simulation. Executed by
    an isolated subprocess that only has access to the variables in the
    'args' argument.
    This function is meant to be used in multiprocessing.Pool.imap_unordered() and
    conforms to its expected function signature.

    Returns:
     - the galaxy index in the FilteredCatalogue list
     - the galaxy data (will be put in AllGalaxyData[index])
     - a dictionary of images that will be appended to the images dictionary,
       or None if there are no individual images for this galaxy (most common case)
    """

    # unpack arguments
    # note that we have to use this approach because
    # multiprocessing.Pool.imap_unordered() requires a function that only
    # takes a single argument
    (
        index,
        galaxy_index,
        catalogue_filename,
        snapshot_filename,
        output_path,
        observational_data_path,
        scaleheight_binsize_kpc,
        scaleheight_lower_gasmass_limit_in_number_of_particles,
        plot_individual_KS_plots,
        orientation_type,
        make_plots,
        main_log,
    ) = args

    # obtain a GalaxyLog for this galaxy and this subprocess
    galaxy_log = main_log.get_galaxy_log(galaxy_index)

    # create an empty data array for this galaxy
    galaxy_data = GalaxyData()

    # (re)load the catalogue to read the properties of this particular galaxy
    # we need to disregard the units to avoid problems with the SFR unit
    catalogue = load_catalogue(catalogue_filename, disregard_units=True)
    groups = load_groups(
        catalogue_filename.replace(".properties", ".catalog_groups"),
        catalogue=catalogue,
    )

    # get the particles belonging to this galaxy
    particles, _ = groups.extract_halo(halo_index=galaxy_index)
    # turn this information into a swiftsimio mask and read the data
    # using swiftsimio
    data, mask = to_swiftsimio_dataset(
        particles, snapshot_filename, generate_extra_mask=True
    )
    # mask out all particles that are actually bound to the galaxy
    # while at it: convert everything to physical coordinates
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
    # convert these to swiftsimio.cosmo_arrays to suppress
    # warnings about mixing variables with and without scale factors
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

    # store some of the catalogue properties directly into the galaxy data array
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

    # provide some information to the user
    galaxy_log.message(
        f"Galaxy {galaxy_index}: stellar mass: {galaxy_data['stellar_mass']:.2e}"
    )

    # Lowest sSFR below which the galaxy is considered passive
    marginal_ssfr = unyt.unyt_quantity(1e-11, units=1 / unyt.year)
    stellar_mass  = catalogue.apertures.mass_star_50_kpc[galaxy_index].to("Msun")
    star_formation_rate = catalogue.apertures.sfr_gas_50_kpc[galaxy_index]

    if stellar_mass == unyt.unyt_quantity(0., units= unyt.msun):
        ssfr =  unyt.unyt_quantity(0., units= 1 / unyt.year)
    else:
        ssfr = star_formation_rate / stellar_mass
        ssfr.convert_to_units("1/yr")

    # Mask for the active objects
    is_active = unyt.unyt_array(
          (ssfr > (1.01 * marginal_ssfr).to(ssfr.units)).astype(np.int_),
          units="dimensionless",
    )

    galaxy_data["is_active"] = is_active

    # get other radius quantities from the catalogue that the orientation
    # calculation might use
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

    # compute some derived quantities and apply some conversions
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

    # done with the pre-processing:
    # now compute the morphology quantities:

    # - axis ratios and angular momenta
    (a, b, c), z_axis = get_axis_lengths_reduced_tensor(galaxy_log, data.stars, Rhalf)
    if (a > 0.0) and (b > 0.0) and (c > 0.0):
        galaxy_data["stars_axis_ca_reduced"] = c / a
        galaxy_data["stars_axis_cb_reduced"] = c / b
        galaxy_data["stars_axis_ba_reduced"] = b / a
        galaxy_data["stars_z_axis_reduced"] = z_axis
    else:
        galaxy_data["stars_axis_ca_reduced"] = np.nan
        galaxy_data["stars_axis_cb_reduced"] = np.nan
        galaxy_data["stars_axis_ba_reduced"] = np.nan
        galaxy_data["stars_z_axis_reduced"][:] = np.nan

    (a, b, c), z_axis = get_axis_lengths_normal_tensor(galaxy_log, data.stars, Rhalf)
    if (a > 0.0) and (b > 0.0) and (c > 0.0):
        galaxy_data["stars_axis_ca_normal"] = c / a
        galaxy_data["stars_axis_cb_normal"] = c / b
        galaxy_data["stars_axis_ba_normal"] = b / a
        galaxy_data["stars_z_axis_normal"] = z_axis
    else:
        galaxy_data["stars_axis_ca_normal"] = np.nan
        galaxy_data["stars_axis_cb_normal"] = np.nan
        galaxy_data["stars_axis_ba_normal"] = np.nan
        galaxy_data["stars_z_axis_normal"][:] = np.nan

    stars_momentum, kappa_corot = get_kappa_corot(
        data.stars, Rhalf, R200crit, Rvir, orientation_type, orientation_vector
    )
    galaxy_data["stars_kappa_co"] = kappa_corot
    galaxy_data["orientation_vector"] = orientation_vector
    galaxy_data["stars_momentum"] = stars_momentum

    (a, b, c), z_axis = get_axis_lengths_reduced_tensor(
        galaxy_log, data.gas, Rhalf, mass_variable="H_neutral_mass"
    )
    if (a > 0.0) and (b > 0.0) and (c > 0.0):
        galaxy_data["gas_axis_ca_reduced"] = c / a
        galaxy_data["gas_axis_cb_reduced"] = c / b
        galaxy_data["gas_axis_ba_reduced"] = b / a
        galaxy_data["gas_z_axis_reduced"] = z_axis
    else:
        galaxy_data["gas_axis_ca_reduced"] = np.nan
        galaxy_data["gas_axis_cb_reduced"] = np.nan
        galaxy_data["gas_axis_ba_reduced"] = np.nan
        galaxy_data["gas_z_axis_reduced"][:] = np.nan

    (a, b, c), z_axis = get_axis_lengths_normal_tensor(
        galaxy_log, data.gas, Rhalf, mass_variable="H_neutral_mass"
    )
    if (a > 0.0) and (b > 0.0) and (c > 0.0):
        galaxy_data["gas_axis_ca_normal"] = c / a
        galaxy_data["gas_axis_cb_normal"] = c / b
        galaxy_data["gas_axis_ba_normal"] = b / a
        galaxy_data["gas_z_axis_normal"] = z_axis
    else:
        galaxy_data["gas_axis_ca_normal"] = np.nan
        galaxy_data["gas_axis_cb_normal"] = np.nan
        galaxy_data["gas_axis_ba_normal"] = np.nan
        galaxy_data["gas_z_axis_normal"][:] = np.nan

    galaxy_data[["gas_momentum", "gas_kappa_co"]] = get_kappa_corot(
        data.gas,
        Rhalf,
        R200crit,
        Rvir,
        orientation_type,
        orientation_vector,
        mass_variable="H_neutral_mass",
    )

    # - KS related quantities
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
        sigma_star,
    ) = calculate_spatially_resolved_KS(
        data, face_on_rmatrix, gas_mask, stars_mask, index
    )
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

    mask = (sigma_star > 0.0) & (sigma_H2 > 0.0)
    galaxy_data.accumulate_median_data(
        "H2_to_star_vs_star_spatial",
        sigma_star[mask],
        sigma_H2[mask] / sigma_star[mask],
    )
    mask = (sigma_SFR > 0.0) & (sigma_H2 > 0.0)
    # manually convert from 1/kpc**2 to 1/pc**2 in the SFR
    galaxy_data.accumulate_median_data(
        "SFR_to_H2_vs_H2_spatial",
        sigma_H2[mask],
        sigma_SFR[mask] / sigma_H2[mask] * 1.0e-6,
    )
    mask = (sigma_star > 0.0) & (sigma_SFR > 0.0)
    galaxy_data.accumulate_median_data(
        "SFR_to_star_vs_star_spatial",
        sigma_star[mask],
        sigma_SFR[mask] / sigma_star[mask] * 1.0e-6,
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

    # - HI size
    galaxy_data[["HI_size", "HI_mass"]] = calculate_HI_size(
        galaxy_log, data, face_on_rmatrix, gas_mask, index
    )

    # - Scaleheight plots
    galaxy_data[["HI_scaleheight", "H2_scaleheight", "stars_scaleheight"]] = get_scaleheight(
            galaxy_log, data, Rhalf, edge_on_rmatrix, gas_mask, stars_mask, index, scaleheight_binsize_kpc,
            scaleheight_lower_gasmass_limit_in_number_of_particles
    )
    

    # if requested, create invidiual plots for this galaxy
    if make_plots:
        # images
        images = {f"ZZZ - Galaxy {galaxy_index:08d}": {}}
        galaxy_images = plot_galaxy(
                catalogue,
                galaxy_index,
                index,
                data,
                face_on_rmatrix,
                edge_on_rmatrix,
                output_path,
            )
        images[f"ZZZ - Galaxy {galaxy_index:08d}"].update(
            galaxy_images["Visualisation"]
        )
        gallery_images = galaxy_images["Gallery"]
    
        if plot_individual_KS_plots:
            # individual KS plots
            galaxy_data.compute_medians()
            KS_images = plot_KS_relations(
                output_path,
                observational_data_path,
                [f"Galaxy {galaxy_index}"],
                [galaxy_data],
                prefix=f"galaxy_{index:03d}_",
                always_plot_scatter=True,
                plot_integrated_quantities=False,
            )
            images[f"ZZZ - Galaxy {galaxy_index:08d}"].update(
                KS_images["Combined surface densities"]
            )
    else:
        images = None
        gallery_images = None

    return index, galaxy_data, images, gallery_images
