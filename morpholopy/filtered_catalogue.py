#!/usr/bin/env python3

"""
filtered_catalogue.py

List of galaxies that need to be analysed.

Reads the catalogue file and applies the appropriate filters
to construct a list of galaxies that need to be analysed, and
a mask determining which galaxies are plotted on an individual basis.
"""

import numpy as np
import unyt

from numpy.typing import NDArray
from velociraptor.catalogue.catalogue import Catalogue


class FilteredCatalogue:
    """
    List of galaxies that need to be analysed.
    """

    # list of galaxies, using their index in the catalogue
    galaxy_indices: NDArray[np.int64]
    # mask for galaxies that need to be plotted
    plot_galaxy: NDArray[bool]

    def __init__(
        self,
        catalogue: Catalogue,
        minimum_mass_stars: unyt.unyt_quantity,
        mass_variable_stars: str,
        minimum_mass_gas: unyt.unyt_quantity,
        mass_variable_gas: str,
        plotting_lower_limit: unyt.unyt_quantity,
        plotting_upper_limit: unyt.unyt_quantity,
        plotting_number: int,
        plotting_seed: int,
    ):
        """
        Constructor.

        Reads the catalogue and only keeps galaxies with
        structure type 10 (centrals) that have a stellar and gas
        mass within the provided limits.

        Parameters:
         - catalogue: VelociraptorCatalogue
           Galaxy catalogue.
         - minimum_mass_stars: unyt.unyt_quantity
           Minimum stellar mass needed to be selected.
         - mass_variable_stars: str
           Name of the stellar mass variable in the catalogue.
         - minimum_mass_gas: unyt.unyt_quantity
           Minimum gas mass needed to be selected.
         - mass_variable_gas: str
           Name of the gas mass variable in the catalogue.
         - plotting_lower_limit: unyt.unyt_quantity
           Minimum stellar mass to be eligible for individual plotting.
         - plotting_upper_limit: unyt.unyt_quantity
           Maximum stellar mass to be eligible for individual plotting.
         - plotting_number: int
           Maximum number of galaxies for which individual plots are made.
         - plotting_seed: int
           Seed for the random number generator that randomly selects
           galaxies to be plotted, if the number of eligible galaxies
           exceeds the maximum number.

        Note that the stellar mass used for the plotting selection uses the
        same mass variable as the other filter.
        """

        # get the masses from the catalogue
        Mstar = catalogue.get_quantity(mass_variable_stars)
        Mgas = catalogue.get_quantity(mass_variable_gas)

        # compute the mask
        mask = (
            (Mstar >= minimum_mass_stars)
            & (catalogue.get_quantity("structure_type.structuretype") == 10)
            & (Mgas > minimum_mass_gas)
        )
        # turn the mask into a list of indices
        self.galaxy_indices = np.nonzero(mask)[0]

        # select galaxies to plot
        self.plot_galaxy = np.zeros(self.galaxy_indices.shape, dtype=bool)
        # first: apply the requested mass cut
        plot_mask = (Mstar[self.galaxy_indices] >= plotting_lower_limit) & (
            Mstar[self.galaxy_indices] < plotting_upper_limit
        )
        # if no galaxies are within the selected range, we cannot plot anything
        if plot_mask.sum() > 0:
            if plot_mask.sum() > plotting_number:
                # more galaxies in the range than we want to plot: select a random subset
                # we generate a list with the indices of all eligible galaxies and then
                # select a random subset using numpy.random.choice (without replacement)
                # the corresponding elements in the mask are set to True
                np.random.seed(plotting_seed)
                plot_idx = np.nonzero(plot_mask)[0]
                # note that replace=False is required to avoid duplicates
                plot_selection = np.random.choice(
                    plot_idx, size=plotting_number, replace=False
                )
                self.plot_galaxy[plot_selection] = True
            else:
                # all galaxies in the range can be plotted
                self.plot_galaxy[plot_mask] = True
