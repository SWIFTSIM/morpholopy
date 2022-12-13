import numpy as np
import unyt
from functools import reduce


class FilteredCatalogue:
    def __init__(
        self,
        catalogue,
        minimum_mass_stars,
        mass_variable_stars,
        minimum_mass_gas,
        mass_variable_gas,
        plotting_lower_limit,
        plotting_upper_limit,
        plotting_number,
        plotting_seed,
    ):
        Mstar = reduce(getattr, mass_variable_stars.split("."), catalogue)
        Mgas = reduce(getattr, mass_variable_gas.split("."), catalogue)
        mask = (
            (Mstar >= minimum_mass_stars)
            & (catalogue.structure_type.structuretype == 10)
            & (Mgas > minimum_mass_gas)
        )
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
                np.random.seed(plotting_seed)
                plot_idx = np.nonzero(plot_mask)[0]
                plot_selection = np.random.choice(
                    plot_idx, size=plotting_number, replace=False
                )
                self.plot_galaxy[plot_selection] = True
            else:
                # all galaxies in the range can be plotted
                self.plot_galaxy[:] = True
