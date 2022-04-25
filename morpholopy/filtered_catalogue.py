import numpy as np
import unyt


class FilteredCatalogue:
    def __init__(self, catalogue, minimum_mass):
        mask = (
            (catalogue.apertures.mass_star_30_kpc >= minimum_mass)
            & (catalogue.structure_type.structuretype == 10)
            & (catalogue.apertures.mass_gas_30_kpc > 0.0 * unyt.Msun)
        )
        self.galaxy_indices = np.nonzero(mask)[0]
