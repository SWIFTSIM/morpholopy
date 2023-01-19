#!/usr/bin/env python3

"""
config.py

Support for configuration files.
This is mostly based on swiftpipeline.config, but with some
additional configuration options and a few hacks.
"""

import numpy as np
from swiftpipeline.config import Config, Script
import unyt

from typing import Dict

# additional configuration options that are directly read from
# the file, and their default values (if not required, in which
# case the default values is set to None).
# We also support integer and floating point options and variables
# with units. Note that the units in the configuration file are
# assumed, so it is important that the name contains the appropriate
# unit.
direct_read = {
    "mass_limit_stars_in_Msun": (None, np.float32, unyt.Msun),
    "mass_variable_stars": ("apertures.mass_star_30_kpc", str, None),
    "mass_limit_gas_in_Msun": (0.0, np.float32, unyt.Msun),
    "mass_variable_gas": ("apertures.mass_gas_30_kpc", str, None),
    "plotting_lower_mass_limit_in_Msun": (None, np.float32, unyt.Msun),
    "plotting_upper_mass_limit_in_Msun": (None, np.float32, unyt.Msun),
    "plotting_number_of_galaxies": (None, np.uint32, None),
    "plotting_random_seed": (42, np.uint32, None),
    "scaleheight_binsize_kpc": (0.02, np.float32, unyt.kpc),
    "orientation_method": ("stars_0xR0.5_R0.5_0sigma", str, None),
}


class MorphologyConfig(Config):
    """
    Subclass of swiftconfig.config.Config that reads additional
    configuration options from the configuration file.

    Note that this class does not add any variables to the Config object,
    it simply modifies the existing variables in the parent class.
    """

    def __init__(self, config_directory: str):
        """
        Constructor. Takes the configuration directory (so the directory
        containing the configuration file, not the file itself) as input.
        """
        super().__init__(config_directory)

        self.__extract_morphology_variables()

    def __extract_morphology_variables(self):
        """
        Extract additional configuration variables.
        Unlike the version in swiftpipeline.config.Config, we do support
        integer/floating point variables and variables with units.
        """
        for variable, (default, type, unit) in direct_read.items():
            val = self.raw_config.get(variable, default)
            if val is None:
                raise AttributeError(f"No value provided for parameter {variable}!")
            try:
                val = type(val)
            except:
                raise ValueError(
                    f"Could not convert value for parameter {variable} from {val} to {type}!"
                )
            if unit is not None:
                val = unyt.unyt_quantity(val, unit).in_base("galactic")
            setattr(self, variable, val)

    def add_images(self, images: Dict):
        """
        Add the images from the given image dictionary to the configuration
        object, by reusing (abusing?) the mechanism used for scripts in the
        normal pipeline. For each image, we create a fake
        swiftpipeline.config.Script without a script file, but with the
        correct image file name, section, title and caption.

        The input dictionary should have the following general structure:
        images = {
          SECTION_TITLE: {
            FIGURE_NAME.png: {
              "title": "TITLE",
              "caption": "CAPTION",
            }
          }
        }
        """
        for section in images:
            for img in images[section]:
                script_dict = images[section][img]
                script_dict["output_file"] = img
                script_dict["section"] = section
                self.raw_scripts.append(Script(script_dict=script_dict))
