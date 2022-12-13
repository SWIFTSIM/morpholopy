import numpy as np
from swiftpipeline.config import Config, Script
import unyt

direct_read = {
    "mass_limit_stars_in_Msun": (None, np.float32, unyt.Msun),
    "mass_variable_stars": ("apertures.mass_star_30_kpc", str, None),
    "mass_limit_gas_in_Msun": (0.0, np.float32, unyt.Msun),
    "mass_variable_gas": ("apertures.mass_gas_30_kpc", str, None),
    "plotting_lower_mass_limit_in_Msun": (None, np.float32, unyt.Msun),
    "plotting_upper_mass_limit_in_Msun": (None, np.float32, unyt.Msun),
    "plotting_number_of_galaxies": (None, np.uint32, None),
    "plotting_random_seed": (42, np.uint32, None),
    "orientation_method": ("stars_0xR0.5_R0.5_0sigma", str, None),
}


class MorphologyConfig(Config):
    def __init__(self, config_directory: str):
        super().__init__(config_directory)

        self.__extract_morphology_variables()

    def __extract_morphology_variables(self):
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

    def add_images(self, images):
        for section in images:
            for img in images[section]:
                script_dict = images[section][img]
                script_dict["output_file"] = img
                script_dict["section"] = section
                self.raw_scripts.append(Script(script_dict=script_dict))
