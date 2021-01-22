"""Definition of cold pool class"""

import pickle

import numpy as np
import pySAM
from pySAM.cold_pool.composite import instant_mean_extraction_data_over_extreme
from pySAM.cold_pool.potential_energy import potential_energy
from pySAM.utils import make_parallel


class ColdPool:

    """Summary

    Attributes:
        BUOYANCY (TYPE): Description
        depth_shear (TYPE): Description
        FMSE (TYPE): Description
        nt (TYPE): Description
        nx (TYPE): Description
        ny (TYPE): Description
        nz (TYPE): Description
        P (TYPE): Description
        POTENTIAL_TEMPERATURE (TYPE): Description
        PRECi (TYPE): Description
        QN (TYPE): Description
        QV (TYPE): Description
        TABS (TYPE): Description
        U (TYPE): Description
        VIRTUAL_TEMPERATURE (TYPE): Description
        VORTICITY (TYPE): Description
        W (TYPE): Description
        X (TYPE): Description
        Y (TYPE): Description
        Z (TYPE): Description
    """

    def __init__(
        self,
        absolute_temperature: np.array,
        instantaneous_precipitation: np.array,
        x_positions: np.array,
        y_positions: np.array,
        z_positions: np.array,
        x_velocity: np.array,
        z_velocity: np.array,
        cloud_base: np.array,
        humidity: np.array,
        pressure: np.array,
        depth_shear: str,
        humidity_evp: np.array,
    ):
        self.TABS = absolute_temperature
        self.PRECi = instantaneous_precipitation
        self.X = x_positions
        self.Y = y_positions
        self.Z = z_positions
        self.U = x_velocity
        self.W = z_velocity
        self.QN = cloud_base
        self.QV = humidity / 1000  # must be kg/kg
        self.P = pressure
        self.depth_shear = depth_shear
        self.QPEVP = humidity_evp

        self.nx = len(self.X)
        self.ny = len(self.Y)
        self.nz = len(self.Z)
        self.nt = len(self.TABS)

        self.FMSE = None
        self.VIRTUAL_TEMPERATURE = None
        self.POTENTIAL_TEMPERATURE = None
        self.BUOYANCY = None
        self.VORTICITY = None

        self.set_basic_variables_from_dataset()

    def save(self, path_to_save: str):

        blacklisted_set = [
            "TABS",
            "PRECi",
            "X",
            "Y",
            "Z",
            "U",
            "W",
            "QN",
            "QV",
            "P",
            "depth_shear",
            "FMSE",
            "VIRTUAL_TEMPERATURE",
            "BUOYANCY",
            "VORTICITY",
            "POTENTIAL_TEMPERATURE",
            "QPEVP",
        ]
        dict = [
            (key, value) for (key, value) in self.__dict__.items() if key not in blacklisted_set
        ]

        f = open(path_to_save, "wb")
        pickle.dump(dict, f, 2)
        f.close()

    def load(self, path_to_load: str):
        f = open(path_to_load, "rb")
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def set_basic_variables_from_dataset(self):
        """Compute basic variables from dataset variable

        One must care of dataset variable shape, here it is adapted to (nt,nz,ny,nx) data
        """
        z_3d_in_time = pySAM.utils.expand_array_to_tzyx_array(
            time_dependence=False,
            input_array=self.Z.values,
            final_shape=np.array((self.nt, self.nz, self.ny, self.nx)),
        )

        vertical_mean_temperature_3d_in_time = pySAM.utils.expand_array_to_tzyx_array(
            time_dependence=True,
            input_array=np.mean(self.TABS.values, axis=(2, 3)),
            final_shape=np.array((self.nt, self.nz, self.ny, self.nx)),
        )

        pressure_3d_in_time = pySAM.utils.expand_array_to_tzyx_array(
            time_dependence=False,
            input_array=self.P.values[: self.nz],
            final_shape=np.array((self.nt, self.nz, self.ny, self.nx)),
        )

        self.FMSE = (
            pySAM.HEAT_CAPACITY_AIR * self.TABS
            + pySAM.GRAVITY * z_3d_in_time
            + pySAM.LATENT_HEAT_AIR * self.QV
        )
        self.VIRTUAL_TEMPERATURE = self.TABS * (
            1
            + (pySAM.MIXING_RATIO_AIR_WATER_VAPOR - 1)
            / pySAM.MIXING_RATIO_AIR_WATER_VAPOR
            * self.QV.values
        )

        vertical_mean_virtual_temperature_3d_in_time = pySAM.utils.expand_array_to_tzyx_array(
            time_dependence=True,
            input_array=np.mean(self.VIRTUAL_TEMPERATURE.values, axis=(2, 3)),
            final_shape=np.array((self.nt, self.nz, self.ny, self.nx)),
        )

        self.POTENTIAL_TEMPERATURE = (
            self.TABS
            * (pySAM.STANDARD_REFERENCE_PRESSURE / pressure_3d_in_time)
            ** pySAM.GAS_CONSTANT_OVER_HEAT_CAPACITY_AIR
        )

        self.BUOYANCY = (
            pySAM.GRAVITY
            * (self.VIRTUAL_TEMPERATURE.values - vertical_mean_virtual_temperature_3d_in_time)
            / vertical_mean_virtual_temperature_3d_in_time
        )

        self.VORTICITY = np.gradient(self.U.values, self.Z, axis=1) - np.gradient(
            self.W.values, self.X, axis=3
        )

    def set_composite_variables(
        self,
        data_name: str,
        variable_to_look_for_extreme: str,
        extreme_events_choice: str,
        x_margin: int,
        y_margin: int = None,
        parallelize: bool = True,
    ) -> np.array:
        """Compute the composite, namely the mean over extreme events, of 2d or 3d variables evolving in time
        This method build attribute

        Args:
            data_name (str): name of the variable composite method is applying to
            variable_to_look_for_extreme (str): name of the variable that describe extreme event
            extreme_events_choice (str): max 1-percentile or 10-percentile
            x_margin (int): width of window zoom
            y_margin (int, optional): depth of window zoom
            parallelize (bool, optional): use all your cpu power
        """
        if data_name not in ["W", "QN", "VORTICITY", "BUOYANCY", "QPEVP"]:
            raise ValueError("data name must be in [W, QN, VORTICITY, BUOYANCY, QPEVP]")
        if variable_to_look_for_extreme not in ["PRECi"]:
            raise ValueError("variable_to_look_for_extreme must be in [PRECi]")

        if parallelize:
            parallel_composite = make_parallel(
                function=instant_mean_extraction_data_over_extreme, nprocesses=pySAM.N_CPU
            )
            composite_variable = parallel_composite(
                iterable_values_1=getattr(self, data_name),
                iterable_values_2=getattr(self, variable_to_look_for_extreme),
                extreme_events_choice=extreme_events_choice,
                x_margin=x_margin,
                y_margin=y_margin,
            )

        else:  # NO PARALLELIZATION
            composite_variable = []
            data = getattr(self, data_name)
            variable_to_look_for_extreme = getattr(self, variable_to_look_for_extreme).values
            for image, variable_extreme in zip(data, variable_to_look_for_extreme):
                composite_variable.append(
                    instant_mean_extraction_data_over_extreme(
                        data=image,
                        variable_to_look_for_extreme=variable_extreme,
                        extreme_events_choice=extreme_events_choice,
                        x_margin=x_margin,
                        y_margin=y_margin,
                    )
                )
        composite_variable = np.array(composite_variable)
        composite_variable = np.mean(composite_variable, axis=0)

        setattr(self, data_name + "_composite", composite_variable)

    def set_potential_energy(self, data_name: str, x_size: int) -> np.array:

        if "_composite" not in data_name:
            raise ValueError("data must be composite variable")

        data_array = getattr(self, data_name)

        potential_energy_array = potential_energy(
            data_array=data_array, z_array=self.Z, x_size=x_size, depth_shear=self.depth_shear
        )

        setattr(self, "potential_energy_" + str(x_size), potential_energy_array)
        setattr(self, "mean_potential_energy_" + str(x_size), np.mean(potential_energy_array))
