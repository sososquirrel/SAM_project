"""Cold Pool class, allows to analyze and describe formation of cold pool"""

import pickle

import numpy as np
import pySAM
from pySAM.cold_pool.composite import instant_mean_extraction_data_over_extreme
from pySAM.cold_pool.geometry_profile import geometry_profile
from pySAM.cold_pool.potential_energy import potential_energy
from pySAM.utils import make_parallel


class ColdPool:

    """ColdPool creates object that gathers all variables useful to analyse and descibe a cold pool.
    It allows you to calculate composite figures, ie statistical overview of extreme events.
    The potential energy of cold pool is here defined and calculated and from there you can deduce the
    propagation velocity of the cold pool. This can be put in parallel with the imposed wind shear.

    Attributes:
        BUOYANCY (xr.DataArray): Buoyancy (y,z,y,x)
        DEPTH_SHEAR (str): depth shear of the wind profile
        FMSE (xr.DataArray): Moist static energy (y,z,y,x)
        nt (int): number of time steps
        nx (int): number of x steps
        ny (int): number of y steps
        nz (int): number of z steps
        P (xr.DataArray): Pressure
        POTENTIAL_TEMPERATURE (xr.DataArray): Potential temperature (y,z,y,x)
        PRECi (xr.DataArray): Precipitation (t,y,x)
        QN (xr.DataArray): QN
        QPEVP (xr.DataArray): QVEPV
        QV (xr.DataArray): QV
        TABS (xr.DataArray): Absolute Temperature
        U (xr.DataArray): x component of velocity
        VIRTUAL_TEMPERATURE (xr.DataArray): Virual temperature
        VORTICITY (xr.DataArray): Vorticity
        W (xr.DataArray): z component of velocity
        X (xr.DataArray): x direction
        Y (xr.DataArray): y direction
        Z (xr.DataArray): z direction

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
        humidity_evp_2d: np.array = None,
        humidity_evp_2d_i: np.array = None,
        plot_mode: bool = False,  # plot_mode=True to partially load dataset and earn time,
    ):

        self.depth_shear = depth_shear
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
        self.QPEVP = humidity_evp
        self.QPEVP_2D = humidity_evp_2d
        self.QPEVPi = humidity_evp_2d_i

        self.nx = len(self.X)
        self.ny = len(self.Y)
        self.nz = len(self.Z)
        self.nt = len(self.TABS)

        if not plot_mode:
            self.FMSE = None
            self.VIRTUAL_TEMPERATURE = None
            self.POTENTIAL_TEMPERATURE = None
            self.BUOYANCY = None
            self.VORTICITY = None
            self.set_basic_variables_from_dataset()

    def save(self, path_to_save: str):
        """Will save the class as a pickle but will ignore attributes wwritten in UPPERCASE
            Those ones are loaded directly from dataset, so no need to store them in pickle

        Args:
            path_to_save (str): path and name of the backup
        """

        # SELECT ONLY attributes whose names are in lower cases
        black_list = [
            key for (key, value) in self.__dict__.items() if (key.isupper() or key == "PRECi")
        ]

        dictionary = {
            key: value for (key, value) in self.__dict__.items() if key not in black_list
        }

        file = open(path_to_save, "wb")
        pickle.dump(dictionary, file, 2)
        file.close()

    def load(self, path_to_load: str):
        """Load cold pool from pickle file

        Args:
            path_to_load (str): path of loaded file
        """
        file = open(path_to_load, "rb")
        tmp_dict = pickle.load(file)
        file.close()

        self.__dict__.update(tmp_dict)

    def set_basic_variables_from_dataset(self):
        """Compute basic variables from dataset variable

        One must care of dataset variable shape, here it is adapted to (nt,nz,ny,nx) data !
        this method is called ONLY IF self.plot_mode == FALSE
        """
        z_3d_in_time = pySAM.utils.expand_array_to_tzyx_array(
            time_dependence=False,
            input_array=self.Z.values,
            final_shape=np.array((self.nt, self.nz, self.ny, self.nx)),
        )

        # @@@@ UNUSED !!!
        # vertical_mean_temperature_3d_in_time = pySAM.utils.expand_array_to_tzyx_array(
        #     time_dependence=True,
        #     input_array=np.mean(self.TABS.values, axis=(2, 3)),
        #     final_shape=np.array((self.nt, self.nz, self.ny, self.nx)),
        # )

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
        y_margin: int,
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
        if data_name not in [
            "W",
            "QN",
            "VORTICITY",
            "BUOYANCY",
            "QPEVP",
            "QPEVP_2D",
            "QPEVPi",
            "QP",
            "rho_QPEVP",
        ]:
            raise ValueError(
                "data name must be in [W, QN, VORTICITY, BUOYANCY, QPEVP, QPEVP_2D, QPEVPi, QP, rho_QPEVP]"
            )
        if variable_to_look_for_extreme not in ["PRECi", "QPEVP_2D"]:
            raise ValueError("variable_to_look_for_extreme must be in [PRECi, QPEVP_2D]")

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

    def set_geometry_profile(
        self,
        data_name: str,
        threshold: float,
    ):
        """Computes the profile of the cold pool in the x and z plane

        Args:
            data_name (str): Name of the variable to catch the cold pool
            threshold (float): Variable threshold, below it is in cold pool, above not
        """
        if "_composite" not in data_name:
            raise ValueError("data must be composite variable")

        data_array = getattr(self, data_name)

        geometry_profile_line = geometry_profile(
            data_array=data_array,
            z_array=self.Z,
            cold_pool_threshold=threshold,
            vertical_level_0=pySAM.LOWEST_ATMOSPHERIC_LEVEL,
        )

        setattr(self, "profile_" + str(int(abs(threshold * 1000))), geometry_profile_line)
        setattr(
            self,
            "mean_height_" + str(int(abs(threshold * 1000))),
            np.mean(geometry_profile_line[1]),
        )

    def set_potential_energy(self, data_name: str, profile_name: str) -> np.array:
        """Computes the potentential energy of a cold pool as a function of x.
        For each x, provided data is integrated in z from the bottom to the top of the cold pool.

        Args:
            data_name (str): Name of the variable to integrate in z
            x_size (int): range of x you want to compute the potential energy

        Raises:
            ValueError: Data must be composite variable
        """
        if "_composite" not in data_name:
            raise ValueError("data must be composite variable")

        data_array = getattr(self, data_name)
        profile_list = getattr(self, profile_name)

        potential_energy_array = potential_energy(
            data_array=data_array, z_array=self.Z.values, geometry_profile=profile_list
        )

        setattr(self, "potential_energy", potential_energy_array)
        setattr(self, "mean_potential_energy", np.mean(potential_energy_array))
