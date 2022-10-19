"""Simulation class, allows to post process outputs of SAM"""

import pickle

import numpy as np
import pySAM
import xarray as xr
from pySAM import config
from pySAM.cape.cape_functions import get_parcel_ascent
from pySAM.cold_pool.cold_pool import ColdPool
from pySAM.cold_pool.composite import instant_mean_extraction_data_over_extreme
from pySAM.squall_line.squall_line import SquallLine
from pySAM.utils import color, color2, distribution_tail, make_parallel, mass_flux


class Simulation:

    """Simulation creates an object that gathers all the datasets from simulation.
    It is designed here to study the convective response to linear wind shear profil.
    Squall lines and cold pools are the main features to be studied in this case.
    An object Simulation represents 3Gb of data, and default settings load all the datasets.
    Everything you calculated is saved in a pickle. Methods save and load provide access to these previous calculus.
    You have a "plot mode" that allows you to partially load dataset and get access to your previous calculus.

    For plotting, a specify color is associated with each case. This might be very useful to keep consistence between plots.

    Attributes:
        cold_pool (ColdPool): gathers all variable and methods to analyse cold pool and describe it
        color (cmap): matplotlib color defined for each case
        dataset_1d (xr.Datastet): Dataset with all 1d variables from SAM
        dataset_2d (xr.Datastet): Dataset with all 2d variables from SAM
        dataset_3d (xr.Datastet): Dataset with all 3d variables from SAM

        depth_shear (str): depth shear of the linear wind shear profil
        path_field_1d (str): path to the dataset_1d storage
        path_field_2d (str): path to the dataset_2d storage
        path_field_3d (str): path to the dataset_3d storage
        run (str): name of the simulation set
        squall_line (SquallLine): gathers all variable and methods to analyse squall line and describe it
        velocity (str): basal velocity of the linear wind shear profil

    """

    def __init__(
        self,
        data_folder_paths: list,
        run: str,
        velocity: str,
        depth_shear: str,
        plot_mode: bool = False,
    ):
        """Init"""
        # print("COUCOU SOSO ! CLASSE CETTE CLASSE !")

        self.run = run
        self.velocity = velocity
        self.depth_shear = depth_shear

        self.data_folder_paths = data_folder_paths

        self.dataset_1d = xr.open_dataset(self.data_folder_paths[0], decode_cf=False)
        self.dataset_2d = xr.open_dataset(self.data_folder_paths[1], decode_cf=False)
        self.dataset_3d = xr.open_dataset(self.data_folder_paths[2], decode_cf=False)

        """
        if not hasattr(self.dataset_3d, "QPEVP"):

            if self.velocity == "0":
                self.add_variable_to_dataset(
                    dataset_name="dataset_3d",
                    variable_name="QPEVP",
                    variable_data_path="/Users/sophieabramian/Desktop/SAM_project/data/"
                    + f"squall4/3D_FILES/QPEVP/RCE_shear_U{self.velocity}"
                    + f"_64_0000302400.com3D.alltimes_{self.run}_QPEVP.nc",
                )
            else:
                self.add_variable_to_dataset(
                    dataset_name="dataset_3d",
                    variable_name="QPEVP",
                    variable_data_path="/Users/sophieabramian/Desktop/SAM_project/data/"
                    + f"squall4/3D_FILES/QPEVP/RCE_shear_U{self.velocity}_H{self.depth_shear}"
                    + f"_64_0000302400.com3D.alltimes_{self.run}_QPEVP.nc",
                )

        """
        # self.dataset_1d.close()
        # self.dataset_2d.close()
        # self.dataset_3d.close()

        self.color = color(self.velocity)

        self.color_2 = color2(self.velocity)

        self.squall_line = SquallLine(
            precipitable_water=self.dataset_2d.PW,
            instantaneous_precipitation=self.dataset_2d.PRECi,
            x_velocity=self.dataset_3d.U,
            # y_velocity=self.dataset_3d.V,
            z_velocity=self.dataset_3d.W,
        )

        self.cold_pool = ColdPool(
            absolute_temperature=self.dataset_3d.TABS,
            precipitation=self.dataset_2d.Prec,
            x_positions=self.dataset_3d.x,
            y_positions=self.dataset_3d.y,
            z_positions=self.dataset_3d.z,
            x_velocity=self.dataset_3d.U,
            z_velocity=self.dataset_3d.W,
            cloud_base=self.dataset_3d.QN,
            humidity=self.dataset_3d.QV,
            pressure=self.dataset_1d.p,
            depth_shear=self.depth_shear,
            humidity_evp=None,
            rho=self.dataset_1d.RHO,
            precip_source=self.dataset_2d.QPSRC,
            plot_mode=plot_mode,
        )

    def add_variable_to_dataset(
        self, dataset_name: str, variable_name: str, variable_data_path: str
    ):
        """adds a netcdf4 variable to a xarray dataset

        Args:
            dataset_name (str): name of the dataset, must be in ['dataset_1d, dataset_2d, dataset_3d']
            variabla_name (str): name of the new variable
            variable_data (np.array): path to get the data
        """
        dataset = getattr(self, dataset_name)
        data_array = xr.open_dataarray(variable_data_path)
        dataset[variable_name] = data_array

    def load(self, backup_folder_path, load_all: bool = True):
        """Load calculated attributes from pickle backup files

        Args:
            backup_folder_path (str): path to saved file
        """

        if self.velocity == "0":

            file = open(
                backup_folder_path + f"{self.run}/simulation/saved_simulation_U{self.velocity}",
                "rb",
            )
        else:
            file = open(
                backup_folder_path
                + f"{self.run}/simulation/saved_simulation_U{self.velocity}_H{self.depth_shear}",
                "rb",
            )

        tmp_dict = pickle.load(file)
        file.close()

        your_blacklisted_set = []

        if not load_all:

            tmp_dict = [
                (key, value) for (key, value) in tmp_dict if key not in your_blacklisted_set
            ]

        self.__dict__.update(tmp_dict)

        self.squall_line.load(
            backup_folder_path
            + f"{self.run}/squall_line/saved_squall_line_U{self.velocity}_H{self.depth_shear}"
        )
        self.cold_pool.load(
            backup_folder_path
            + f"{self.run}/cold_pool/saved_cold_pool_U{self.velocity}_H{self.depth_shear}"
        )

        # self.initialize()

    def save(self, backup_folder_path):
        """Save current instances of the class except starting datasets

        Args:
            backup_folder_path (str): path to the saving file
        """
        your_blacklisted_set = [
            "dataset_1d",
            "dataset_2d",
            "dataset_3d",
            "squall_line",
            "cold_pool",
        ]
        dict2 = [
            (key, value)
            for (key, value) in self.__dict__.items()
            if key not in your_blacklisted_set
        ]

        if self.velocity == "0":

            file = open(
                backup_folder_path + f"{self.run}/simulation/saved_simulation_U{self.velocity}",
                "wb",
            )

        else:
            file = open(
                backup_folder_path
                + f"{self.run}/simulation/saved_simulation_U{self.velocity}_H{self.depth_shear}",
                "wb",
            )
        pickle.dump(dict2, file, 2)
        file.close()

        self.squall_line.save(
            backup_folder_path
            + f"{self.run}/squall_line/saved_squall_line_U{self.velocity}_H{self.depth_shear}"
        )

        self.cold_pool.save(
            backup_folder_path
            + f"{self.run}/cold_pool/saved_cold_pool_U{self.velocity}_H{self.depth_shear}"
        )

    def get_cape(
        self,
        temperature: str = "TABS",
        vertical_array: str = "z",
        pressure: str = "p",
        humidity_ground: str = "QV",
        parallelize: bool = True,
        set_parcel_ascent_composite_1d: bool = True,
    ) -> np.array:

        # get the variable in np.array
        temperature_array = getattr(self.dataset_3d, temperature).values

        nt, nz, ny, nx = temperature_array.shape

        z_array = getattr(self.dataset_3d, vertical_array).values
        pressure_array = getattr(self.dataset_1d, pressure).values[:nz]
        humidity_ground_array = getattr(self.dataset_3d, humidity_ground).values[:, 0, :, :]

        # calculate parcel_ascent
        if parallelize:
            parallel_parcel_ascent = make_parallel(
                function=get_parcel_ascent, nprocesses=config.N_CPU
            )
            parcel_ascent = parallel_parcel_ascent(
                iterable_values_1=temperature_array,
                iterable_values_2=humidity_ground_array / 1000,
                pressure=pressure_array,
                vertical_array=z_array,
            )

        else:
            parcel_ascent = []

            for temperature_i, humidity_ground_i in zip(
                temperature_array,
                humidity_ground_array / 1000,
            ):

                T_parcel_i = get_parcel_ascent(
                    temperature=temperature_i,
                    humidity_ground=humidity_ground_i,
                    pressure=pressure_array,
                    vertical_array=z_array,
                )

                parcel_ascent.append(T_parcel_i)

        parcel_ascent = np.array(parcel_ascent)

        dz = np.gradient(z_array)
        dz_3d = pySAM.utils.expand_array_to_tzyx_array(
            input_array=dz, time_dependence=False, final_shape=temperature_array.shape
        )

        cape = pySAM.GRAVITY * np.sum(
            dz_3d
            * pySAM.utils.max_point_wise(
                matrix_1=np.zeros_like(parcel_ascent),
                matrix_2=((parcel_ascent - temperature_array) / temperature_array),
            ),
            axis=1,
        )

        setattr(self, "cape", cape)

        if set_parcel_ascent_composite_1d:
            setattr(self, "parcel_ascent", parcel_ascent)

            self.set_composite_variables(
                data_name="parcel_ascent",
                variable_to_look_for_extreme="cape",
                extreme_events_choice="max",
                x_margin=40,
                y_margin=40,
                dataset_for_variable_2d="",
                dataset_for_variable_3d="",
                return_1D=True,
            )

            delattr(self, "parcel_ascent")

        else:
            None

    def set_composite_variables(
        self,
        data_name: str,
        variable_to_look_for_extreme: str,
        extreme_events_choice: str,
        x_margin: int,
        y_margin: int,
        parallelize: bool = True,
        return_3D: bool = False,
        dataset_for_variable_2d: str = "dataset_2d",
        dataset_for_variable_3d: str = "dataset_3d",
        return_1D: bool = False,
    ) -> np.array:

        """Compute the composite, namely the conditionnal mean, of 2d or 3d variables evolving in time
        This method builds attribute

        Args:
            data_name (str): name of the variable composite method is applying to
            variable_to_look_for_extreme (str): name of the variable that describe extreme event
            extreme_events_choice (str): max 1-percentile or 10-percentile
            x_margin (int): width of window zoom
            y_margin (int, optional): depth of window zoom
            parallelize (bool, optional): use all your cpu power
        """

        if dataset_for_variable_3d == "":
            data_3d = getattr(self, data_name)
        else:
            data_3d = getattr(getattr(self, dataset_for_variable_3d), data_name)

        if dataset_for_variable_2d == "":
            data_2d = getattr(self, variable_to_look_for_extreme)
        else:
            data_2d = getattr(
                getattr(self, dataset_for_variable_2d), variable_to_look_for_extreme
            )

        if type(data_2d) == xr.core.dataarray.DataArray:
            data_2d = data_2d.values

        if type(data_3d) == xr.core.dataarray.DataArray:
            data_3d = data_3d.values

        if parallelize:

            parallel_composite = make_parallel(
                function=instant_mean_extraction_data_over_extreme, nprocesses=pySAM.N_CPU
            )
            composite_variable = parallel_composite(
                iterable_values_1=data_3d,
                iterable_values_2=data_2d,
                extreme_events_choice=extreme_events_choice,
                x_margin=x_margin,
                y_margin=y_margin,
                return_3D=return_3D,
            )

        else:  # NO PARALLELIZATION
            composite_variable = []
            data = data_3d
            variable_to_look_for_extreme = data_2d
            for image, variable_extreme in zip(data, variable_to_look_for_extreme):
                composite_variable.append(
                    instant_mean_extraction_data_over_extreme(
                        data=image,
                        variable_to_look_for_extreme=variable_extreme,
                        extreme_events_choice=extreme_events_choice,
                        x_margin=x_margin,
                        y_margin=y_margin,
                        return_3D=return_3D,
                    )
                )

        composite_variable = np.array(composite_variable)
        composite_variable = np.mean(composite_variable, axis=0)

        if return_3D:
            setattr(
                self,
                data_name + "_composite_" + variable_to_look_for_extreme + "_3D",
                composite_variable,
            )
        else:
            setattr(
                self,
                data_name + "_composite_" + variable_to_look_for_extreme,
                composite_variable,
            )
        if return_1D:
            if return_3D:
                setattr(
                    self,
                    data_name + "_composite_" + variable_to_look_for_extreme + "_1D",
                    composite_variable[:, x_margin, y_margin],
                )
            else:
                setattr(
                    self,
                    data_name + "_composite_" + variable_to_look_for_extreme + "_1D",
                    composite_variable[:, x_margin],
                )

    def set_distribution_tail(
        self,
        variable_name: str,
        dataset_variable: str,
        number_of_nines: int,
        no_dataset: bool = False,
    ):

        if no_dataset:

            variable = getattr(self, variable_name)

        else:

            variable = getattr(getattr(self, dataset_variable), variable_name)

        output_list = distribution_tail(data=variable, number_of_nines=number_of_nines)

        setattr(
            self,
            variable_name + "_distribution_tail_" + str(number_of_nines),
            output_list,
        )

    def set_mass_flux(self, mass_name: str, vertical_velocity_name: str):

        density_variable = getattr(getattr(self, "dataset_1d"), mass_name)

        vertical_velocity = getattr(getattr(self, "dataset_3d"), vertical_velocity_name)

        mass_flux_3D = mass_flux(density=density_variable, vertical_velocity=vertical_velocity)

        setattr(
            self,
            "RHO_W",
            mass_flux_3D.values,
        )

    def set_gradient_qsat(self, humidity_name: str):
        humidity = getattr(getattr(self, "dataset_3d"), humidity_name)
        dqdz = -np.diff(humidity.values, axis=1)

        setattr(
            self,
            "DQV",
            dqdz,
        )
