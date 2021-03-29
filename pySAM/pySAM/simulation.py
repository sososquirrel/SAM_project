"""Simulation class, allows to post process outputs of SAM"""

import pickle

import xarray as xr
from pySAM.cold_pool.cold_pool import ColdPool
from pySAM.squall_line.squall_line import SquallLine
from pySAM.utils import color, color2


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

        self.run = run
        self.velocity = velocity
        self.depth_shear = depth_shear

        self.data_folder_paths = data_folder_paths

        xr.open_dataset(self.data_folder_paths[0])

        self.dataset_1d = xr.open_dataset(self.data_folder_paths[0], decode_cf=False)
        self.dataset_2d = xr.open_dataset(self.data_folder_paths[1], decode_cf=False)
        self.dataset_3d = xr.open_dataset(self.data_folder_paths[2], decode_cf=False)

        if not hasattr(self.dataset_3d, "QPEVP"):

            self.add_variable_to_dataset(
                dataset_name="dataset_3d",
                variable_name="QPEVP",
                variable_data_path="/Users/sophieabramian/Desktop/SAM_project/data/"
                + f"squall4/3D_FILES/QPEVP/RCE_shear_U{self.velocity}_H{self.depth_shear}"
                + f"_64_0000302400.com3D.alltimes_{self.run}_QPEVP.nc",
            )

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
            instantaneous_precipitation=self.dataset_2d.PRECi,
            x_positions=self.dataset_3d.x,
            y_positions=self.dataset_3d.y,
            z_positions=self.dataset_3d.z,
            x_velocity=self.dataset_3d.U,
            z_velocity=self.dataset_3d.W,
            cloud_base=self.dataset_3d.QN,
            humidity=self.dataset_3d.QV,
            pressure=self.dataset_1d.p,
            depth_shear=self.depth_shear,
            humidity_evp=self.dataset_3d.QPEVP,
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

    def load(self, backup_folder_path):
        """Load calculated attributes from pickle backup files

        Args:
            backup_folder_path (str): path to saved file
        """
        file = open(
            backup_folder_path
            + f"{self.run}/simulation/saved_simulation_U{self.velocity}_H{self.depth_shear}",
            "rb",
        )
        tmp_dict = pickle.load(file)
        file.close()
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
