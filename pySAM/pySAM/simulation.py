"""coucou"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pySAM
import xarray as xr
from pySAM.cold_pool.cold_pool import ColdPool
from pySAM.squall_line.squall_line import SquallLine
from pySAM.utils import color


class Simulation:

    """Summary

    Attributes:
        dataset_1d (TYPE): DescriphuihdziuhEIUDHUZItion
        dataset_2d (TYPE): Description
        dataset_3d (TYPE): Description

        depth_shear (TYPE): Description
        path_field_1d (TYPE): Description
        path_field_2d (TYPE): Description
        path_field_3d (TYPE): Description
        run (TYPE): Description
        squall_line (TYPE): Description
        velocity (TYPE): Description

    Deleted Attributes:
        run (TYPE): Description
    """

    def __init__(self, data_folder_path: str, run: str, velocity: str, depth_shear: str):
        """Init"""

        self.run = run
        self.velocity = velocity
        self.depth_shear = depth_shear

        # data_folder_path = "/Users/sophieabramian/Desktop/SAM_project/data"
        self.path_field_1d = (
            data_folder_path
            + f"{self.run}/1D_FILES/RCE_shear_U{self.velocity}_H{self.depth_shear}_{self.run}.nc"
        )
        self.path_field_2d = (
            data_folder_path
            + f"/{self.run}/2D_FILES/RCE_shear_U{self.velocity}_H{self.depth_shear}_64.2Dcom_1_{self.run}.nc"
        )
        self.path_field_3d = (
            data_folder_path
            + f"{self.run}/3D_FILES/RCE_shear_U{self.velocity}_H{self.depth_shear}_64_0000302400.com3D.alltimes_{self.run}.nc"
        )

        self.dataset_1d = xr.open_dataset(self.path_field_1d, decode_cf=False)
        self.dataset_2d = xr.open_dataset(self.path_field_2d, decode_cf=False)
        self.dataset_3d = xr.open_dataset(self.path_field_3d, decode_cf=False)

        # self.dataset_1d.close()
        # self.dataset_2d.close()
        # self.dataset_3d.close()

        self.color = color(self.velocity, self.depth_shear)

        self.add_variable_to_dataset(
            dataset_name="dataset_3d",
            variable_name="QPEVP",
            variable_data_path=data_folder_path
            + f"squall4/3D_FILES/QPEVP/RCE_shear_U{self.velocity}_H{self.depth_shear}_64_0000302400.com3D.alltimes_{self.run}_QPEVP.nc",
        )

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
        f = open(
            backup_folder_path
            + f"{self.run}/simulation/saved_simulation_U{self.velocity}_H{self.depth_shear}",
            "rb",
        )
        tmp_dict = pickle.load(f)
        f.close()
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
        # save netcdf4 data
        # self.dataset_1D.to_netcdf(path=self.path_fields_1D, mode='w')
        # self.dataset_2D.to_netcdf(path=self.path_fields_2D, mode='w')

        # save other type of data in pickles
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
        f = open(
            backup_folder_path
            + f"{self.run}/simulation/saved_simulation_U{self.velocity}_H{self.depth_shear}",
            "wb",
        )
        pickle.dump(dict2, f, 2)
        f.close()

        self.squall_line.save(
            backup_folder_path
            + f"{self.run}/squall_line/saved_squall_line_U{self.velocity}_H{self.depth_shear}"
        )
        self.cold_pool.save(
            backup_folder_path
            + f"{self.run}/cold_pool/saved_cold_pool_U{self.velocity}_H{self.depth_shear}"
        )
