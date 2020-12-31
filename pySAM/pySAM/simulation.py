"""coucou"""

import numpy as np
import pySAM
import xarray as xr
from pySAM.squall_line.squall_line import SquallLine

print(pySAM.PI)


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
            data_folder_path + f"/{self.run}/1D_FILES/U{self.velocity}_H{self.depth_shear}"
        )
        self.path_field_2d = (
            data_folder_path + f"{self.run}/2D_FILES/U{self.velocity}_H{self.depth_shear}"
        )
        self.path_field_3d = (
            data_folder_path + f"{self.run}/3D_FILES/U{self.velocity}_H{self.depth_shear}"
        )

        self.dataset_1d = xr.open_dataset(self.path_field_1d, decode_cf=False, autoclose=True)
        self.dataset_2d = xr.open_dataset(self.path_field_2d, decode_cf=False, autoclose=True)
        self.dataset_3d = xr.open_dataset(self.path_field_3d, decode_cf=False, autoclose=True)

        # self.dataset_1d.close()
        # self.dataset_2d.close()
        # self.dataset_3d.close()

        self.squall_line = SquallLine(
            precipitable_water=self.dataset_2d.PW,
            instantaneous_precipitation=self.dataset_2d.PRECi,
            x_velocity=self.dataset_3d.U,
            y_velocity=self.dataset_3d.V,
            z_velocity=self.dataset_3d.W,
        )
