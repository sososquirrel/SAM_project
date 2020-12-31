import numpy as numpy
import pySAM
import xarray as xr
from pySAM.squall_line.squall_line import SquallLine

print(pySAM.PI)


class Simulation:
    def __init__(self, base_path, run, velocity, depth_shear):

        self.run = run
        self.velocity = velocity
        self.depth_shear = depth_shear

        # base_path = "/Users/sophieabramian/Desktop/SAM_project/data"
        self.path_field_1d = (
            base_path + f"/{self.run}/1D_FILES/U{self.velocity}_H{self.depth_shear}"
        )
        self.path_field_2d = (
            base_path + f"{self.run}/2D_FILES/U{self.velocity}_H{self.depth_shear}"
        )
        self.path_field_3d = (
            base_path + f"{self.run}/3D_FILES/U{self.velocity}_H{self.depth_shear}"
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
