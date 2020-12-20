import os

import numpy as numpy
import xarray as xr
from netcdf4 import Dataset


class Simulation:
	def __init__(self, 'run'=run, 'velocity'=velocity, 'depth_shear'=depth_shear):
		
		self.velocity = velocity
		self.depth_shear = depth_shear
		self.run = run

		self.path_field_1D = f'/Users/sophieabramian/Desktop/SAM_project/data/{self.run}/1D_FILES/U{self.velocity}_H{self.depth_shear}'
		self.path_field_2D = '..'
		self.path_field_3D = '..'

		self.dataset_1D = xr.open_dataset(set.path_field_1D, decode_cf=False)
		self.dataset_2D = xr.open_dataset(set.path_field_2D, decode_cf=False)
		self.dataset_3D = xr.open_dataset(set.path_field_3D, decode_cf=False)

		self.dataset_1D.close()
		self.dataset_2D.close()
		self.dataset_3D.close()



self.squall_line = SquallLine('PW'=self.dataset_2D.PW, 'Preci'=self.dataset_2D.PRECi, 'U'=self.dataset_3D.U, 'V'=self.dataset_3D.V, 'W'=self.dataset_3D.W)
