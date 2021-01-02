"""config file"""

import multiprocessing

import numpy as np

PI = 3.141592653


# universal constant
GRAVITY = 9.8  # (m/s2)

# computer constant
N_CPU = multiprocessing.cpu_count()

# data folder
DATA_FOLDER_PATH = "/Users/sophieabramian/Desktop/SAM_project/data/"

# air properties
HEAT_CAPACITY_AIR = 1004  # (J/K/kg)
LATENT_HEAT_AIR = 2.5 * 10 ** 6  # (J/kg)
GAS_CONSANT_AIR = 287  # (J/K/kg)

# vapour propoerties
GAS_CONSTANT_WATER_VAPOR = 461.5  # (J/K/kg)

# mixing ratio
MIXING_RATIO_AIR_WATER_VAPOR = 0.622  # (%)


# squall line parameters
THETA_ARRAY = np.linspace(0, np.pi, 50)
MU_GAUSSIAN = np.array([0.0, 0.0])
SIGMA_GAUSSIAN = np.array([[1.0, -0.985], [-0.985, 1.0]])
