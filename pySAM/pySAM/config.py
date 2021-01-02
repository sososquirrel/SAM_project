"""config file"""

import multiprocessing

import numpy as np

PI = 3.141592653
GRAVITY = 9.8
N_CPU = multiprocessing.cpu_count()

DATA_FOLDER_PATH = "/Users/sophieabramian/Desktop/SAM_project/data/"


# squall line parameters
MU_GAUSSIAN = np.array([0.0, 0.0])
SIGMA_GAUSSIAN = np.array([[1.0, -0.985], [-0.985, 1.0]])
