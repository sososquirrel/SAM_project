"""Base functions for computing potential energy"""

import numpy as np


def hight_max_index(z_array: np.array, depth_shear: str) -> int:
    """Returns the maximum index of the cold pool hight, which is the upper boundary for integrate buoyancy.

    Args:
        z_array (np.array): vertical array of shape (nz,)
        depth_shear (str): depth shear of imposed profil

    Returns:
        int: Index of the cold pool hight, is over-evaluate
    """
    cold_pool_hight_max = 0.8 * float(
        depth_shear
    )  # cold pools are known to scale depth shear, here we take 0.8 depth shear depth

    cold_pool_hight_max_index = np.where(z_array < cold_pool_hight_max)[0][-1]

    return cold_pool_hight_max_index


def potential_energy(
    data_array: np.array, z_array: np.array, x_size: int, depth_shear: str
) -> np.array:
    """Return the energy potential of cold pool as a function of x, the imposed flow direction.
    Common input is buoyancy composite, but you can also use temperature anomaly.
    X is a regular spaced array, that start at the extreme left of the cold
    (generally the maximu of precipitation), and end 10's km to the right.
    The output is the intgrale of buoyancy composite over the cold pool domains

    Args:
        data_array (np.array): Buoyancy composite, temperature composite, of shape (nz,nx). data_array[nx//2] must be max of precipitation
        z_array (np.array): vertical array, of shape (nz,)
        x_size (int): typically half of the length of your cold pool in x direction
        depth_shear (str): cold pools are known to scale depth shear, 1.5 of depth shear will be the upper boudnary for integration

    Returns:
        potential_energy_array (np.array) : energy potential off the cold pool as a funciton of x, of shape (x_size,)

    """
    potential_energy_array = []

    x_max_precip = int(
        data_array.shape[1] / 2
    )  # remainder : the input must be centered in the maximum precipitation

    cold_pool_hight_max_index = hight_max_index(z_array=z_array, depth_shear=depth_shear)

    for x_index in range(x_size):
        data_array_x = data_array[:cold_pool_hight_max_index, x_max_precip + x_index]

        if len(np.where(data_array_x < 0)[0]) == 0:
            potential_energy_x = 0

        else:
            y_intersect_index = np.where(data_array_x < -0.0005)[0][-1]

            dz = np.diff(z_array[: y_intersect_index + 1])

            potential_energy_x = np.sum(
                -data_array[:y_intersect_index, x_max_precip + x_index] * dz
            )
        potential_energy_array.append(potential_energy_x)

    return np.array(potential_energy_array)
