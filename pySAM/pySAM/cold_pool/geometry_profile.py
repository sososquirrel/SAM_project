"""Base function to get cold pool profil from a composite image"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import xarray as xr


def cold_pool_contour(
    data_array: np.array,
    z_array: np.array,
    cold_pool_threshold: float,
    vertical_level_0: float,
):
    """Summary

    Args:
        data_array (np.array): Description
        x_array (np.array): Description
        z_array (np.array): Description
        cold_pool_threshold (float): Description
        vertical_level_0 (float): Description
    """
    # x_max_precip = int(
    #     data_array.shape[1] / 2
    # )  # remainder : the input must be centered in the maximum precipitation

    n_x_range = data_array.shape[1]
    x_array = np.linspace(-(n_x_range // 2), (n_x_range // 2), n_x_range)

    X, Z = np.meshgrid(x_array, z_array)

    contours = plt.contour(X, Z, data_array, [cold_pool_threshold])
    plt.close()

    list_index_guess_cold_pool = []
    for i in range(len(contours.allsegs[0])):
        if vertical_level_0 in contours.allsegs[0][i]:
            list_index_guess_cold_pool.append(i)

    max_length_of_cp_line, index_cold_pool = 0, 0
    for index in list_index_guess_cold_pool:
        if len(contours.allsegs[0][index]) > max_length_of_cp_line:
            max_length_of_cp_line = len(contours.allsegs[0][index])
            index_cold_pool = index

    return contours.allsegs[0][index_cold_pool]


def convert_contour_to_list(contour: plt.contour) -> (np.array, np.array):
    """Summary

    Args:
        contour (plt.contour): Description

    Returns:
        np.array, np.array: Description
    """
    abscisse_points = np.array([line[0] for line in contour])
    ordonate_points = np.array([line[1] for line in contour])

    print(abscisse_points)
    return abscisse_points, ordonate_points


def cold_pool_start_and_end(contour_list: list, x_array: np.array) -> (int, int):
    """Summary

    Args:
        contour_list (list): Description
        x_array (np.array): Description

    Returns:
        int, int: Description
    """
    idx_min, idx_max = (np.abs(x_array - contour_list[0][0])).argmin(), (
        np.abs(x_array - contour_list[0][-1])
    ).argmin()

    print(idx_min, idx_max)
    return idx_min + 1, idx_max - 1


def interpolate_to_my_data(
    contour_list: list, x_array: np.array, idx_list: list
) -> (np.array, np.array):
    """Summary

    Args:
        contour_list (list): Description
        x_array (np.array): Description

    Returns:
        np.array, np.array: Description
    """
    interpolated_array = scipy.interpolate.interp1d(contour_list[0], contour_list[1])

    idx_min, indx_max = idx_list[0], idx_list[1]

    return x_array[idx_min:indx_max], interpolated_array(x_array[idx_min:indx_max])


def geometry_profile(
    data_array: np.array,
    z_array: np.array,
    cold_pool_threshold: float,
    vertical_level_0: float,
) -> (np.array, np.array):
    """Summary

    Args:
        data_array (np.array): Description
        z_array (np.array): Description
        cold_pool_threshold (float): Description
        vertical_level_0 (float): Description
    """
    contour = cold_pool_contour(
        data_array=data_array,
        z_array=z_array,
        cold_pool_threshold=cold_pool_threshold,
        vertical_level_0=vertical_level_0,
    )

    n_x_range = data_array.shape[1]
    final_x_points = np.linspace(-(n_x_range // 2), (n_x_range // 2), n_x_range)

    contour_list = convert_contour_to_list(contour=contour)

    idx_min, idx_max = cold_pool_start_and_end(
        contour_list=contour_list, x_array=final_x_points
    )

    x_points_profil, y_points_profil = interpolate_to_my_data(
        contour_list=contour_list, x_array=final_x_points, idx_list=[idx_min, idx_max]
    )

    final_y_points = vertical_level_0 * np.ones(n_x_range)
    final_y_points[idx_min:idx_max] = y_points_profil

    return final_x_points, final_y_points
