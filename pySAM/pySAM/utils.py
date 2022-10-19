"""Useful functions for any class"""


import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from multiprocess import Pool


def make_parallel(function, nprocesses):
    """Works similar to a decorator to paralelize "stupidly parallel"
    problems. Decorators and multiprocessing don't play nicely because
    of naming issues.

    Inputs
    ======
    function : the function that will be parallelized. The FIRST
        argument is the one to be iterated on (in parallel). The other
        arguments are the same in all the parallel runs of the function
        (they can be named or unnamedarguments).
    nprocesses : int, the number of processes to run. Default is None.
        It is passed to multiprocessing.Pool (see that for details).

    Output
    ======
    A paralelized function. DO NOT NAME IT THE SAME AS THE INPUT
    FUNCTION.

    SEE EXAMPLE BELOW

    """

    def apply(
        iterable_values_1,
        *args,
        iterable_values_2=None,
        **kwargs,
    ):
        if type(iterable_values_1) not in [
            list,
            np.array,
            xr.core.dataarray.DataArray,
            np.ndarray,
        ]:
            print(iterable_values_1)
            raise ValueError(
                "Your first iterable value type is not standard, must be in [list, np.array, xarray.core.dataarray.DataArray, np.ndarray,]"
            )

        if type(iterable_values_1) not in [
            list,
            np.array,
            np.ndarray,
        ]:
            iterable_values_1 = iterable_values_1.values

        # pylint: disable=E1102
        args = list(args)
        processes_pool = Pool(nprocesses)

        if iterable_values_2 is not None:
            if type(iterable_values_2) not in [
                list,
                np.array,
                xr.core.dataarray.DataArray,
                np.ndarray,
            ]:
                raise ValueError(
                    "Your second iterable value type is not standard, must be in [list, np.array, xarray.core.dataarray.DataArray, np.ndarray,]"
                )

            if type(iterable_values_2) not in [
                list,
                np.array,
                np.ndarray,
            ]:
                iterable_values_2 = iterable_values_2.values

            results = [
                processes_pool.apply_async(function, args=[value1, value2] + args, kwds=kwargs)
                for (value1, value2) in zip(iterable_values_1, iterable_values_2)
            ]

        elif iterable_values_2 is None:
            results = [
                processes_pool.apply_async(function, args=[value1] + args, kwds=kwargs)
                for value1 in iterable_values_1
            ]

        processes_pool.close()
        # processes_pool.terminate()

        return [r.get() for r in results]

    return apply


def expand_array_to_tzyx_array(
    time_dependence: bool, input_array: np.array, final_shape: np.array
) -> np.array:
    """Get a (nt,nz,ny,nx) array form a (nt,nz) array
    It makes data get volumic

    It works also if the data only depends on z

    Args:
        time_dependence (bool): True if the data is of shape of (nt,nz), False if shape=(nz)
        input_array (np.array): the array you want to extend
        final_shape (np.array): dimension of your willing array, must be (nt,nz,ny,nx) in this order
    """
    if len(final_shape) != 4:
        raise ValueError("Output must be (t,z,y,x) type")

    if not time_dependence:
        if len(input_array.shape) != 1:
            raise ValueError("Input array with no time dependence must be one-dimensionnal")

        if input_array.shape[0] != final_shape[1]:
            raise ValueError(
                "z length of final shape must be equal to the length of input array"
            )

        output_array = input_array[None, :, None, None]

        output_array = np.repeat(output_array, final_shape[0], axis=0)
        output_array = np.repeat(output_array, final_shape[2], axis=2)
        output_array = np.repeat(output_array, final_shape[3], axis=3)

    else:
        if len(input_array.shape) != 2:
            raise ValueError("Input array with time dependence must be 2-dimensionnal")

        if input_array.shape != tuple(final_shape[:2]):
            raise ValueError(
                "time and z lengths of final shape must be equal to the length of input array"
            )

        output_array = input_array[:, :, None, None]

        output_array = np.repeat(output_array, final_shape[2], axis=2)
        output_array = np.repeat(output_array, final_shape[3], axis=3)

    return output_array


def expand_array_to_zyx_array(input_array: np.array, final_shape: np.array) -> np.array:
    """Get a (nz,ny,nx) array form a (nz) array
    It makes data get volumic

    Args:
        time_dependence (bool): True if the data is of shape of (nt,nz), False if shape=(nz)
        input_array (np.array): the array you want to extend
        final_shape (np.array): dimension of your willing array, must be (nt,nz,ny,nx) in this order
    """
    if len(final_shape) != 3:
        raise ValueError("Output must be (z,y,x) type")

    output_array = input_array[:, None, None]

    output_array = np.repeat(output_array, final_shape[1], axis=1)
    output_array = np.repeat(output_array, final_shape[2], axis=2)

    return output_array


def color(velocity: str):
    """Returns a specific color for each simulation, very convenient for plot

    Args:
        velocity (str): Basal velocity
        depth (str): Depth of the shear

    Returns:
        TYPE: cmap color
    """
    if velocity == "0":
        return "grey"

    cmap = plt.cm.get_cmap("hsv")
    return cmap(float(velocity) / 20)


def color2(velocity: str):
    """Returns a specific color for each simulation, very convenient for plot

    Args:
        velocity (str): Basal velocity
        depth (str): Depth of the shear

    Returns:
        TYPE: cmap color
    """
    if velocity == "0":
        return "grey"

    cmap = plt.cm.get_cmap("Spectral_r")
    return cmap(float(velocity) / 20)


def generate_1d_2d_3d_paths(run: str, velocity: str, depth_shear: str, data_folder_path: str):
    """Generates paths to 1D, 2D and 3D data fields

    Args:
        run (str): Name of the simulation set
        velocity (str): Basal velocity of the linear wind shear profile
        depth_shear (str): Depth of shear of the linear wind shear profiles
        data_folder_path (str): Parent folder where all datasets are

    Returns:
        TYPE: List of 3 paths [path_field_1d, path_field_2d, path_field_3d]
    """

    if depth_shear == "inf":
        path_field_1d = data_folder_path + f"{run}/1D_FILES/RCE_shear_U{velocity}_{run}.nc"
        path_field_2d = (
            data_folder_path + f"/{run}/2D_FILES/RCE_shear_U{velocity}_64.2Dcom_1_{run}.nc"
        )

        path_field_3d = (
            data_folder_path
            + f"{run}/3D_FILES/RCE_shear_U{velocity}_64_0000302400.com3D.alltimes_{run}.nc"
        )

    else:

        path_field_1d = (
            data_folder_path + f"{run}/1D_FILES/RCE_shear_U{velocity}_H{depth_shear}_{run}.nc"
        )
        path_field_2d = (
            data_folder_path
            + f"/{run}/2D_FILES/RCE_shear_U{velocity}_H{depth_shear}_64.2Dcom_1_{run}.nc"
        )

        path_field_3d = (
            data_folder_path
            + f"{run}/3D_FILES/RCE_shear_U{velocity}_H{depth_shear}_64_0000302400.com3D.alltimes_{run}.nc"
        )

    return path_field_1d, path_field_2d, path_field_3d


def max_point_wise(matrix_1: np.array, matrix_2: np.array):
    """Summary

    Args:
        matrix_1 (np.array): first matrix
        matrix_2 (np.array): second matrix

    Returns:
        TYPE: Description
    """
    matrix_output = np.copy(matrix_1)

    matrix_output[matrix_2 - matrix_1 > 0] = matrix_2[matrix_2 - matrix_1 > 0]

    return matrix_output


def min_point_wise(matrix_1: np.array, matrix_2: np.array):
    """Summary

    Args:
        matrix_1 (np.array): first matrix
        matrix_2 (np.array): second matrix

    Returns:
        TYPE: Description
    """
    matrix_output = np.copy(matrix_1)

    matrix_output[matrix_2 - matrix_1 < 0] = matrix_2[matrix_2 - matrix_1 < 0]

    return matrix_output


def create_reduced_nb_timestep_netcdf4_for_test(
    input_netcdf4_file: str, output_name: str
):  # pragma: no cover
    """Create a very small netcdf4 file for testing only

    Args:
        input_netcdf4_file (str): path for the big netcdf4 file
        output_name (str): name of the ouput netcdf4 file
    """
    data = xr.open_dataset(input_netcdf4_file, decode_cf=False)
    dataset_out = data.loc[dict(time=[30.0, 30.04167, 30.08333])]
    dataset_out.to_netcdf(output_name)


def get_integrated_quantity(density: np.array, z_array: np.array, quantity: np.array):

    if quantity.shape[0] != density.shape[0]:
        nt = quantity.shape[0]
        nt_long = density.shape[0]
        density = density[nt_long - nt :, :]

    if z_array.shape[0] != density.shape[1]:
        nz = z_array.shape[0]
        density = density[:, :nz]

    rho = density
    rho_3D = np.tile(rho, (128, 128, 1, 1))
    rho_3D = rho_3D.T
    rho_3D = np.swapaxes(rho_3D, 0, 1)

    product_rho_qpsrc = quantity * rho_3D

    dz = np.gradient(z_array.T)
    dz_3D = np.tile(dz, (121, 128, 128, 1))
    dz_3D = np.swapaxes(dz_3D, 1, 3)

    integrated_quantity = np.sum(product_rho_qpsrc * dz_3D, axis=1)

    return integrated_quantity


def distribution_tail(data: np.array, number_of_nines: int):
    start = 0.9
    list_of_percentile = [start]
    for i in range(9 * number_of_nines):
        next_value = list_of_percentile[-1] + 0.01 / (10 ** (i // 9))
        list_of_percentile.append(next_value)

    value_of_percentile = [np.quantile(data, i) for i in list_of_percentile]

    list_of_percentile = np.array(list_of_percentile)

    return list_of_percentile, value_of_percentile


def mass_flux(density: np.array, vertical_velocity: np.array):

    nt, nz, ny, nx = vertical_velocity.shape

    # make density a 3D variable
    if len(density.shape) == 2:
        if density.shape[0] != nt:
            nt_longer = density.shape[0]
            density = density[nt_longer - nt :, :]

        if density.shape[1] != nz:
            density = density[:, :nz]

        density_3D = np.tile(density, (ny, nx, 1, 1))
        density_3D = density_3D.T
        density_3D = np.swapaxes(density_3D, 0, 1)

        # calculate the mass flux
        ## the grid for vertical velocity and density are mismatched, explain why 0.5
        mass_flux = (
            vertical_velocity[:, 1:, :, :]
            * 0.5
            * (density_3D[:, 1:, :, :] + density_3D[:, :-1, :, :])
        )

        return mass_flux

    else:
        print("density must be rho(t,z)")
