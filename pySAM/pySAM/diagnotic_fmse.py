import numpy as np
import xarray as xr


def diagnostic_fmse_z(
    fmse_array: np.array,
    z_array: np.array,
    data_array: np.array,
    time_step: int,
    nb_bins_fmse: int = 50,
    fmse_range: str = "max",
):

    if type(data_array) not in [
        list,
        np.array,
        xr.core.dataarray.DataArray,
        np.ndarray,
    ]:
        raise ValueError(
            "data_array type is not standard, must be in [list, np.array, xarray.core.dataarray.DataArray, np.ndarray,]"
        )

    if type(data_array) not in [
        list,
        np.array,
        np.ndarray,
    ]:
        data_array = data_array.values

    if fmse_range not in ["max", "1-percentile"]:
        raise ValueError("fmse_range must be in [max, 1_percentile]")

    nz = z_array.shape[0]
    output_matrix = np.zeros((nz, nb_bins_fmse))

    data_array_i = data_array[time_step]
    fmse_array_i = fmse_array[time_step]

    if fmse_range == "max":
        total_min, total_max = (np.min(fmse_array_i), np.max(fmse_array_i))

    if fmse_range == "1-percentile":
        total_min, total_max = (np.percentile(fmse_array_i, 1), np.percentile(fmse_array_i, 99))

    total_range = np.linspace(total_min, total_max, nb_bins_fmse)

    for zz in range(nz - 1):

        ind_xy = np.array(
            [
                np.where(
                    np.logical_and(
                        total_range[i] <= fmse_array_i[zz],
                        fmse_array_i[zz] <= total_range[i + 1],
                    )
                )
                for i in range(total_range.shape[0] - 1)
            ],
            dtype="object",
        )

        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            foo = np.nanmean(x, axis=1)

        mean_value_zz_fmse = np.array(
            [np.mean(data_array_i[zz, ind_xy[i][0], ind_xy[i][1]]) for i in range(len(ind_xy))]
        )

        output_matrix[zz, 1:] = mean_value_zz_fmse

    return output_matrix
