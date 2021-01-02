"""Base functions for compute composite plots"""

import numpy as np
from scipy import signal


def extract_circular_block(
    data: np.array,
    x_index_middle_array: np.array,
    x_margin: int,
    y_index_middle_array: np.array = None,
    y_margin: int = None,
):
    if len(data.shape) == 3 and y_index_middle_array == None and y_margin == None:
        raise ValueError("3D data requires y_index_middle_array and y_margin")

    if len(data.shape) == 2 and y_index_middle_array != None and y_margin != None:
        raise ValueError("no y_index_middle_array and y_margin for 2D data")

    if len(data.shape) == 3:
        concatenated_data_x = np.concatenate((data, data, data), axis=1)
        concatenated_data_xy = np.concatenate(
            (concatenated_data_x, concatenated_data_x, concatenated_data_x), axis=1
        )

        nx, ny, nz = data.shape

        extracted_data = []

        for x_index_middle, y_index_middle in zip(x_index_middle_list, y_index_middle_list):
            extracted_data.append(
                concatenated_data_xy[
                    nx + x_index_middle - x_margin : nx + x_index_middle + x_margin,
                    ny + y_index_middle - y_margin : ny + y_index_middle + y_margin,
                ]
            )

    elif len(data.shape) == 2:
        concatenated_data_x = np.concatenate((data, data, data), axis=1)
        nx, nz = data.shape

        extracted_data = []
        for x_index_middle in x_index_middle_list:
            extracted_data.append(
                concatenated_data_x[
                    nx + x_index_middle - x_margin : nx + x_index_middle + x_margin
                ]
            )

    return np.array(extracted_data)


def instant_extraction_over_extreme_events(
    data: np.array, variable_to_look_for_extreme: np.array, x_margin: int, y_margin: int = None
) -> np.array:

    if len(data.shape) == 3 and y_margin == None:
        raise ValueError("3D data requires y_margin")

    if len(data.shape) == 2 and y_margin != None:
        raise ValueError("no y_margin for 2D data")

    if len(data.shape) == 2:
        x_index_middle = np.where(
            variable_to_look_for_extreme == np.max(variable_to_look_for_extreme)
        )[0]

    # exctraction_over_extreme_events=
