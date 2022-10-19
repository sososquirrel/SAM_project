"""Base functions for compute composite plots"""

import numpy as np

# from scipy import signal


def extract_circular_block(
    data: np.array,
    x_index_middle_array: np.array,
    x_margin: int,
    y_index_middle_array: np.array,
    y_margin: int,
):
    """Summary

    Args:
        data (np.array): Description
        x_index_middle_array (np.array): Description
        x_margin (int): Description
        y_index_middle_array (np.array, optional): Description
        y_margin (int, optional): Description
    """
    # if len(data.shape) == 3 and y_index_middle_array is None and y_margin is None:
    #     raise ValueError("3D data requires y_index_middle_array and y_margin")

    # if None in [y_index_middle_array, y_margin]:
    #     assert y_index_middle_array is None
    #     assert y_margin is None

    # if len(data.shape) == 2 and y_index_middle_array is not None and y_margin is not None:
    #     raise ValueError("no y_index_middle_array and y_margin for 2D data")

    # if y_index_middle_array is not None:
    if y_index_middle_array.shape[0] != x_index_middle_array.shape[0]:
        raise ValueError("x middle array and y middle array must be same size")

    if len(data.shape) == 3:
        concatenated_data_x = np.concatenate((data, data, data), axis=2)
        concatenated_data_xy = np.concatenate(
            (concatenated_data_x, concatenated_data_x, concatenated_data_x), axis=1
        )

        _, ny, nx = data.shape

        extracted_data = []

        for x_index_middle, y_index_middle in zip(x_index_middle_array, y_index_middle_array):
            extracted_data.append(
                concatenated_data_xy[
                    :,
                    ny + y_index_middle - y_margin : ny + y_index_middle + y_margin + 1,
                    nx + x_index_middle - x_margin : nx + x_index_middle + x_margin + 1,
                ]
            )

    elif len(data.shape) == 2:
        concatenated_data_x = np.concatenate((data, data, data), axis=1)
        concatenated_data_xy = np.concatenate(
            (concatenated_data_x, concatenated_data_x, concatenated_data_x), axis=0
        )

        nx, ny = data.shape

        extracted_data = []
        for x_index_middle, y_index_middle in zip(x_index_middle_array, y_index_middle_array):
            extracted_data.append(
                concatenated_data_xy[
                    ny + y_index_middle - y_margin : ny + y_index_middle + y_margin + 1,
                    nx + x_index_middle - x_margin : nx + x_index_middle + x_margin + 1,
                ]
            )

    return np.array(extracted_data)


def extreme_index(
    variable_to_look_for_extreme: np.array, extreme_events_choice: str
) -> np.array:
    """Summary

    Args:
        nb_dim_data (int): Description
        variable_to_look_for_extreme (np.array): Description
        extreme_events_choice (str): Description
    """
    if extreme_events_choice not in [
        "max",
        "1-percentile",
        "0.1-percentile",
        "0.01-percentile",
        "10-percentile",
        "min",
        "99-percentile",
        "90-percentile",
    ]:
        raise ValueError(
            "extreme_events_choice must be in [max,0.1-percentile,0.01-percentile, 1-percentile,10-percentile]"
        )

    if extreme_events_choice == "max":
        index_middle_array = np.where(
            variable_to_look_for_extreme == np.max(variable_to_look_for_extreme)
        )

    elif extreme_events_choice == "min":
        index_middle_array = np.where(
            variable_to_look_for_extreme == np.min(variable_to_look_for_extreme)
        )

    else:
        if extreme_events_choice == "1-percentile":
            percentile_value = 1

        if extreme_events_choice == "0.1-percentile":
            percentile_value = 0.1

        if extreme_events_choice == "0.01-percentile":
            percentile_value = 0.01

        if extreme_events_choice == "10-percentile":
            percentile_value = 10

    index_middle_array = np.where(
        variable_to_look_for_extreme > np.quantile(variable_to_look_for_extreme, 0.999)
    )

    # if nb_dim_data == 2:
    #   return np.unique(index_middle_array[1])

    return index_middle_array


def instant_mean_extraction_data_over_extreme(
    data: np.array,
    variable_to_look_for_extreme: np.array,
    extreme_events_choice: str,
    x_margin: int,
    y_margin: int = None,
    return_3D: bool = False,
):
    # pylint: disable=R1705
    """Summary

    Args:
        data (np.array): Description
        variable_to_look_for_extreme (np.array): Description
        extreme_events_choice (str): Description
        x_margin (int): Description
        y_margin (int, optional): Description
    """

    if len(data.shape) not in [2, 3]:
        raise ValueError("data must be either 2D or 3D")

    # if len(data.shape) == 3 and y_margin is None:
    #     raise ValueError("3D data requires y_margin")

    # if len(data.shape) == 2 and y_margin is not None:
    #     raise ValueError("no y_margin required for 2D data")

    if len(data.shape) == 3:
        y_index_middle_array, x_index_middle_array = extreme_index(
            variable_to_look_for_extreme=variable_to_look_for_extreme,
            extreme_events_choice=extreme_events_choice,
        )

        instant_data_over_extreme = extract_circular_block(
            data=data,
            x_index_middle_array=x_index_middle_array,
            x_margin=x_margin,
            y_index_middle_array=y_index_middle_array,
            y_margin=y_margin,
        )

        if len(instant_data_over_extreme) == 1:
            if not return_3D:
                return np.mean(instant_data_over_extreme[0], axis=1)
            else:
                return instant_data_over_extreme[0]

        instant_data_over_extreme = np.mean(instant_data_over_extreme, axis=0)

        return instant_data_over_extreme

    else:  # len(data.shape) == 2

        y_index_middle_array, x_index_middle_array = extreme_index(
            variable_to_look_for_extreme=variable_to_look_for_extreme,
            extreme_events_choice=extreme_events_choice,
        )

        instant_data_over_extreme = extract_circular_block(
            data=data,
            x_index_middle_array=x_index_middle_array,
            x_margin=x_margin,
            y_index_middle_array=y_index_middle_array,
            y_margin=y_margin,
        )

        if len(instant_data_over_extreme) == 1:
            return np.mean(instant_data_over_extreme[0], axis=1)

        instant_data_over_extreme = np.mean(instant_data_over_extreme, axis=(0, 1))
        # instant_data_over_extreme = np.mean(instant_data_over_extreme, axis=1)
        return instant_data_over_extreme
