"""Base functions for compute composite plots"""

import numpy as np

# from scipy import signal


def extract_circular_block(
    data: np.array,
    x_index_middle_array: np.array,
    x_margin: int,
    y_index_middle_array: np.array = None,
    y_margin: int = None,
):
    """Summary

    Args:
        data (np.array): Description
        x_index_middle_array (np.array): Description
        x_margin (int): Description
        y_index_middle_array (np.array, optional): Description
        y_margin (int, optional): Description
    """
    if len(data.shape) == 3 and y_index_middle_array is None and y_margin is None:
        raise ValueError("3D data requires y_index_middle_array and y_margin")

    if len(data.shape) == 2 and y_index_middle_array is not None and y_margin is not None:
        raise ValueError("no y_index_middle_array and y_margin for 2D data")

    if len(data.shape) == 3:
        concatenated_data_x = np.concatenate((data, data, data), axis=2)
        concatenated_data_xy = np.concatenate(
            (concatenated_data_x, concatenated_data_x, concatenated_data_x), axis=1
        )

        nz, ny, nx = data.shape

        extracted_data = []

        for x_index_middle, y_index_middle in zip(x_index_middle_array, y_index_middle_array):
            extracted_data.append(
                concatenated_data_xy[
                    :,
                    ny + y_index_middle - y_margin : ny + y_index_middle + y_margin,
                    nx + x_index_middle - x_margin : nx + x_index_middle + x_margin,
                ]
            )

    elif len(data.shape) == 2:
        concatenated_data_x = np.concatenate((data, data, data), axis=0)
        nx, nz = data.shape

        extracted_data = []
        for x_index_middle in x_index_middle_array:
            extracted_data.append(
                concatenated_data_x[
                    nx + x_index_middle - x_margin : nx + x_index_middle + x_margin, :
                ]
            )

    return np.array(extracted_data)


def extreme_index(
    data: np.array, variable_to_look_for_extreme: np.array, extreme_events_choice: str
) -> np.array:

    if extreme_events_choice not in ["max", "1-percentile", "10-percentile"]:
        raise ValueError("data name must be in [max,1-percentile,10-percentile]")

    if len(data.shape) == 2:
        if extreme_events_choice == "max":
            x_index_middle_array = np.where(
                variable_to_look_for_extreme == np.max(variable_to_look_for_extreme)
            )[0]

        if extreme_events_choice == "1-percentile":
            x_index_middle_array = np.unique(
                np.where(
                    variable_to_look_for_extreme
                    > np.percentile(variable_to_look_for_extreme, 99)
                )[0]
            )

        if extreme_events_choice == "10-percentile":
            x_index_middle_array = np.unique(
                np.where(
                    variable_to_look_for_extreme
                    > np.percentile(variable_to_look_for_extreme, 90)
                )[0]
            )
        return np.array(x_index_middle_array)

    if len(data.shape) == 3:
        if extreme_events_choice == "max":
            x_index_middle_array, y_index_middle_array = np.where(
                variable_to_look_for_extreme == np.max(variable_to_look_for_extreme)
            )

        if extreme_events_choice == "1-percentile":
            x_index_middle_array, y_index_middle_array = np.where(
                variable_to_look_for_extreme > np.percentile(variable_to_look_for_extreme, 99)
            )

        if extreme_events_choice == "10-percentile":
            x_index_middle_array, y_index_middle_array = np.where(
                variable_to_look_for_extreme > np.percentile(variable_to_look_for_extreme, 90)
            )
        return np.array(x_index_middle_array), np.array(y_index_middle_array)


def instant_mean_extraction_data_over_extreme(
    data: np.array,
    variable_to_look_for_extreme: np.array,
    extreme_events_choice: str,
    x_margin: int,
    y_margin: int = None,
):

    if len(data.shape) == 3 and y_margin is None:
        raise ValueError("3D data requires y_margin")

    if len(data.shape) == 2 and y_margin is not None:
        raise ValueError("no y_margin for 2D data")

    if len(data.shape) == 3:
        x_index_middle_array, y_index_middle_array = extreme_index(
            data=data,
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
        else:
            instant_data_over_extreme = np.mean(instant_data_over_extreme, axis=0)
            return instant_data_over_extreme

    if len(data.shape) == 2:
        x_index_middle_array = extreme_index(
            data=data,
            variable_to_look_for_extreme=variable_to_look_for_extreme,
            extreme_events_choice=extreme_events_choice,
        )

        instant_data_over_extreme = extract_circular_block(
            data=data, x_index_middle_array=x_index_middle_array, x_margin=x_margin
        )

        if len(instant_data_over_extreme) == 1:
            return instant_data_over_extreme
        else:
            instant_data_over_extreme = np.mean(instant_data_over_extreme, axis=0)
            instant_data_over_extreme = np.mean(instant_data_over_extreme, axis=1)
            return instant_data_over_extreme
