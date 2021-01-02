"""Base functions for compute composite plots"""

import numpy as np
import pytest
from pySAM.cold_pool.composite import extract_circular_block, extreme_index
from scipy import signal


def test_extract_circular_block():

    dummy_data_3d = np.ones((10, 15, 5))
    dummy_data_3d[0, 10, 4] = 10
    dummy_data_3d[0, 13, 4] = 10

    dummy_data_2d = np.ones((6, 7))
    dummy_data_2d[0, 4] = 10

    with pytest.raises(ValueError) as e:
        extract_circular_block(dummy_data_3d, np.array([3]), 1)
    assert "3D data requires y_index_middle_array and y_margin" in str(e)

    with pytest.raises(ValueError) as e:
        extract_circular_block(dummy_data_2d, np.array([3]), 1, np.array([2]), 1)
    assert "no y_index_middle_array and y_margin for 2D data" in str(e)

    # extract_circular_block(dummy_data_3d, 3, 1, 2, 5)
    # extract_circular_block(dummy_data_2d, 3, 1)

    test_2D = extract_circular_block(dummy_data_2d, np.array([2, 3, 5], dtype=int), 1)
    # print(test_2D.shape)
    assert test_2D.shape == (3, 2 * 1, 7)

    test_3D = extract_circular_block(
        dummy_data_3d, np.array([2, 3, 5], dtype=int), 1, np.array([2, 3, 5], dtype=int), 2
    )
    assert test_3D.shape == (3, 10, 2 * 2, 2 * 1)


def test_extreme_index():

    dummy_data_3d = np.random.random((30, 50, 100))
    dummy_data_2d = np.random.random((100, 50))

    dummy_variable_to_look_for_extreme_2d = np.random.random((100, 50))

    with pytest.raises(ValueError) as e:
        extreme_index(dummy_data_2d, dummy_variable_to_look_for_extreme_2d, "yoo")
    assert "data name must be in [max,1-percentile,10-percentile]" in str(e)

    test_max_2d = extreme_index(dummy_data_2d, dummy_variable_to_look_for_extreme_2d, "max")
    print(test_max_2d)

    test_1_perc_2d = extreme_index(
        dummy_data_2d, dummy_variable_to_look_for_extreme_2d, "1-percentile"
    )
    print(test_1_perc_2d)

    test_10_perc_2d = extreme_index(
        dummy_data_2d, dummy_variable_to_look_for_extreme_2d, "10-percentile"
    )
    print(test_10_perc_2d)

    test_max_3d = extreme_index(dummy_data_3d, dummy_variable_to_look_for_extreme_2d, "max")
    print(test_max_3d)

    test_1_perc_3d = extreme_index(
        dummy_data_3d, dummy_variable_to_look_for_extreme_2d, "1-percentile"
    )
    print(test_1_perc_3d)

    test_10_perc_3d = extreme_index(
        dummy_data_3d, dummy_variable_to_look_for_extreme_2d, "10-percentile"
    )
    print(test_1_perc_3d)


test_extreme_index()
