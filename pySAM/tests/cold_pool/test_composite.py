"""Base functions for compute composite plots"""

import numpy as np
import pytest
from pySAM.cold_pool.composite import extract_circular_block, extreme_index
from scipy import signal


def test_extract_circular_block():

    dummy_data_3d = np.ones((3, 4, 5))
    dummy_data_3d[0, :, 1] = 10
    dummy_data_3d[0, 1, :] = 10
    dummy_data_3d[0, :, -1] = 5
    dummy_data_3d[2, :, -1] = 6

    dummy_data_2d = np.ones((6, 7))
    dummy_data_2d[0, 3:] = 10
    dummy_data_2d[1, 5:] = 10

    dummy_data_3d_2 = np.ones((2, 2, 2))
    dummy_data_3d_2[0, 0, :] = 10
    dummy_data_3d_2[1, 1, 0] = 5

    # be careful dimensions order : z, y , x for 3D and y, x for 2D

    with pytest.raises(ValueError) as e:
        extract_circular_block(dummy_data_3d, np.array([3]), 1)
    assert "3D data requires y_index_middle_array and y_margin" in str(e)

    with pytest.raises(ValueError) as e:
        extract_circular_block(dummy_data_2d, np.array([3]), 1, np.array([2]), 1)
    assert "no y_index_middle_array and y_margin for 2D data" in str(e)

    test_2D = extract_circular_block(
        data=dummy_data_2d, x_index_middle_array=np.array([3]), x_margin=2
    )
    assert test_2D.shape == (1, 2 * 2 + 1, 7)
    assert (
        test_2D
        == np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            ]
        )
    ).all()

    test_3D = extract_circular_block(
        dummy_data_3d,
        x_margin=3,
        x_index_middle_array=np.array([1]),
        y_margin=2,
        y_index_middle_array=np.array([1]),
    )

    assert test_3D.shape == (1, 3, 2 * 2 + 1, 2 * 3 + 1)

    test_3D_2 = extract_circular_block(
        dummy_data_3d_2,
        x_margin=1,
        x_index_middle_array=np.array([0]),
        y_margin=2,
        y_index_middle_array=np.array([1]),
    )

    assert (
        test_3D_2
        == np.array(
            [
                [
                    [
                        [1.0, 1.0, 1.0],
                        [10.0, 10.0, 10.0],
                        [1.0, 1.0, 1.0],
                        [10.0, 10.0, 10.0],
                        [1.0, 1.0, 1.0],
                    ],
                    [
                        [1.0, 5.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 5.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 5.0, 1.0],
                    ],
                ]
            ]
        )
    ).all()


def test_extreme_index():

    dummy_data_3d = np.random.random((30, 50, 100))
    dummy_data_2d = np.random.random((100, 50))

    dummy_variable_to_look_for_extreme_2d = np.random.random((100, 50))

    with pytest.raises(ValueError) as e:
        extreme_index(2, dummy_variable_to_look_for_extreme_2d, "yoo")
    assert "data name must be in [max,1-percentile,10-percentile]" in str(e)

    test_max_2d = extreme_index(2, dummy_variable_to_look_for_extreme_2d, "max")
    print(" 2D MAX", test_max_2d)

    test_1_perc_2d = extreme_index(2, dummy_variable_to_look_for_extreme_2d, "1-percentile")
    print("2D 1 percentile", test_1_perc_2d)

    test_10_perc_2d = extreme_index(2, dummy_variable_to_look_for_extreme_2d, "10-percentile")
    print("----------")
    print(" 2D 10 percentile", test_10_perc_2d)

    test_max_3d = extreme_index(3, dummy_variable_to_look_for_extreme_2d, "max")
    print(test_max_3d)

    test_1_perc_3d = extreme_index(3, dummy_variable_to_look_for_extreme_2d, "1-percentile")
    print(test_1_perc_3d)

    test_10_perc_3d = extreme_index(3, dummy_variable_to_look_for_extreme_2d, "10-percentile")
    print(test_1_perc_3d)


test_extreme_index()
