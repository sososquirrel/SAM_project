"""Base functions for compute composite plots"""

import numpy as np
import pytest
from pySAM.cold_pool.composite import extract_circular_block
from scipy import signal

dummy_data_3d = np.ones((10, 15, 5))
dummy_data_3d[0, 10, 4] = 1
dummy_data_3d[0, 15, 4] = 1

dummy_data_2d = np.ones((6, 7))
dummy_data_2d[0, 4] = 1


def test_extract_circular_block():

    with pytest.raises(ValueError) as e:
        extract_circular_block(dummy_data_3d, np.array([3]), 1)
    assert "3D data requires y_index_middle_array and y_margin" in str(e)

    with pytest.raises(ValueError) as e:
        extract_circular_block(dummy_data_2d, np.array([3]), 1, np.array([2]), 1)
    assert "no y_index_middle_array and y_margin for 2D data" in str(e)

    # extract_circular_block(dummy_data_3d, 3, 1, 2, 5)
    # extract_circular_block(dummy_data_2d, 3, 1)

    test_2D = extract_circular_block(dummy_data_2d, np.array([2, 3, 5], dtype=int), 1)
    print(test_2D.shape)

    test_3D = extract_circular_block(
        dummy_data_2d, np.array([2, 3, 5], dtype=int), 1, np.array([2, 3, 5], dtype=int), 2
    )
    print(test_3D.shape)
