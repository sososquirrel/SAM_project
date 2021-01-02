"""Base functions for compute composite plots"""

import numpy as np
import pytest
from pySAM.cold_pool.composite import extract_circular_block
from scipy import signal

dummy_data_3d = np.random.random((10, 15, 5))
dummy_data_2d = np.random.random((6, 7))


def test_extract_circular_block():

    with pytest.raises(ValueError) as e:
        extract_circular_block(dummy_data_3d, 3, 1)
    assert "3D data requires y_index_middle and y_margin" in str(e)

    with pytest.raises(ValueError) as e:
        extract_circular_block(dummy_data_2d, 3, 1, 2, 1)
    assert "no y_index_middle and y_margin for 2D data" in str(e)

    # extract_circular_block(dummy_data_3d, 3, 1, 2, 5)
    # extract_circular_block(dummy_data_2d, 3, 1)

    extract_circular_block(dummy_data_2d, np.array([2, 3, 5], dtype=int), 1)
