import numpy as np
import pySAM
import pytest
from pySAM.squall_line.squall_line import SquallLine

dummy_data = np.random.random((3, 10, 10))


def test_set_distribution_angles():
    squall_line = SquallLine(dummy_data, dummy_data, dummy_data, dummy_data)

    with pytest.raises(ValueError) as e:
        squall_line.set_distribution_angles(
            data_name="hey soso",
            angles_range=np.linspace(0, 2, 10),
            mu=np.array([0, 0]),
            sigma=np.array([[1.0, -0.985], [-0.985, 1.0]]),
        )
    assert "data name must be in [PW, PRECi, U, W]" in str(e)

    with pytest.raises(ValueError) as e:
        squall_line.set_distribution_angles(
            data_name="PW",
            angles_range=np.random.random((2, 4)),
            mu=np.array([0, 0]),
            sigma=np.array([[1.0, -0.985], [-0.985, 1.0]]),
        )

    squall_line.PW

    angles_with_paralellization = squall_line.set_distribution_angles(
        data_name="PW",
        angles_range=np.linspace(0, 2, 10),
        mu=np.array([0, 0]),
        sigma=np.array([[1.0, -0.985], [-0.985, 1.0]]),
    )

    angles_without_paralellization = squall_line.set_distribution_angles(
        data_name="PW",
        angles_range=np.linspace(0, 2, 10),
        mu=np.array([0, 0]),
        sigma=np.array([[1.0, -0.985], [-0.985, 1.0]]),
        parallelize=False,
    )

    assert angles_with_paralellization == angles_without_paralellization
