"""Definition of cold pool class"""

import numpy as np
import pySAM


def test_ColdPool():
    class ColdPool:

        """Summary

        Attributes:
            distribution_angles (TYPE): Description
            PRECi (TYPE): Description
            PW (TYPE): Description
            U (TYPE): Description
            V (TYPE): Description
            W (TYPE): Description
        """

        def __init__(
            self,
            absolute_temperature: np.array,
            instantaneous_precipitation: np.array,
            x_positions: np.array,
            y_positions: np.array,
            z_positions: np.array,
            x_velocity: np.array,
            z_velocity: np.array,
            cloud_base: np.array,
        ):

            self.TABS = absolute_temperature
            self.PRECi = instantaneous_precipitation
            self.X = x_positions
            self.Y = y_positions
            self.Z = z_positions
            self.U = x_velocity
            self.W = z_x_velocity
            self.QN = cloud_base
