"""Definition of squall line class"""

import numpy as np
import pySAM
from pySAM.squall_line.angle_detection import multi_angle_instant_convolution
from pySAM.utils import make_parallel


class SquallLine:

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
        precipitable_water,
        instantaneous_precipitation,
        x_velocity,
        z_velocity,
    ):
        self.PW = precipitable_water
        self.PRECi = instantaneous_precipitation
        self.U = x_velocity
        self.W = z_velocity

        self.distribution_angles = None

    def set_distribution_angles(
        self,
        data_name: str,
        angles_range: np.array,
        mu: np.array,
        sigma: np.array,
        parallelize: bool = True,
    ) -> np.array:
        """set a new attribute to squall line object : self.distribution_angles, that gives convolution values over angles

        Args:
            data_name (str): name of data to use, must be in ['PW', 'PRECi', 'U', 'V', 'W' ]
            angles_range (np.array): angles for which convolution will be computed
            mu (np.array): mean vector for gaussian filter
            sigma (np.array): covariance matrix for gaussian filter
            parallelize (bool, optional): Use multiprocessing to dispatch tasks over available CPUs
        """

        if data_name not in ["PW", "PRECi", "U", "W"]:
            raise ValueError("data name must be in [PW, PRECi, U, W]")
        if len(angles_range.shape) != 1:
            raise ValueError("angles_range must be 1D")

        if parallelize:
            parallel_multi_angle = make_parallel(
                function=multi_angle_instant_convolution, nprocesses=pySAM.N_CPU
            )

            angles_distribution = parallel_multi_angle(
                iterable_values_1=getattr(self, data_name),
                theta_range=angles_range,
                mu=mu,
                sigma=sigma,
            )

        else:  # NO PARALLELIZATION
            angles_distribution = []
            data = getattr(self, data_name)
            for image in data:
                angles_distribution.append(
                    multi_angle_instant_convolution(
                        image, theta_range=angles_range, mu=mu, sigma=sigma
                    )
                )

        self.distribution_angles = np.mean(np.array(angles_distribution), axis=0)

        self.angle_degrees = (
            (
                angles_range[
                    np.where(self.distribution_angles == np.max(self.distribution_angles))[0]
                ]
                - np.pi / 2
            )
            * 180
            / np.pi
        )
