"""Squall Line class, allows to analyze organisation of convection into squall lines"""

import pickle

import numpy as np
import pySAM
from pySAM.squall_line.angle_detection import multi_angle_instant_convolution
from pySAM.utils import make_parallel


class SquallLine:

    """SquallLine creates object that gathers all information of squall lines.
    You measure the organisation of convection, get the orientation of the line along time,
    get the global angle of the squall line

    Attributes:
        angle_degrees (float): Global orientation angle of the squall line in degrees.
                                0 degree means the squall line is perpendicular to the shear
        angle_radian (float): Global orientation angle of the squall line in radian.
                                0 degree means the squall line is perpendicular to the shear
        distribution_angles (np.array): Distribution of the orientation along time

        PRECi (xr.array): Precipitation [mm] (t,z,y,x)
        PW (xr.array): Precipitable Water [mm] (t,z,y,x)
        U (xr.array): X component of the velocity field [m/s]
        W (xr.array): Z component of the velocity field [m/s]

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
        self.angle_degrees = None
        self.angle_radian = None

    def save(self, path_to_save: str):
        """Saves instance of the class except from starting data

        Args:
            path_to_save (str): path to the saving file
        """
        black_list = [
            key for (key, value) in self.__dict__.items() if (key.isupper() or key == "PRECi")
        ]

        dictionary = {
            key: value for (key, value) in self.__dict__.items() if key not in black_list
        }

        file = open(path_to_save, "wb")
        pickle.dump(dictionary, file, 2)
        file.close()

    def load(self, path_to_load: str):
        # pylint: disable=R0801
        """Loads calculated variables

        Args:
            path_to_load (str): path to the saved file
        """
        file = open(path_to_load, "rb")
        tmp_dict = pickle.load(file)
        file.close()
        self.__dict__.update(tmp_dict)

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

        self.angle_radian = (
            angles_range[
                np.where(self.distribution_angles == np.max(self.distribution_angles))[0]
            ]
            - np.pi / 2
        )

    def set_maximum_variance_step(self, data_name: str):
        """Returns the timestep where the squall lines is the most
        organized according to a reference data ie where the variance of PW (for exemple) is maximum

        Args:
            data_name (str): 2d fields that well represents convective organisation
        """
        variance_evolution = []
        for data in getattr(self, data_name):
            variance_evolution.append([np.var(data)])

        maximum_variance_step = np.where(variance_evolution == np.max(variance_evolution))[0][0]
        setattr(self, data_name + "_maximum_variance_step", maximum_variance_step)
