import matplotlib.pyplot as plt
import numpy as np
import pySAM
import pytest
from pySAM.squall_line.angle_detection import (multivariate_gaussian,
                                               rotate_sigma)


def test_multivariate_gaussian():
    im = np.ones((100, 100))
    im[0, 6] = 18
    N = im.shape[0]
    M = im.shape[1]
    X = np.linspace(-3, 3, M)
    Y = np.linspace(-3, 3, N)
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    mu = np.array([0.0, 0.0])
    sigma = np.array([[1.0, -0.985], [-0.985, 1.0]])

    sigma_rot = np.array([[1.98500000e00, 1.79289683e-16], [1.30408774e-16, 1.50000000e-02]])

    multivariate_gaussian(pos, mu, sigma)
    print(multivariate_gaussian(pos, mu, sigma).shape)
    plt.imshow(multivariate_gaussian(pos, mu, sigma))
    plt.show()
    plt.imshow(multivariate_gaussian(pos, mu, sigma_rot))
    plt.show()


# test_multivariate_gaussian()


for theta in np.linspace(0, np.pi, 10):
    sigma = rotate_sigma(sigma=pySAM.SIGMA_GAUSSIAN, theta=theta)

    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    x, y = np.meshgrid(x, y)

    # positions will store x,y coordinates of desired points
    positions = np.empty(x.shape + (2,))
    positions[:, :, 0] = x
    positions[:, :, 1] = y

    gaussian_filter = multivariate_gaussian(pos=positions, mu=pySAM.MU_GAUSSIAN, sigma=sigma)

    plt.imshow(gaussian_filter)
    plt.show()
