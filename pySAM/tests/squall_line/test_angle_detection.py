import numpy as np
import pySAM
import pytest


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

    multivariate_gaussian(pos, mu, sigma)
