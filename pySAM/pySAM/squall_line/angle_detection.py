"""Base functions to measure the orientation angle of a squal line"""

import collections

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def rotate_sigma(sigma: np.array, theta: float) -> np.array:
    """Returns a 2d gaussian with the principle axis oriented to theta compared to the horizontal

    Args:
        sigma (np.array): covariance in standard base
        theta (float): angle

    Returns:
        np.array: rotated covariance matrix at angle theta
    """

    if sigma.shape != (2, 2):
        raise ValueError("covariance matrix size must be (2,2)")

    _, passage = np.linalg.eig(sigma)

    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    matrix_product = np.dot(rotation_matrix, passage)
    rotated_sigma = matrix_product.T @ sigma @ matrix_product

    return rotated_sigma


def multivariate_gaussian(pos: np.array, mu: np.array, sigma: np.array) -> np.array:
    """From positions given by pos matrix, multivariate_gaussian return the associated values

    Args:
        pos (np.array): 3D array of size n*n*2 with coordinate positions for each point in grid
        mu (np.array): 1D vector of size 2 with x mean and y mean
        sigma (np.array): 2*2 covariance matrix (between x and y)

    Returns:
        np.array: 2D gaussian array of mean equal to mu and variance to sigma over the pos grid
    """

    if len(pos.shape) != 3:
        raise ValueError("pos must be a 3D array")
    if pos.shape[-1] != 2:
        raise ValueError(
            "last number of shape must be equal to 2 as it deals with 2D coordinates"
        )
    if sigma.shape[0] != mu.shape[0]:
        raise ValueError("sigma and mu have not the same base size")

    n_dimension = mu.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)

    normalization_ratio = np.sqrt((2 * np.pi) ** n_dimension * sigma_det)

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum("...k,kl,...l->...", pos - mu, sigma_inv, pos - mu)

    return np.exp(fac / (-2)) / normalization_ratio


def normalized_autocorrelation(image: np.array) -> np.array:
    """Retunrs autocorrelation from image

    Args:
        image (np.array): 2D image for which correlation will be computed

    Returns:
        np.array: normalized autocorrelation
    """

    if len(image.shape) != 2:
        raise ValueError("image must be a 2D array")

    image = image - np.mean(image)
    autocorrelation = signal.correlate2d(image, image, mode="same")

    if np.max(autocorrelation) == 0:
        raise ValueError("max value of autocorrelation is equal to 0")

    autocorrelation = autocorrelation / np.max(autocorrelation)

    return autocorrelation


def convolution_with_gaussian(
    autocorrelation_image: np.array, theta: float, mu: np.array, sigma: np.array
) -> float:
    """Scalar product between autocorrelation image and a gaussian with certain angle theta

    Args:
        autocorrelation_image (np.array): Description
        theta (float): angle of gaussian
        mu (np.array): mean of gaussian
        sigma (np.array): variance of gaussian

    Deleted Parameters:
        image (np.array): image

    No Longer Returned:
        float: scalar product between autocorrelation image and a gaussian with certain angle theta
    """

    length = autocorrelation_image.shape[0]
    width = autocorrelation_image.shape[1]
    x = np.linspace(-3, 3, length)
    y = np.linspace(-3, 3, width)
    x, y = np.meshgrid(x, y)

    # positions will store x,y coordinates of desired points
    positions = np.empty(x.shape + (2,))
    positions[:, :, 0] = x
    positions[:, :, 1] = y

    gaussian_filter = multivariate_gaussian(
        pos=positions, mu=mu, sigma=rotate_sigma(sigma, theta)
    )

    convolution_value = np.mean(autocorrelation_image * gaussian_filter)

    return convolution_value


def multi_angle_instant_convolution(
    image: np.array, theta_range: np.array, mu: np.array, sigma: np.array
) -> np.array:
    """Computes the values of convolution for multiple angles

    Args:
        image (np.array): image
        theta_range (np.array): array of theta for the gaussian filter
        mu (np.array): means
        sigma (np.array): variance

    Returns:
        np.array: array of convolutions for multiple angles
    """
    autocorrelation_image = normalized_autocorrelation(image)

    convolution_values = []

    for theta in theta_range:
        convolution_values.append(
            convolution_with_gaussian(
                autocorrelation_image=autocorrelation_image, theta=theta, mu=mu, sigma=sigma
            )
        )

    return np.array(convolution_values)


def distribution_sampling(data_value: list, most_common_limit: int = 5):

    counter_dict = collections.Counter(data_value)
    most_common_list = counter_dict.most_common(most_common_limit)

    mean_result = np.mean(np.array([key for key, __ in most_common_list]))
    deviation_result = np.sqrt(np.var(np.array([key for key, __ in most_common_list])))

    return (mean_result, deviation_result)
