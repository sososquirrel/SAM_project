"""functions to get the condensation rate int(rho*w dq)"""
import numpy as np
from pySAM.utils import expand_array_to_tzyx_array


def get_condensation_rate_2(
    vertical_velocity: np.array, density: np.array, z_array: np.array, humidity: np.array
):
    """Summary

    Args:
        vertical_velocity (np.array): Description
        density (np.array): Description
        z_array (np.array): Description
        humidity (np.array): Description
    """

    nt, nz, ny, nx = vertical_velocity.shape

    if vertical_velocity.shape[0] != density.shape[0]:
        nt = vertical_velocity.shape[0]
        nt_long = density.shape[0]
        density = density[nt_long - nt :, :]

    if z_array.shape[0] != density.shape[1]:
        nz = z_array.shape[0]
        density = density[:, :nz]

    w = vertical_velocity

    rho = density
    rho_3D = np.tile(rho, (ny, nx, 1, 1))
    rho_3D = rho_3D.T
    rho_3D = np.swapaxes(rho_3D, 0, 1)

    product_rho_w = w * rho_3D

    dz = np.gradient(z_array.T)
    dz_3D = np.tile(dz, (nt, ny, nx, 1))
    dz_3D = np.swapaxes(dz_3D, 1, 3)

    # dq = dz_3D * np.gradient(humidity, axis=0)
    dq = np.gradient(humidity, z_array, axis=1)
    CR = np.sum(product_rho_w * (-dq), axis=1)

    return CR


def get_condensation_rate(
    vertical_velocity: np.array, density: np.array, humidity: np.array, return_3D: bool = False
):

    nt, nz, ny, nx = vertical_velocity.shape

    if vertical_velocity.shape[0] != density.shape[0]:
        nt = vertical_velocity.shape[0]
        nt_long = density.shape[0]
        density = density[nt_long - nt :, :]

    if vertical_velocity.shape[1] != density.shape[1]:
        nz = vertical_velocity.shape[1]
        density = density[:, :nz]

    w = vertical_velocity

    rho = density
    rho_3D = np.tile(rho, (ny, nx, 1, 1))
    rho_3D = rho_3D.T
    rho_3D = np.swapaxes(rho_3D, 0, 1)

    if return_3D:
        CR = (
            w[:, 1:, :, :]
            * (-np.diff(humidity, axis=1))
            * 0.5
            * (rho_3D[:, 1:, :, :] + rho_3D[:, :-1, :, :])
        )

    else:

        CR = np.sum(
            w[:, 1:, :, :]
            * (-np.diff(humidity, axis=1))
            * 0.5
            * (rho_3D[:, 1:, :, :] + rho_3D[:, :-1, :, :]),
            axis=1,
        )

    return CR


def get_integrated_quantity(density: np.array, z_array: np.array, quantity: np.array):

    if z_array.shape[0] != density.shape[1]:
        nz = z_array.shape[0]
        density = density[:, :nz]

    rho = density
    rho_3D = np.tile(rho, (128, 128, 1, 1))
    rho_3D = rho_3D.T
    rho_3D = np.swapaxes(rho_3D, 0, 1)

    product_rho_qpsrc = quantity * rho_3D

    dz = np.gradient(z_array.T)
    dz_3D = np.tile(dz, (121, 128, 128, 1))
    dz_3D = np.swapaxes(dz_3D, 1, 3)

    integrated_quantity = np.sum(product_rho_qpsrc * dz_3D, axis=1)

    return integrated_quantity
