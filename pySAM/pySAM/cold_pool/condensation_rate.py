"""functions to get the condensation rate int(rho*w dq)"""
import numpy as np
from pySAM.utils import expand_array_to_tzyx_array


def get_condensation_rate(
    vertical_velocity: np.array, density: np.array, z_array: np.array, humidity: np.array
):

    if vertical_velocity.shape[0] != density.shape[0]:
        nt = vertical_velocity.shape[0]
        nt_long = density.shape[0]
        density = density[nt_long - nt :, :]

    if z_array.shape[0] != density.shape[1]:
        nz = z_array.shape[0]
        density = density[:, :nz]

    w = vertical_velocity

    rho = density
    rho_3D = np.tile(rho, (128, 128, 1, 1))
    rho_3D = rho_3D.T
    rho_3D = np.swapaxes(rho_3D, 0, 1)

    product_rho_w = w * rho_3D

    dz = np.gradient(z_array.T)
    dz_3D = np.tile(dz, (121, 128, 128, 1))
    dz_3D = np.swapaxes(dz_3D, 1, 3)

    dq = dz_3D * np.gradient(humidity, axis=0)
    CR = np.sum(product_rho_w * (-dq), axis=1)

    return CR
