"""Summary
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def expand_three_time_larger(data: np.array):
    """This function expands an array of size (m1,m2...,mN,nx,ny) to the size (m1,m2...,mN,3nx,3ny)
    which avoid boundary problem in interpolation.

    Args:
        data (np.array): an array with at least 2 axis

    Returns:
        TYPE: the array expand to three times in both last axis
    """
    concatenated_data_x = np.concatenate((data, data, data), axis=-1)
    concatenated_data_xy = np.concatenate(
        (concatenated_data_x, concatenated_data_x, concatenated_data_x), axis=-2
    )

    return concatenated_data_xy


def clip_to_domain(
    x0: float, y0: float, x_left: float, x_right: float, y_bottom: float, y_top: float
):
    """This function takes into account the doubly periodicity of the domain and for given coordonate couple returns the value in the domain

    Args:
        x0 (float): Description
        y0 (float): Description
        x_left (float): Description
        x_right (float): Description
        y_bottom (float): Description
        y_top (float): Description
    """
    done = False
    while done == False:
        if x0 <= x_right and x0 >= x_left and y0 <= y_top and y0 >= y_bottom:
            x0, y0 = x0, y0
            done = True
        if x0 > x_right:
            x0, y0 = x_left + (x0 - x_right), y0
        if x0 < x_left:
            x0, y0 = x_right - (x_left - x0), y0
        if y0 > y_top:
            x0, y0 = x0, y_bottom + (y0 - y_top)
        if y0 < y_bottom:
            x0, y0 = x0, y_top - (y_bottom - y0)

    return x0, y0


def indxs(p: np.array, x: np.array, y: np.array):
    """Summary

    Args:
        p (np.array): Description
        x (np.array): Description
        y (np.array): Description

    Returns:
        TYPE: Description
    """
    i = np.int(x.shape[1] * (p[0] - x[0, 0]) / (x[0, -1] - x[0, 0]))
    j = np.int(y.shape[0] * (p[1] - y[0, 0]) / (y[-1, 0] - y[0, 0]))
    return i, j


def vitesse_field_formating(vector_field_u: np.array, vector_field_v: np.array, n_iter: int):
    """Summary

    Args:
        vector_field_u (np.array): Description
        vector_field_v (np.array): Description
        n_iter (int): Description

    Returns:
        TYPE: Description
    """
    V = np.zeros_like(np.array([vector_field_u, vector_field_v]))
    V = np.swapaxes(V, 0, 1)
    V[:, 1, :, :], V[:, 0, :, :] = vector_field_u, vector_field_v

    V = expand_three_time_larger(data=V)

    return V


# a simple quadratic interpolation of the mesh-grid vector filed
# to a quadratically interpolated vector field at a point p inside
# mesh-grid the square in which p is located
def VF(p: np.array, x: np.array, y: np.array, V: np.array):
    """Summary

    Args:
        p (np.array): Description
        x (np.array): Description
        y (np.array): Description
        V (np.array): Description

    Returns:
        TYPE: Description
    """
    i, j = indxs(p, x, y)
    if 0 < i and i < x.shape[1] - 1 and 0 < j and j < y.shape[0] - 1:
        a = (p[0] - x[0, i]) / (x[0, i + 1] - x[0, i])
        b = (p[1] - y[j, 0]) / (y[j + 1, 0] - y[j, 0])
        W = (1 - a) * (1 - b) * V[:, j, i] + (1 - a) * b * V[:, j, i + 1]
        W = W + a * (1 - b) * V[:, j + 1, i] + a * b * V[:, j + 1, i + 1]
        return W  # / np.linalg.norm(W) # you can also normalize the vector field to get only the trajecotry, without accounting for parametrization
    else:
        return np.array([0.0, 0.0])


# integrating the v#ector field one time step
# starting from a given point p
# uses Runge-Kutta 4 integrations, which
# allows you to sample the vector fields at four different near-by points and
# 'average' the result
def VF_flow(
    p: np.array,
    x: np.array,
    y: np.array,
    V: np.array,
    t_step: float,
    x_left: float,
    x_right: float,
    y_top: float,
    y_bottom: float,
):
    """Summary

    Args:
        p (np.array): Description
        x (np.array): Description
        y (np.array): Description
        V (np.array): Description
        t_step (float): Description
        x_left (float): Description
        x_right (float): Description
        y_top (float): Description
        y_bottom (float): Description
    """
    x_max = int(np.max(x))
    y_max = int(np.max(y))

    x_min = int(np.max(x))
    y_max = int(np.max(y))

    k1 = VF(p, x, y, V)
    k2 = VF(p + t_step * k1 / 2, x, y, V)
    k3 = VF(p + t_step * k2 / 2, x, y, V)
    k4 = VF(p + t_step * k3, x, y, V)

    new_p = p + t_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    new_p_clipped = clip_to_domain(
        x0=new_p[0], y0=new_p[1], x_left=x_left, x_right=x_right, y_bottom=y_bottom, y_top=y_top
    )
    return new_p_clipped


def VF_trajectory(
    p: np.array,
    x: np.array,
    y: np.array,
    Vx: np.array,
    Vy: np.array,
    t_step: float,
    n_iter: int,
    x_left: float,
    x_right: float,
    y_top: float,
    y_bottom: float,
):
    """Summary

    Args:
        p (np.array): Description
        x (np.array): Description
        y (np.array): Description
        Vx (np.array): Description
        Vy (np.array): Description
        t_step (float): Description
        n_iter (int): Description
        x_left (float): Description
        x_right (float): Description
        y_top (float): Description
        y_bottom (float): Description
    """
    V = vitesse_field_formating(vector_field_u=Vx, vector_field_v=Vy, n_iter=n_iter)

    traj = np.empty((2, n_iter), dtype=float)
    traj[:, 0] = p

    for m in range(n_iter - 1):
        traj[:, m + 1] = VF_flow(
            p=traj[:, m],
            x=x,
            y=y,
            V=V[m],
            t_step=t_step,
            x_left=x_left,
            x_right=x_right,
            y_bottom=y_bottom,
            y_top=y_top,
        )
        m = m + 1
    return traj
