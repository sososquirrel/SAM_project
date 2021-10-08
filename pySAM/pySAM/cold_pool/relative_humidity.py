"""functions to get relative humidity"""
import numpy as np
from pySAM.utils import expand_array_to_tzyx_array


def get_omega(temperature: np.array) -> np.array:
    T00n = 253.16
    T0n = 273.16
    normalized_T = (temperature - T00n) / (T0n - T00n)
    normalized_T[normalized_T > 1] = 1
    normalized_T[normalized_T < 0] = 0
    om = normalized_T

    return om


def get_qsatw(pressure_mbar: np.array, temperature: np.array) -> np.array:

    coeff_a = np.array(
        [
            6.11239921,
            0.443987641,
            0.142986287e-1,
            0.264847430e-3,
            0.302950461e-5,
            0.206739458e-7,
            0.640689451e-10,
            -0.952447341e-13,
            -0.976195544e-15,
        ]
    )
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = coeff_a

    T_celsius = temperature - 273.16
    T_celsius[T_celsius < -80] = -80
    dt = T_celsius

    esatw = a0 + dt * (
        a1 + dt * (a2 + dt * (a3 + dt * (a4 + dt * (a5 + dt * (a6 + dt * (a7 + a8 * dt))))))
    )  # in mbar

    pmbar_tzyx = expand_array_to_tzyx_array(
        time_dependence=False, input_array=pressure_mbar, final_shape=esatw.shape
    )

    denom_max_w = np.maximum(pmbar_tzyx, esatw - pmbar_tzyx)

    qsatw = 0.622 * esatw / denom_max_w  # kg/kg

    return qsatw


def get_qsati(pressure_mbar: np.array, temperature: np.array) -> np.array:

    coeff_b = np.array(
        [
            6.11147274,
            0.503160820,
            0.188439774e-1,
            0.420895665e-3,
            0.615021634e-5,
            0.602588177e-7,
            0.385852041e-9,
            0.146898966e-11,
            0.252751365e-14,
        ]
    )
    b0, b1, b2, b3, b4, b5, b6, b7, b8 = coeff_b

    T_celsius = temperature - 273.16
    T_celsius[T_celsius < -100] = -100
    dt = T_celsius

    # dt    = (T>185).*(T-273.16) + (T<=185).*(max(-100,T-273.16)); #sat.f90 use some additional interpolation below 184K

    esati = (temperature > 185) * 1 * (
        b0
        + dt
        * (b1 + dt * (b2 + dt * (b3 + dt * (b4 + dt * (b5 + dt * (b6 + dt * (b7 + b8 * dt)))))))
    ) + (temperature <= 185) * 1 * (
        0.00763685 + dt * (0.000151069 + dt * 7.48215e-07)
    )  # mbar

    pmbar_tzyx = expand_array_to_tzyx_array(
        time_dependence=False, input_array=pressure_mbar, final_shape=esati.shape
    )

    denom_max_i = np.maximum(esati, pmbar_tzyx - esati)
    qsati = 0.622 * esati / denom_max_i  # kg/kg

    return qsati


def get_qsatt(pressure_mbar: np.array, temperature: np.array) -> np.array:
    qsatw = get_qsatw(pressure_mbar, temperature)
    qsati = get_qsati(pressure_mbar, temperature)
    omega = get_omega(temperature)

    qsatt = omega * qsatw + (1 - omega) * qsati

    return qsatt
