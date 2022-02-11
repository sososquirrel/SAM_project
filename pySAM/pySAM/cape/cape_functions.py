"""Summary
"""
import matplotlib.pyplot as plt
import numpy as np
from pySAM import config, config_sam_constants, utils
from pySAM.cape import config_cape
from scipy import interpolate, optimize


def saturation_pressure(liquid_or_ice: str, temperature_in_celsius: np.array) -> np.array:
    """Summary

    Args:
        liquid_or_ice (str): mode if you want to calculate the saturation in liquid or ice
        temperature_in_celsius (np.array): the temperature field, could be 1d, 2d, 3D, 4d!

    Returns:
        np.array: the saturation pressure, same size as temperature_in_celsius
    """
    if liquid_or_ice == "liquid":
        A, B, C, D = config_cape.AW, config_cape.BW, config_cape.CW, config_cape.DW
    else:
        A, B, C, D = config_cape.AI, config_cape.BI, config_cape.CI, config_cape.DI

    Ps = A * np.exp(
        (B - temperature_in_celsius / D)
        * (temperature_in_celsius / (C + temperature_in_celsius))
    )

    return Ps


def omega_n(
    temperature_in_kelvin: np.array,
    T_00n: float = config_sam_constants.T_00n,
    T_0n: float = config_sam_constants.T_0n,
):
    """Summary

    Args:
        temperature_in_kelvin (np.array): Description
        T_00n (float, optional): Description
        T_0n (float, optional): Description
    """
    if np.isscalar(temperature_in_kelvin):

        centered_temperature = (temperature_in_kelvin - T_00n) / (T_0n - T_00n)

        return max(centered_temperature, 1)

    else:

        centered_temperature = (temperature_in_kelvin - T_00n) / (T_0n - T_00n)
        matrix_of_ones = np.ones_like(centered_temperature)
        matrix_of_zeros = np.zeros_like(centered_temperature)

        output1 = utils.min_point_wise(matrix_1=matrix_of_ones, matrix_2=centered_temperature)

        output2 = utils.max_point_wise(matrix_1=matrix_of_zeros, matrix_2=output1)

        return output2


def saturation_mixing_ratio(temperature_in_kelvin: np.array, pressure: np.array):
    """Summary

    Args:
        temperature_in_kelvin (np.array): Description
        pressure (np.array): Description

    Returns:
        TYPE: Description
    """
    temperature_in_celsius = temperature_in_kelvin + config.ABSOLUTE_ZERO

    e_saturation_water = saturation_pressure(
        liquid_or_ice="liquid", temperature_in_celsius=temperature_in_celsius
    )
    e_saturation_ice = saturation_pressure(
        liquid_or_ice="ice", temperature_in_celsius=temperature_in_celsius
    )

    if not np.isscalar(pressure) and not pressure.shape == ():

        if len(temperature_in_kelvin.shape) == 3:
            pressure_3D = utils.expand_array_to_zyx_array(
                input_array=pressure,
                final_shape=e_saturation_ice.shape,
            )

            max_pressure_esatw = utils.max_point_wise(
                matrix_1=e_saturation_water, matrix_2=pressure_3D
            )
            max_pressure_esati = utils.max_point_wise(
                matrix_1=e_saturation_ice, matrix_2=pressure_3D
            )

            q_saturation_water = (
                config_cape.MIXING_RATIO_AIR_WATER_VAPOR
                * e_saturation_water
                / max_pressure_esatw
            )

            q_saturation_ice = (
                config_cape.MIXING_RATIO_AIR_WATER_VAPOR * e_saturation_ice / max_pressure_esati
            )

        if len(temperature_in_kelvin.shape) == 2:

            max_pressure_esatw = utils.max_point_wise(
                matrix_1=e_saturation_water, matrix_2=pressure
            )
            max_pressure_esati = utils.max_point_wise(
                matrix_1=e_saturation_ice, matrix_2=pressure
            )

            q_saturation_water = (
                config_cape.MIXING_RATIO_AIR_WATER_VAPOR
                * e_saturation_water
                / max_pressure_esatw
            )

            q_saturation_ice = (
                config_cape.MIXING_RATIO_AIR_WATER_VAPOR * e_saturation_ice / max_pressure_esati
            )

        elif len(temperature_in_kelvin.shape) == 1:

            max_pressure_esatw = utils.max_point_wise(
                matrix_1=e_saturation_water, matrix_2=pressure
            )
            max_pressure_esati = utils.max_point_wise(
                matrix_1=e_saturation_ice, matrix_2=pressure
            )

            q_saturation_water = (
                config_cape.MIXING_RATIO_AIR_WATER_VAPOR
                * e_saturation_water
                / max_pressure_esatw
            )

            q_saturation_ice = (
                config_cape.MIXING_RATIO_AIR_WATER_VAPOR * e_saturation_ice / max_pressure_esati
            )
    elif np.isscalar(pressure) or pressure.shape == ():
        if len(temperature_in_kelvin.shape) == 2:
            pressure_2D = pressure * np.ones_like(temperature_in_kelvin)

            max_pressure_esatw = utils.max_point_wise(
                matrix_1=e_saturation_water, matrix_2=pressure_2D
            )
            max_pressure_esati = utils.max_point_wise(
                matrix_1=e_saturation_ice, matrix_2=pressure_2D
            )

            q_saturation_water = (
                config_cape.MIXING_RATIO_AIR_WATER_VAPOR
                * e_saturation_water
                / max_pressure_esatw
            )

            q_saturation_ice = (
                config_cape.MIXING_RATIO_AIR_WATER_VAPOR * e_saturation_ice / max_pressure_esati
            )

        else:

            max_pressure_esatw = max(pressure, e_saturation_water)
            max_pressure_esati = max(pressure, e_saturation_ice)

            q_saturation_water = (
                config_cape.MIXING_RATIO_AIR_WATER_VAPOR
                * e_saturation_water
                / max_pressure_esatw
            )

            q_saturation_ice = (
                config_cape.MIXING_RATIO_AIR_WATER_VAPOR * e_saturation_ice / max_pressure_esati
            )

    omega = omega_n(temperature_in_kelvin=temperature_in_kelvin)

    saturation_mixing_ratio = omega * q_saturation_water + (1 - omega) * q_saturation_ice

    return saturation_mixing_ratio


def get_altitude_LCL_column(
    pressure: np.array,
    vertical_array: np.array,
    temperature_ground: float,
    humidity_ground: float,
    initial_z_guess: float = config.INITIAL_Z,
    heat_capacity_air: float = config.HEAT_CAPACITY_AIR,
    gravity: float = config.GRAVITY,
):
    """Summary

    Args:
        pressure (np.array): Description
        vertical_array (np.array): Description
        temperature_ground (float): Description
        humidity_ground (float): Description
        initial_z_guess (float, optional): Description
        heat_capacity_air (float, optional): Description
        gravity (float, optional): Description
    """

    t_linear_interpolate = interpolate.interp1d(
        vertical_array,
        temperature_ground
        + gravity / heat_capacity_air * (config.LOWEST_ATMOSPHERIC_LEVEL - vertical_array),
    )
    pressure_interpolate = interpolate.interp1d(vertical_array, pressure)

    def rsat_minus_rground(Z: float):
        """Summary

        Args:
            Z (float): Description

        Returns:
            TYPE: Description
        """

        return (
            saturation_mixing_ratio(
                temperature_in_kelvin=t_linear_interpolate(Z), pressure=pressure_interpolate(Z)
            )
            - humidity_ground
        )

    altitude_of_LCL = optimize.newton(
        rsat_minus_rground, initial_z_guess
    )  # find root of the function, i.e. find z such rsat(z)=r_ground

    return altitude_of_LCL


def get_altitude_LCL(
    pressure: np.array,
    vertical_array: np.array,
    temperature_ground_2d: np.array,
    humidity_ground_2d: np.array,
    initial_z_guess: np.array = config.INITIAL_Z_2D,
    heat_capacity_air: float = config.HEAT_CAPACITY_AIR,
    gravity: float = config.GRAVITY,
    lowest_atmospheric_level: float = config.LOWEST_ATMOSPHERIC_LEVEL,
):
    """Summary

    Args:
        pressure (np.array): Description
        vertical_array (np.array): Description
        temperature_ground_2d (np.array): Description
        humidity_ground_2d (np.array): Description
        initial_z_guess (np.array, optional): Description
        heat_capacity_air (float, optional): Description
        gravity (float, optional): Description
        lowest_atmospheric_level (float, optional): Description

    Deleted Parameters:
        temperature_ground (float): Description
        humidity_ground (float): Description
    """

    z_3D = utils.expand_array_to_zyx_array(
        vertical_array,
        final_shape=(
            vertical_array.shape[0],
            temperature_ground_2d.shape[0],
            temperature_ground_2d.shape[1],
        ),
    )
    delta_z_3d = lowest_atmospheric_level - z_3D

    t_linear_3d = temperature_ground_2d + config.GRAVITY / config.HEAT_CAPACITY_AIR * delta_z_3d
    t_linear_3d = t_linear_3d.reshape(
        t_linear_3d.shape[0], t_linear_3d.shape[1] * t_linear_3d.shape[2]
    )
    t_linear_3d_inter = interpolate.interp1d(vertical_array, t_linear_3d, axis=0)

    pressure_3d = utils.expand_array_to_zyx_array(
        pressure,
        final_shape=(
            vertical_array.shape[0],
            temperature_ground_2d.shape[0],
            temperature_ground_2d.shape[1],
        ),
    )
    pressure_3d = pressure_3d.reshape(
        pressure_3d.shape[0], pressure_3d.shape[1] * pressure_3d.shape[2]
    )
    pressure_interpolate = interpolate.interp1d(vertical_array, pressure_3d, axis=0)

    r0_minus_rsat_0 = humidity_ground_2d - saturation_mixing_ratio(
        temperature_in_kelvin=temperature_ground_2d,
        pressure=pressure[0] * np.ones_like(temperature_ground_2d),
    )

    indexes_condensation_from_bottom = np.where(np.abs(r0_minus_rsat_0) < 0.001)

    def rsat_minus_rground(Z: np.array):

        """Summary

        Args:
            Z (np.array): Description

        Returns:
            TYPE: Description
        """

        Z_flat = Z.reshape((Z.shape[0] * Z.shape[1]))

        t_2D = np.array(
            [t_linear_3d_inter(Z_flat[i])[i] for i in range(Z.shape[0] * Z.shape[1])]
        )
        p_2D = np.array(
            [pressure_interpolate(Z_flat[i])[i] for i in range(Z.shape[0] * Z.shape[1])]
        )

        t_2D = t_2D.reshape(Z.shape[0], Z.shape[1])
        p_2D = p_2D.reshape(Z.shape[0], Z.shape[1])

        Id = np.ones_like(Z)
        Id[indexes_condensation_from_bottom] = (
            Z[indexes_condensation_from_bottom]
            - lowest_atmospheric_level * np.ones_like(Z)[indexes_condensation_from_bottom]
        )

        return (
            saturation_mixing_ratio(temperature_in_kelvin=t_2D, pressure=p_2D)
            - humidity_ground_2d * Id
        )

    altitude_of_LCL = optimize.newton(
        rsat_minus_rground, initial_z_guess
    )  # find root of the function, i.e. find z such rsat(z)=r_ground

    return altitude_of_LCL


def get_variable_3D_at_lcl(variable: np.array, map_z_lcl: np.array, vertical_array: np.array):
    """Summary

    Args:
        variable (np.array): Description
        map_z_lcl (np.array): Description
        vertical_array (np.array): Description

    Returns:
        TYPE: Description
    """
    variable = variable.reshape(variable.shape[0], variable.shape[1] * variable.shape[2])
    variable_interpolate = interpolate.interp1d(vertical_array, variable, axis=0)

    map_z_lcl_flat = map_z_lcl.reshape(map_z_lcl.shape[0] * map_z_lcl.shape[1])

    output_2D = np.array(
        [
            variable_interpolate(map_z_lcl_flat[i])[i]
            for i in range(map_z_lcl.shape[0] * map_z_lcl.shape[1])
        ]
    )
    output_2D = output_2D.reshape(map_z_lcl.shape[0], map_z_lcl.shape[1])

    return output_2D


def dry_moist_static_energy_2D(
    temperature_2D: np.array,
    altitudes_2D: np.array,
    pressure_2D: np.array,
    gravity: float = config.GRAVITY,
    cp: float = config.HEAT_CAPACITY_AIR,
    L_cond: float = config.L_c,
    L_sub: float = config.L_s,
):
    """Summary

    Args:
        temperature_2D (np.array): Description
        altitudes_2D (np.array): Description
        pressure_2D (np.array): Description
        gravity (float, optional): Description
        cp (float, optional): Description
        L_cond (float, optional): Description
        L_sub (float, optional): Description
    """
    r_sat = saturation_mixing_ratio(temperature_2D, pressure_2D)
    w_n = omega_n(temperature_2D)

    return temperature_2D + gravity / cp * altitudes_2D


def moist_static_energy(
    temperature: np.array,
    humidity_ground: float,
    altitudes: np.array,
    pressure: np.array,
    gravity: float = config.GRAVITY,
    cp: float = config.HEAT_CAPACITY_AIR,
    L_cond: float = config.L_c,
    L_sub: float = config.L_s,
):
    """Summary

    Args:
        temperature (np.array): Description
        humidity_ground (float): Description
        altitudes (np.array): Description
        pressure (np.array): Description
        gravity (float, optional): Description
        cp (float, optional): Description
        L_cond (float, optional): Description
        L_sub (float, optional): Description
    """
    r_sat = saturation_mixing_ratio(temperature, pressure)
    w_n = omega_n(temperature)
    z_3D = utils.expand_array_to_zyx_array(input_array=altitudes, final_shape=temperature.shape)

    return (
        temperature
        + gravity / cp * z_3D
        - (
            utils.max_point_wise(
                matrix_1=np.zeros_like(r_sat),
                matrix_2=humidity_ground * np.ones_like(r_sat) - r_sat,
            )
        )
        / cp
        * (L_cond * w_n + L_sub * (1 - w_n))
    )


def get_temperature_profile_parcel(
    temperature: np.array,
    humidity_ground: float,
    altitudes: np.array,
    pressure: np.array,
    moist_static_energy_to_conserve: np.array,
    initial_T_guess: np.array,
    gravity: float = config.GRAVITY,
    cp: float = config.HEAT_CAPACITY_AIR,
    L_cond: float = config.L_c,
    L_sub: float = config.L_s,
):
    """Summary

    Args:
        temperature (np.array): Description
        humidity_ground (float): Description
        altitudes (np.array): Description
        pressure (np.array): Description
        moist_static_energy_to_conserve (np.array): Description
        initial_T_guess (np.array): Description
        gravity (float, optional): Description
        cp (float, optional): Description
        L_cond (float, optional): Description
        L_sub (float, optional): Description
    """

    def diff_MSE(
        temperature: np.array,
        humidity_ground: float,
        altitudes: np.array,
        pressure: np.array,
        moist_static_energy_to_conserve: np.array,
    ):
        """Summary

        Args:
            temperature (np.array): Description
            humidity_ground (float): Description
            altitudes (np.array): Description
            pressure (np.array): Description
            moist_static_energy_to_conserve (np.array): Description
        """
        return moist_static_energy_to_conserve - moist_static_energy(
            temperature=temperature,
            humidity_ground=humidity_ground,
            altitudes=altitudes,
            pressure=pressure,
        )

    temperature_profile_parcel = optimize.newton(
        func=diff_MSE,
        x0=initial_T_guess,
        args=(
            humidity_ground,
            altitudes,
            pressure,
            moist_static_energy_to_conserve,
        ),
    )

    return temperature_profile_parcel


def get_parcel_ascent(
    temperature: np.array,
    humidity_ground: np.array,
    pressure: np.array,
    vertical_array: np.array,
):

    temperature_ground = temperature[0, :, :]
    temperature_mean_3d = utils.expand_array_to_zyx_array(
        np.mean(temperature, axis=(1, 2)), final_shape=temperature.shape
    )

    lcl_altitudes = get_altitude_LCL(
        pressure=pressure,
        vertical_array=vertical_array,
        temperature_ground_2d=temperature_ground,
        humidity_ground_2d=humidity_ground,
    )

    lcl_temperatures = get_variable_3D_at_lcl(
        variable=temperature, map_z_lcl=lcl_altitudes, vertical_array=vertical_array
    )

    pressure_3d = utils.expand_array_to_zyx_array(pressure, final_shape=temperature.shape)
    lcl_pressures = get_variable_3D_at_lcl(
        variable=pressure_3d, map_z_lcl=lcl_altitudes, vertical_array=vertical_array
    )

    lcl_mse = dry_moist_static_energy_2D(
        temperature_2D=lcl_temperatures, altitudes_2D=lcl_altitudes, pressure_2D=lcl_pressures
    )

    temperature_parcel_field = get_temperature_profile_parcel(
        temperature=temperature,
        humidity_ground=humidity_ground,
        altitudes=vertical_array,
        pressure=pressure,
        moist_static_energy_to_conserve=lcl_mse,
        initial_T_guess=temperature_mean_3d,
    )

    return temperature_parcel_field
