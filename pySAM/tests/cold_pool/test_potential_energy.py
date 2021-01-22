import matplotlib.pyplot as plt
import numpy as np
import pytest

data_array = np.zeros((15, 20))  # idealized cold pool
data_array[0:5, 10:18] = -1
data_array[6, 12] = -1

print(data_array)

z_array = np.linspace(0, 14, 15)

depth_shear = "5"

x_size = 10

plt.imshow(data_array)


def test_hight_max_index():

    cold_pool_hight_max = 1.5 * float(
        depth_shear
    )  # cold pools are known to scale depth shear, here we take 1.5 depth shear depth

    cold_pool_hight_max_index = np.where(z_array < cold_pool_hight_max)[0][-1]

    return cold_pool_hight_max_index


print(test_hight_max_index())


def test_potential_energy():

    potential_energy_array = []

    x_max_precip = int(
        data_array.shape[1] / 2
    )  # remainder : the input must be centered in the maximum precipitation

    cold_pool_hight_max_index = test_hight_max_index()

    print(cold_pool_hight_max_index)

    for x_index in range(x_size):
        data_array_x = data_array[:cold_pool_hight_max_index, x_max_precip + x_index]

        if len(np.where(data_array_x < 0)[0]) == 0:
            potential_energy_x = 0
        else:

            y_intersect_index = np.where(data_array_x < 0)[0][-1]

            print("y_intersect_index", y_intersect_index)
            print("y_intersect", z_array[y_intersect_index])

            dz = np.diff(z_array[: y_intersect_index + 1])
            print("dz", dz)

            potential_energy_x = -np.sum(
                -data_array[:y_intersect_index, x_max_precip + x_index] * dz
            )
            print("potential_energy_x", potential_energy_x)
        potential_energy_array.append(potential_energy_x)

    return -np.array(potential_energy_array)


print(test_potential_energy())

plt.plot(test_potential_energy())
plt.show()
