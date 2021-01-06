import matplotlib.pyplot as plt
import numpy as np
import pySAM
import xarray as xr
from simulation import Simulation

if __name__ == "__main__":
    # data folder
    data_folder_path = "/Users/sophieabramian/Desktop/SAM_project/data/"

    simulation = Simulation(
        data_folder_path=data_folder_path, run="squall4", velocity="10", depth_shear="1000"
    )

    print(simulation.squall_line.PW)
    # plt.imshow(simulation.squall_line.PW[-1])
    # plt.show()

    print(simulation.squall_line.distribution_angles)

    simulation.squall_line.set_distribution_angles(
        data_name="PW",
        angles_range=pySAM.THETA_ARRAY,
        mu=pySAM.MU_GAUSSIAN,
        sigma=pySAM.SIGMA_GAUSSIAN,
        parallelize=True,
    )
    print(simulation.squall_line.distribution_angles.shape)
    print(simulation.squall_line.angle_degrees)
    plt.plot(pySAM.THETA_ARRAY - np.pi / 2, simulation.squall_line.distribution_angles)
    plt.show()


# print(simulation.cold_pool.BUOYANCY)

# plt.imshow(simulation.cold_pool.BUOYANCY[5, :, :, 5])
# plt.show()
"""
simulation.cold_pool.set_composite_variables(
    data_name="BUOYANCY",
    variable_to_look_for_extreme="PRECi",
    extreme_events_choice="max",
    x_margin=20,
    y_margin=2,
    parallelize=False,
)

print(simulation.cold_pool.BUOYANCY_composite.shape)
test = simulation.cold_pool.BUOYANCY_composite
plt.imshow(test, cmap="jet")
plt.show()
"""
