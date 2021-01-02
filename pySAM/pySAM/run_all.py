import matplotlib.pyplot as plt
import numpy as np
import pySAM
import xarray as xr
from simulation import Simulation

simulation = Simulation(run="squall4", velocity="10", depth_shear="1000")


if __name__ == "__main__":
    """
    print(simulation.squall_line.PW)
    plt.imshow(simulation.squall_line.PW[-1])
    # plt.show()

    print(simulation.squall_line.distribution_angles)

    simulation.squall_line.set_distribution_angles(
        data_name="PW",
        angles_range=np.linspace(0, np.pi, 5),
        mu=pySAM.MU_GAUSSIAN,
        sigma=pySAM.SIGMA_GAUSSIAN,
        parallelize=False,
    )
    """
# print(simulation.cold_pool.BUOYANCY)

# plt.imshow(simulation.cold_pool.BUOYANCY[5, :, :, 5])
# plt.show()

simulation.cold_pool.set_composite_variables(
    data_name="W",
    variable_to_look_for_extreme="PRECi",
    extreme_events_choice="max",
    x_margin=10,
    y_margin=2,
)
