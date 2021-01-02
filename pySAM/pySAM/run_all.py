import matplotlib.pyplot as plt
import numpy as np
import pySAM
import xarray as xr
from simulation import Simulation

simulation = Simulation(run="squall4", velocity="10", depth_shear="1000")


if __name__ == "__main__":
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
