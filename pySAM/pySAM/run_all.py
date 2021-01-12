import matplotlib.pyplot as plt
import numpy as np
import pySAM
import xarray as xr
from simulation import Simulation

if __name__ == "__main__":

    # data folder
    data_folder_path = "/Users/sophieabramian/Desktop/SAM_project/data/"

    back_up_folder_path = (
        "/Users/sophieabramian/Desktop/SAM_project/simulation_instances_backup/"
    )

    for velocity in ["2.5", "5", "7.5", "10", "12.5", "15", "17.5", "20"]:
        print(velocity)

        simulation = Simulation(
            data_folder_path=data_folder_path,
            run="squall4",
            velocity=velocity,
            depth_shear="1000",
        )
        simulation.load(back_up_folder_path)

        """
        print("calcul of angle distribution")

        simulation.squall_line.set_distribution_angles(
            data_name="PW",
            angles_range=pySAM.THETA_ARRAY,
            mu=pySAM.MU_GAUSSIAN,
            sigma=pySAM.SIGMA_GAUSSIAN,
            parallelize=True,
        )

        print("calcul of composite variable")

        for data_name in ["BUOYANCY", "W", "QN", "VORTICITY"]:
            print("calcul of" + data_name)

            simulation.cold_pool.set_composite_variables(
                data_name=data_name,
                variable_to_look_for_extreme="PRECi",
                extreme_events_choice="max",
                x_margin=20,
                y_margin=2,
                parallelize=True,
            )

        print("calcul of set_potential_energy")
        for x_size in [5, 10, 15, 20]:
            print("for x_size = " + str(x_size))
            simulation.cold_pool.set_potential_energy(
                data_name="BUOYANCY_composite", x_size=x_size
            )

        simulation.save(back_up_folder_path)
    """
