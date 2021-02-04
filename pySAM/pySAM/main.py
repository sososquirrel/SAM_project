"will be removed"
import pySAM
from pySAM.simulation import Simulation
from pySAM.utils import generate_1d_2d_3d_paths

if __name__ == "__main__":

    # data folder
    DATA_FOLDER_PATH = "/Users/sophieabramian/Desktop/SAM_project/data/"

    BACKUP_FOLDER_PATH = (
        "/Users/sophieabramian/Desktop/SAM_project/simulation_instances_backup/"
    )

    for velocity in ["17.5"]:
        print(velocity)

        DATA_FOLDER_PATHS = generate_1d_2d_3d_paths(
            run="squall4",
            velocity=velocity,
            depth_shear="1000",
            data_folder_path=DATA_FOLDER_PATH,
        )

        simulation = Simulation(
            data_folder_paths=DATA_FOLDER_PATHS,
            run="squall4",
            velocity=velocity,
            depth_shear="1000",
        )

        simulation.load(BACKUP_FOLDER_PATH)
        simulation.save(BACKUP_FOLDER_PATH)
        """
        print("calcul of max variance")

        simulation.squall_line.set_maximum_variance_step(data_name="PW")

        print("calcul of angle distribution")


        simulation.squall_line.set_distribution_angles(
            data_name="PW",
            angles_range=pySAM.THETA_ARRAY,
            mu=pySAM.MU_GAUSSIAN,
            sigma=pySAM.SIGMA_GAUSSIAN,
            parallelize=True,
        )

        print("calcul of composite variable")

        for data_name in ["BUOYANCY", "QPEVP", "W", "QN", "VORTICITY"]:
            print("composite calcul of " + data_name)

            simulation.cold_pool.set_composite_variables(
                data_name=data_name,
                variable_to_look_for_extreme="PRECi",
                extreme_events_choice="max",
                x_margin=40,
                y_margin=2,
                parallelize=False,
            )

        print("calcul of set_potential_energy")
        for x_size in [5, 10, 15, 20]:
            print("for x_size = " + str(x_size))
            simulation.cold_pool.set_potential_energy(
                data_name="BUOYANCY_composite", x_size=x_size
            )

        simulation.save(BACKUP_FOLDER_PATH)
        """
