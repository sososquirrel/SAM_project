# MOVE TO PRODUCTION

# import matplotlib.pyplot as plt
# from matplotlib import cm, ticker
# from pySAM.simulation import Simulation

# data_folder_path = "/Users/sophieabramian/Desktop/SAM_project/data/"

# back_up_folder_path = "/Users/sophieabramian/Desktop/SAM_project/simulation_instances_backup/"

# figu, axu = plt.subplots(2)

# for velocity in ["2.5", "5"]:  # ["2.5", "5", "7.5", "10", "12.5", "15", "17.5", "20"]:
#     print(velocity)
#     simulation = Simulation(
#         data_folder_path=data_folder_path,
#         run="squall4",
#         velocity=velocity,
#         depth_shear="1000",
#     )

#     simulation.load(back_up_folder_path)

#     simulation.cold_pool.set_geometry_profil(
#         data_name="BUOYANCY_composite",
#         threshold=-0.015,
#     )
#     axu[0].plot(
#         simulation.cold_pool.profil_15[0],
#         simulation.cold_pool.profil_15[1],
#         color=simulation.color,
#     )

# plt.show()
