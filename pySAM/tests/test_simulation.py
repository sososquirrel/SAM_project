from pySAM.simulation import Simulation

path1 = "test_data/test_dataset_1D.nc"
path2 = "test_data/test_dataset_2D.nc"
path3 = "test_data/test_dataset_3D.nc"

paths = [path1, path2, path3]

velocity = "10"


def test_Simulation():
    simulation = Simulation(
        data_folder_paths=paths,
        run="squall4",
        velocity=velocity,
        depth_shear="1000",
    )
    print(simulation.cold_pool.U)


if __name__ == "__main__":
    test_Simulation()
