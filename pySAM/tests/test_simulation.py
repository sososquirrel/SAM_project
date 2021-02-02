from pySAM.simulation import Simulation

path1 = "test_data/RCE_shear_U10_H1000_squall4.nc"
path2 = "test_data/RCE_shear_U10_H1000_64.2Dcom_1_squall4.nc"
path3 = "test_data/RCE_shear_U10_H1000_64_0000302400.com3D.alltimes_squall4.nc"

paths = [path1, path2, path3]

velocity = "10"


def test_Simulation():
    simulation = Simulation(
        data_folder_paths=paths,
        run="squall4",
        velocity=velocity,
        depth_shear="1000",
    )

def test_


if __name__ == "__main__":
    test_Simulation()
