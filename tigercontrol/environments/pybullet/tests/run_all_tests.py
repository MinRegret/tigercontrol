from tigercontrol.environments.pybullet.tests.test_cartpole import test_cartpole
from tigercontrol.environments.pybullet.tests.test_cartpole_double import test_cartpole_double
from tigercontrol.environments.pybullet.tests.test_cartpole_swingup import test_cartpole_swingup
from tigercontrol.environments.pybullet.tests.test_simulator import test_simulator
from tigercontrol.environments.pybullet.tests.test_kuka import test_kuka
from tigercontrol.environments.pybullet.tests.test_kuka_diverse import test_kuka_diverse
from tigercontrol.environments.pybullet.tests.test_minitaur import test_minitaur
from tigercontrol.environments.pybullet.tests.test_ant import test_ant
from tigercontrol.environments.pybullet.tests.test_humanoid import test_humanoid


def run_all_tests(steps=1000, show=False):
    print("\nrunning all pybullet environments tests...\n")
    test_simulator(verbose=show)
    test_cartpole(verbose=show)
    test_cartpole_swingup(verbose=show)
    test_cartpole_double(verbose=show)
    test_kuka(verbose=show)
    test_kuka_diverse(verbose=show)
    test_minitaur(verbose=show)
    test_ant(verbose=show)
    test_humanoid(verbose=show)
    print("\nall pybullet environments tests passed\n")
  
if __name__ == "__main__":
    run_all_tests(show=False)