import optimization_functions
import numpy as np
from optimization_functions import visualize
from CEC2022 import CEC2022


def test_rastrigins_function():
    # global maximum for different dimensions tested
    assert round(optimization_functions.rastrigins_function(np.array([4.52299])), 5) == 40.35329
    assert round(optimization_functions.rastrigins_function(np.array([4.52299, 4.52299])), 5) == 80.70658
    assert round(optimization_functions.rastrigins_function(np.array([4.52299, 4.52299, 4.52299])), 5) == 121.05987
    assert round(optimization_functions.rastrigins_function(np.array([4.52299, 4.52299, 4.52299,
                                                                      4.52299])), 5) == 161.41316
    assert round(optimization_functions.rastrigins_function(np.array([4.52299, 4.52299, 4.52299,
                                                                      4.52299, 4.52299])), 5) == 201.76645
    assert round(optimization_functions.rastrigins_function(np.array([4.52299, 4.52299, 4.52299,
                                                                      4.52299, 4.52299, 4.52299])), 5) == 242.11974
    assert round(optimization_functions.rastrigins_function(np.array([4.52299, 4.52299, 4.52299, 4.52299,
                                                                      4.52299, 4.52299, 4.52299])), 5) == 282.47303

    # plot Rastrigin's Function (2D)
    # visualize((-5.12, 5.12), optimization_functions.rastrigins_function)


def test_zakharov_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.zakharov_function(np.array([0, 0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.zakharov_function(np.array([0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.zakharov_function(np.array([0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.zakharov_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.zakharov_function(np.array([0, 0])), 2) == 0
    assert round(optimization_functions.zakharov_function(np.array([0])), 2) == 0

    # plot Zakharov Function (2D)
    # cec_zakharov_function = CEC2022.cec2022_func(func_num=1)
    # visualize((-5, 10), optimization_functions.zakharov_function)
    # visualize((-5, 10), cec_zakharov_function.values)


def test_rosenbrock_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.rosenbrock_function(np.array([1, 1, 1, 1, 1, 1])), 2) == 0
    assert round(optimization_functions.rosenbrock_function(np.array([1, 1, 1, 1, 1])), 2) == 0
    assert round(optimization_functions.rosenbrock_function(np.array([1, 1, 1, 1])), 2) == 0
    assert round(optimization_functions.rosenbrock_function(np.array([1, 1, 1])), 2) == 0
    assert round(optimization_functions.rosenbrock_function(np.array([1, 1])), 2) == 0
    assert round(optimization_functions.rosenbrock_function(np.array([1])), 2) == 0

    # plot Rosenbrock's Function (2D)
    # visualize((-10, 10), optimization_functions.rosenbrock_function)


def test_expanded_schaffers_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.expanded_schaffers_function(np.array([0, 0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.expanded_schaffers_function(np.array([0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.expanded_schaffers_function(np.array([0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.expanded_schaffers_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.expanded_schaffers_function(np.array([0, 0])), 2) == 0

    # plot Expaned Schaffer's Function (2D)
    # visualize((-100, 100), optimization_functions.expanded_schaffers_function)


def test_bent_cigar_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.bent_cigar_function(np.array([0, 0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.bent_cigar_function(np.array([0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.bent_cigar_function(np.array([0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.bent_cigar_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.bent_cigar_function(np.array([0, 0])), 2) == 0
    assert round(optimization_functions.bent_cigar_function(np.array([0])), 2) == 0

    # plot Bent Cigar Function (2D)
    # visualize((-100, 100), optimization_functions.bent_cigar_function)


def test_levy_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.levy_function(np.array([1, 1, 1, 1, 1, 1])), 2) == 0
    assert round(optimization_functions.levy_function(np.array([1, 1, 1, 1, 1])), 2) == 0
    assert round(optimization_functions.levy_function(np.array([1, 1, 1, 1])), 2) == 0
    assert round(optimization_functions.levy_function(np.array([1, 1, 1])), 2) == 0
    assert round(optimization_functions.levy_function(np.array([1, 1])), 2) == 0
    assert round(optimization_functions.levy_function(np.array([1])), 2) == 0

    # plot Levy Function (2D)
    # visualize((-10, 10), optimization_functions.levy_function)
    # cec_levy_function = CEC2022.cec2022_func(func_num=5)

    # visualize((-10, 10), cec_levy_function.values)


def test_high_conditioned_elliptic_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.high_conditioned_elliptic_function(np.array([0, 0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.high_conditioned_elliptic_function(np.array([0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.high_conditioned_elliptic_function(np.array([0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.high_conditioned_elliptic_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.high_conditioned_elliptic_function(np.array([0, 0])), 2) == 0

    # plot High Conditioned Elliptic Function (2D)
    # visualize((-100, 100), optimization_functions.high_conditioned_elliptic_function)


def test_happycat_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.happycat_function(np.array([-1, -1, -1, -1, -1, -1])), 2) == 0
    assert round(optimization_functions.happycat_function(np.array([-1, -1, -1, -1, -1])), 2) == 0
    assert round(optimization_functions.happycat_function(np.array([-1, -1, -1, -1])), 2) == 0
    assert round(optimization_functions.happycat_function(np.array([-1, -1, -1])), 2) == 0
    assert round(optimization_functions.happycat_function(np.array([-1, -1])), 2) == 0
    assert round(optimization_functions.happycat_function(np.array([-1])), 2) == 0

    # plot Happycat Function (2D)
    # visualize((-20, 20), optimization_functions.happycat_function)


def test_discus_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.discus_function(np.array([0, 0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.discus_function(np.array([0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.discus_function(np.array([0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.discus_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.discus_function(np.array([0, 0])), 2) == 0

    # plot Discus Function (2D)
    # visualize((-100, 100), optimization_functions.discus_function)


def test_ackleys_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.ackleys_function(np.array([0, 0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.ackleys_function(np.array([0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.ackleys_function(np.array([0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.ackleys_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.ackleys_function(np.array([0, 0])), 2) == 0
    assert round(optimization_functions.ackleys_function(np.array([0])), 2) == 0

    # plot Ackley's Function (2D)
    # visualize((-32.768, 32.768), optimization_functions.ackleys_function)


def test_schaffers_f7_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.schaffers_f7_function(np.array([0, 0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.schaffers_f7_function(np.array([0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.schaffers_f7_function(np.array([0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.schaffers_f7_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.schaffers_f7_function(np.array([0, 0])), 2) == 0

    # plot Schaffer's F7 Function (2D)
    # visualize((-100, 100), optimization_functions.schaffers_f7_function)


def test_hgbat_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.hgbat_function(np.array([-1, -1, -1, -1, -1, -1])), 2) == 0
    assert round(optimization_functions.hgbat_function(np.array([-1, -1, -1, -1, -1])), 2) == 0
    assert round(optimization_functions.hgbat_function(np.array([-1, -1, -1, -1])), 2) == 0
    assert round(optimization_functions.hgbat_function(np.array([-1, -1, -1])), 2) == 0
    assert round(optimization_functions.hgbat_function(np.array([-1, -1])), 2) == 0
    assert round(optimization_functions.hgbat_function(np.array([-1])), 2) == 0

    # plot HGBat Function (2D)
    # visualize((-15, 15), optimization_functions.hgbat_function)


def test_griewanks_function():
    # global minimum for different dimensions tested
    assert round(optimization_functions.griewanks_function(np.array([0, 0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.griewanks_function(np.array([0, 0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.griewanks_function(np.array([0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.griewanks_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.griewanks_function(np.array([0, 0])), 2) == 0

    # plot Griewank's F7 Function (2D)
    # visualize((-100, 100), optimization_functions.griewanks_function)
