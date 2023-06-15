from hs import HS
from optimization_comparison.optimization_functions import rastrigins_function


def test_reproduction():
    dimensions = 6
    boundaries = (-5.12, 5.12)
    optimizer = HS(10000, dimensions, boundaries, True)
    population = optimizer.init_population(5)
    child = optimizer.reproduction(population)
    assert len(child) == dimensions

    # TODO: sometimes fails
    assert all(boundaries[0] <= gene <= boundaries[1] for gene in child)


def test_evaluation():
    boundaries = (-5.12, 5.12)
    optimizer = HS(10000, 6, boundaries, True)
    population = optimizer.init_population(5)
    child = optimizer.reproduction(population)
    evaluated_population = optimizer.evaluation(population, rastrigins_function, child)

    assert len(evaluated_population) == len(population)
