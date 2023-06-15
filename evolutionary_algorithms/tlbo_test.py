from tlbo import TLBO
from optimization_functions.optimization_functions import rastrigins_function


def test_mutation():
    dimensions = 6
    boundaries = (-5.12, 5.12)
    optimizer = TLBO(1, dimensions, boundaries, maximize=True)
    population = optimizer.init_population(5)
    mutated_population = optimizer.mutation(population, rastrigins_function)

    assert len(mutated_population) == len(population)
    assert len(mutated_population[0][0]) == len(population[0]) == dimensions


def test_evaluation():
    dimensions = 2
    boundaries = (-5.12, 5.12)
    sphere_function = lambda x: x[0] ** 2 + x[1] ** 2
    optimizer = TLBO(1, dimensions, boundaries, maximize=False)
    population = optimizer.init_population(5)
    evaluated_population, best_individual, mean_individual = optimizer.evaluation(population, sphere_function)

    assert len(best_individual) == dimensions
    assert len(mean_individual) == dimensions
    assert all(sphere_function(best_individual) <= ind[1] for ind in evaluated_population)

    optimizer = TLBO(1, dimensions, boundaries, maximize=True)
    population = optimizer.init_population(5)
    evaluated_population, best_individual, mean_individual = optimizer.evaluation(population, rastrigins_function)

    assert len(best_individual) == dimensions
    assert len(mean_individual) == dimensions
    assert all(rastrigins_function(best_individual) >= ind[1] for ind in evaluated_population)


def test_crossover():
    dimensions = 6
    boundaries = (-5.12, 5.12)
    optimizer = TLBO(1, dimensions, boundaries, maximize=True)
    population = optimizer.init_population(5)
    evaluated_population, best_individual, mean_individual = optimizer.evaluation(population, rastrigins_function)
    crossed_population = optimizer.crossover(evaluated_population, rastrigins_function)
    assert len(crossed_population) == len(population)
    assert len(crossed_population[0]) == len(population[0]) == dimensions
