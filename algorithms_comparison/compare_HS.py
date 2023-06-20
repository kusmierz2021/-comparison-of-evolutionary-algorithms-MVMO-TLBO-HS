from objective_function import ObjectiveFunction
from pyharmonysearch import harmony_search
from optimization_functions.optimization_functions import rosenbrock_function
import numpy as np
import random
import logging
import time
import pickle


assert round(rosenbrock_function(np.array([1, 1, 1, 1, 1, 1])), 2) == 0

logging.basicConfig(filename='hs_compare.log', filemode='w', format='%(message)s')
np.random.seed(42)
random.seed(42)

pop_size_num = {
    10: 10_000,
    100: 1000,
    1000: 100
}

rosenbrock_boundaries = (-10, 10)
dimensions = 6


if __name__ == '__main__':
    obj_fun = ObjectiveFunction()
    assert round(obj_fun.get_fitness(np.array([1, 1, 1, 1, 1, 1])), 2) == 0
    num_processes = 1
    num_iterations = 150  # because random_seed is defined, there's no point in running this multiple times
    initial_harmonies = [[random.randint(-10, 10) for _ in range(6)] for _ in range(100)]
    results = harmony_search(obj_fun, num_processes, num_iterations, initial_harmonies=initial_harmonies)
    print('Elapsed time: {}\nBest harmony: {}\nBest fitness: {}'.format(results.elapsed_time, results.best_harmony, results.best_fitness))
