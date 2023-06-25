from algorithms_comparison.MVMO import MVMO as another_MVMO
from evolutionary_algorithms.mvmo import MVMO
from optimization_functions.optimization_functions import rosenbrock_function, zakharov_function
import numpy as np
import random
import logging
import time
import pickle

assert round(zakharov_function(np.array([0, 0, 0, 0, 0, 0])), 2) == 0
assert round(rosenbrock_function(np.array([1, 1, 1, 1, 1, 1])), 2) == 0

logging.basicConfig(filename='mvmo_compare_v2_z.log', filemode='w', format='%(message)s')
np.random.seed(42)
random.seed(42)

pop_size_num = {
    10: 10_000,
    100: 1000,
    1000: 100
}

zakharov_boundaries = (-10, 10)
rosenbrock_boundaries = (-10, 10)
dimensions = 6
mutation_size = 1
iterations = 1000
bds = [(zakharov_boundaries[0], zakharov_boundaries[1]) for _ in range(dimensions)]

for pop_size in [10, 100, 1000]:
    my_results = []
    my_times = []
    other_results = []
    other_times = []
    for i in range(pop_size_num[pop_size]):
        with open(f'./populations/init_pop_{pop_size}_nr_{i+1}', 'rb') as handle:
            population = pickle.load(handle)
        normalized_population = [(ind - zakharov_boundaries[0]) / (zakharov_boundaries[1] - zakharov_boundaries[0]) for ind in population]

        start = time.time()
        my_results.append(MVMO(iterations, dimensions, zakharov_boundaries, False, mutation_size=mutation_size).optimize(population, zakharov_function)[1])
        end = time.time()
        my_times.append(end - start)

        start = time.time()
        other_results.append(zakharov_function(another_MVMO(logger=False, iterations=iterations, num_mutation=mutation_size, population_size=len(normalized_population)).optimize(obj_fun=zakharov_function, bounds=bds)['x']))
        end = time.time()
        other_times.append(end - start)

    logging.warning(f'pop_size -> {pop_size}\n')
    logging.warning(f'my results\n\tmean -> {sum(my_results) / len(my_results)}\n\t{my_results}')
    logging.warning(f'my times\n\tmean -> {sum(my_times) / len(my_times)}\n\t{my_times}\n')
    logging.warning(f'other results\n\tmean -> {sum(other_results) / len(other_results)}\n\t{other_results}')
    logging.warning(f'other times\n\tmean -> {sum(other_times) / len(other_times)}\n\t{other_times}\n\n')
