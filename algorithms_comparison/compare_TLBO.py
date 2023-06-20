import numpy as np
from another_TLBO import fitness_rosenbrock, tlbo
from evolutionary_algorithms.tlbo import TLBO
from optimization_functions.optimization_functions import rosenbrock_function
import pickle
import random
import logging
import time
assert round(rosenbrock_function(np.array([1, 1, 1, 1, 1, 1])), 2) == 0
assert round(fitness_rosenbrock(np.array([1, 1, 1, 1, 1, 1])), 2) == 0

logging.basicConfig(filename='tlbo_compare.log', filemode='w', format='%(message)s')
np.random.seed(42)
random.seed(42)
rosenbrock_boundaries = (-10, 10)
iterations = 250
dimensions = 6

pop_size_num = {
    10: 10_000,
    100: 1000,
    1000: 100
}

for pop_size in [10, 100, 1000]:
    my_results = []
    my_times = []
    other_results = []
    other_times = []
    for i in range(pop_size_num[pop_size]):
        with open(f'./populations/init_pop_{pop_size}_nr_{i+1}', 'rb') as handle:
            population = pickle.load(handle)

        start = time.time()
        my_results.append(rosenbrock_function(TLBO(iterations, dimensions, rosenbrock_boundaries, False).optimize(population, rosenbrock_function)))
        end = time.time()
        my_times.append(end - start)

        start = time.time()
        other_results.append(fitness_rosenbrock(tlbo(fitness_rosenbrock, iterations, population, rosenbrock_boundaries)))
        end = time.time()
        other_times.append(end - start)

    logging.warning(f'pop_size -> {pop_size}\n')
    logging.warning(f'my results\n\tmean -> {sum(my_results)/len(my_results)}\n\t{my_results}')
    logging.warning(f'my times\n\tmean -> {sum(my_times)/len(my_times)}\n\t{my_times}\n')
    logging.warning(f'other results\n\tmean -> {sum(other_results)/len(other_results)}\n\t{other_results}')
    logging.warning(f'other times\n\tmean -> {sum(other_times)/len(other_times)}\n\t{other_times}\n\n')
