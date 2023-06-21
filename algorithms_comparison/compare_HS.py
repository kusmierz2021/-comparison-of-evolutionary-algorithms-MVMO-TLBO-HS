from objective_function import ObjectiveFunction
from pyharmonysearch import harmony_search
from evolutionary_algorithms.hs import HS
from optimization_functions.optimization_functions import rosenbrock_function
import numpy as np
import random
import logging
import time
import pickle


assert round(rosenbrock_function(np.array([1, 1, 1, 1, 1, 1])), 2) == 0


np.random.seed(42)
random.seed(42)

pop_size_num = {
    10: 10_000,
    100: 1000,
    1000: 100
}

rosenbrock_boundaries = (-10, 10)
dimensions = 6
iterations = 100

if __name__ == '__main__':

    logging.basicConfig(filename=f'hs_compare.log', filemode='w', format='%(message)s')
    num_processes = 1
    workers = 1

    for pop_size in [10, 100, 1000]:
        my_results = []
        my_times = []
        other_results = []
        other_times = []
        obj_fun = ObjectiveFunction(iterations=iterations, population_size=pop_size)
        assert round(obj_fun.get_fitness(np.array([1, 1, 1, 1, 1, 1])), 2) == 0

        for i in range(pop_size_num[pop_size]):
            with open(f'./populations/init_pop_{pop_size}_nr_{i + 1}', 'rb') as handle:
                population = pickle.load(handle)

            start = time.time()
            my_results.append(HS(iterations, dimensions, rosenbrock_boundaries, False, hmcr=0.75, par=0.50).optimize(population,
                                                                                                          rosenbrock_function)[1])
            end = time.time()
            my_times.append(end - start)

            initial_harmonies = [ind.tolist() for ind in population]

            start = time.time()
            other_results.append(harmony_search(obj_fun, num_processes, workers, initial_harmonies=initial_harmonies).best_fitness)
            end = time.time()
            other_times.append(end - start)

        logging.warning(f'pop_size -> {pop_size}\n')
        logging.warning(f'my results\n\tmean -> {sum(my_results) / len(my_results)}\n\t{my_results}')
        logging.warning(f'my times\n\tmean -> {sum(my_times) / len(my_times)}\n\t{my_times}\n')
        logging.warning(f'other results\n\tmean -> {sum(other_results) / len(other_results)}\n\t{other_results}')
        logging.warning(f'other times\n\tmean -> {sum(other_times) / len(other_times)}\n\t{other_times}\n\n')
