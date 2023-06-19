import numpy as np
from another_TLBO import fitness_rastrigin, tlbo
from evolutionary_algorithms.tlbo import TLBO
from optimization_functions.optimization_functions import rastrigins_function
import pickle
import random
import logging
assert round(rastrigins_function(np.array([4.52299, 4.52299, 4.52299, 4.52299,
                                                                      4.52299, 4.52299, 4.52299])), 5) == 282.47303
assert round(fitness_rastrigin(np.array([4.52299, 4.52299, 4.52299, 4.52299,
                                                                      4.52299, 4.52299, 4.52299])), 5) == 282.47303

logging.basicConfig(filename='tlbo_compare.log', filemode='w', format='%(message)s')
np.random.seed(42)
random.seed(42)
rastrigins_boundaries = (-5.12, 5.12)


# prepare populations
# tlbo_optimizer = TLBO(100_000, 6, rastrigins_boundaries, False)
# for pop_size in [10, 100, 1000]:
#     for i in range(100):
#         tlbo_population = tlbo_optimizer.init_population(pop_size)
#         with open(f'./tlbo_populations/init_pop_{pop_size}_nr_{i+1}', 'wb') as handle:
#             pickle.dump(tlbo_population, handle, protocol=pickle.HIGHEST_PROTOCOL)

# read population
# with open('./tlbo_populations/init_pop_100_nr_1', 'rb') as handle:
#         population = pickle.load(handle)


for pop_size in [10, 100, 1000]:
    my_results = []
    other_results = []
    for i in range(100):
        with open(f'./tlbo_populations/init_pop_{pop_size}_nr_{i+1}', 'rb') as handle:
            population = pickle.load(handle)

        tlbo_optimizer = TLBO(100, 6, rastrigins_boundaries, False)
        my_results.append(rastrigins_function(tlbo_optimizer.optimize(population, rastrigins_function)))
        other_results.append(fitness_rastrigin(tlbo(fitness_rastrigin, 100, population, rastrigins_boundaries)))
    logging.warning(f'pop_size -> {pop_size}')
    logging.warning(f'my results\n\tmean -> {sum(my_results)/len(my_results)}\n\t{my_results}')
    logging.warning(f'other results\n\tmean -> {sum(other_results)/len(other_results)}\n\t{other_results}')
