import numpy as np
from CEC2022 import cec2022_func
import random
from evolutionary_algorithms.mvmo import MVMO
import time


cec_optimum_dict = {
        1: 300,
        2: 400,
        3: 600,
        4: 800,
        5: 900,
        6: 1800,
        7: 2000,
        8: 2200,
        9: 2300,
        10: 2400,
        11: 2600,
        12: 2700
}
# nx = 10  # dimensions - 2, 10, 20
# mx = 5  # population size - int
fx_n = 1  # function - 1, 2, 3, ..., 11, 12
maxFES = 200_000
boundaries = (-100, 100)
dimensions = 10
CEC = cec2022_func(func_num=fx_n)
runs = 30
pop_size = 10

seed_ind = (dimensions / 10 * fx_n * runs + 1) - runs
seed_ind = seed_ind % 1000 + 1

with open('./input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        print(seeds)
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]



np.random.seed(seeds[int(seed_ind - 1)])
random.seed(seeds[int(seed_ind - 1)])

mvmo_optimizer = MVMO(maxFES, dimensions, boundaries, False, mutation_size=1, cec_optimum=cec_optimum_dict[fx_n], cec_error_value=10 ** -8)
mvmo_population = mvmo_optimizer.init_population(pop_size)
start = time.time()
mvmo_optimizer.optimize(mvmo_population, CEC.values)
end = time.time()


print(f'time required: {end - start}')
