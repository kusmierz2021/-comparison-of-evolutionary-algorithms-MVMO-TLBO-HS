import numpy as np
import pickle
import os
from CEC2022 import cec2022_func
import random
from evolutionary_algorithms.hs import HS



np.random.seed(42)
random.seed(42)



nx = 10  # dimensions - 2, 10, 20
mx = 1  # population size - int
fx_n = 1  # function - 1, 2, 3, ..., 11, 12
maxFES = 200000
boundaries = (-100, 100)
CEC = cec2022_func(func_num=fx_n)

# left for tests
x = 200.0*np.random.rand(nx,mx)*0.0-100.0

# read population
with open('./populations/init_pop_10_nr_1', 'rb') as handle:
        population = pickle.load(handle)

print(np.array(population))
print(np.transpose(np.array(population)))
# population = np.array(population)
# results = CEC.values(x)
# print(results)

# start = time.time()
# HS((maxFES / mx), nx, boundaries, False, hmcr=0.75, par=0.50).optimize(population, zakharov_function)[1]
# end = time.time()
# my_times.append(end - start)