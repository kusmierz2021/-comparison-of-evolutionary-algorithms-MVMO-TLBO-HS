# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:14:04 2022

@author: Abhishek Kumar
"""

import numpy as np
import pickle
import os
from CEC2022 import cec2022_func
import random
from evolutionary_algorithms.mvmo import MVMO
from optimization_functions.optimization_functions import rosenbrock_function, zakharov_function

np.random.seed(42)
random.seed(42)



nx = 10  # dimensions - 2, 10, 20
mx = 10  # population size - int
fx_n = 12  # function - 1, 2, 3, ..., 11, 12

CEC = cec2022_func(func_num=fx_n)

# left for tests
x = 200.0*np.random.rand(nx,mx)*0.0-100.0

# read population
with open('./populations/init_pop_10_nr_1', 'rb') as handle:
        population = pickle.load(handle)

results = CEC.values(population)
print(results)
