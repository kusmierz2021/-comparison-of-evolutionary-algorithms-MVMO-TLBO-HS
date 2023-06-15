from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.tlbo import TLBO
import threading
from optimization_functions.optimization_functions import *
from optimization_functions.CEC2022 import CEC2022
import pickle


if __name__ == '__main__':
    rastrigins_boundaries = (-5.12, 5.12)
    zakharov_boundaries = (-5, 10)
    cec_zakharov_boundaries = (-100, 100)
    rosenbrock_boundaries = (-10, 10)
    levy_boundaries = (-10, 10)

    # tlbo_optimizer = TLBO(100_000, 6, rastrigins_boundaries, True)
    # tlbo_population = tlbo_optimizer.init_population(100)
    # tlbo_optimizer.optimize(tlbo_population, rastrigins_function)

    # hs in this case is so much better with 100 as init population size
    # hs_optimizer = HS(10_000, 6, rastrigins_boundaries, True, hmcr=0.9)
    # hs_population = hs_optimizer.init_population(100)
    # hs_optimizer.optimize(hs_population, rastrigins_function)

    # mvmo_optimizer = MVMO(100_000, 6, rastrigins_boundaries, True, mutation_size=1)
    # mvmo_population = mvmo_optimizer.init_population(100)
    # mvmo_optimizer.optimize(mvmo_population, rastrigins_function)

    # it was totally surprising result!
    # tlbo_optimizer = TLBO(100_000, 6, zakharov_boundaries, False)
    # tlbo_population = tlbo_optimizer.init_population(10)
    # tlbo_optimizer.optimize(tlbo_population, zakharov_function)

    # hs_optimizer = HS(100_000, 6, zakharov_boundaries, False, hmcr=0.9)
    # hs_population = hs_optimizer.init_population(100)
    # hs_optimizer.optimize(hs_population, zakharov_function)

    # mvmo_optimizer = MVMO(10_000, 6, zakharov_boundaries, False, mutation_size=1)
    # mvmo_population = mvmo_optimizer.init_population(100)
    # mvmo_optimizer.optimize(mvmo_population, zakharov_function)

    # tlbo_optimizer = TLBO(10_000_000, 6, rosenbrock_boundaries, False)
    # tlbo_population = tlbo_optimizer.init_population(5)
    # tlbo_optimizer.optimize(tlbo_population, rosenbrock_function)




    # # 30 hs test for different init population and its size
    # for pop_size in [10, 100, 1000]:
    #     for i in range(10):
    #
    #         hs_optimizer = HS(10, 6, rosenbrock_boundaries, False, hmcr=0.8)
    #         hs_population = hs_optimizer.init_population(pop_size)
    #         with open(f'./hs_populations/init_pop_{pop_size}_nr_{i+1}', 'wb') as handle:
    #             pickle.dump(hs_population, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #         hs_optimizer.optimize(hs_population, rosenbrock_function)

    # with open('./hs_populations/init_pop_100_nr_1', 'rb') as handle:
    #     hs_population = pickle.load(handle)
    #
    # for _ in range(1):
    #     hs_optimizer = HS(100000, 2, rosenbrock_boundaries, False, hmcr=0.93)
    #     hs_population = hs_optimizer.init_population(100)
        # with open('./hs_populations/init_pop_100', 'wb') as handle:
        #     pickle.dump(hs_population, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # hs_optimizer.optimize(hs_population, rosenbrock_function)
        # hs_optimizer = HS(10, 2, rosenbrock_boundaries, False, hmcr=0.93, par=0.18)
        # hs_optimizer.optimize(hs_population, rosenbrock_function)

    # mvmo_optimizer = MVMO(1_000_000, 2, rosenbrock_boundaries, False, mutation_size=1)
    # mvmo_population = mvmo_optimizer.init_population(10)
    # mvmo_optimizer.optimize(mvmo_population, rosenbrock_function)


    # TODO: cec testing does not work, changes in CEC2022 code are needed
    # cec_levy_function = CEC2022.cec2022_func(func_num=5).values
    # cec_zakharov_function = CEC2022.cec2022_func(func_num=1).values

    # tlbo_optimizer = TLBO(10_000_000, 10, cec_zakharov_boundaries, False)
    # tlbo_population = tlbo_optimizer.init_population(100)
    # # tlbo_optimizer.optimize(tlbo_population, levy_function)
    # tlbo_optimizer.optimize(tlbo_population, cec_zakharov_function)

    # hs_optimizer = HS(10_000_000, 10, cec_zakharov_boundaries, False, hmcr=0.8)
    # hs_population = hs_optimizer.init_population(25)
    # # hs_optimizer.optimize(hs_population, levy_function)
    # hs_optimizer.optimize(hs_population, cec_zakharov_function)

    # mvmo_optimizer = MVMO(10_000_00000, 10, cec_zakharov_boundaries, False, mutation_size=1)
    # mvmo_population = mvmo_optimizer.init_population(10)
    # # mvmo_optimizer.optimize(mvmo_population, levy_function)
    # mvmo_optimizer.optimize(mvmo_population, cec_zakharov_function)


    #
    # # comparing with other TLBO implementation
    # tlbo_optimizer = TLBO(100, 6, (-10, 10), False)
    # # tlbo_population = tlbo_optimizer.init_population(50)
    # # with open('./tlbo_populations/init_pop_50', 'wb') as handle:
    # #     pickle.dump(tlbo_population, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('./tlbo_populations/init_pop_50', 'rb') as handle:
    #     tlbo_population = pickle.load(handle)
    #
    #
    # tlbo_optimizer.optimize(tlbo_population, rastrigins_function)



