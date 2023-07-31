from evolutionary_algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
import numpy as np
from optimization_functions.optimization_functions import rastrigins_function
from tqdm import tqdm
import random
import logging


class HS(EvolutionaryAlgorithm):
    def __init__(self, iterations: int, dimensions: int, boundaries: tuple[float, float], maximize: bool,
                 hmcr: float = None, par: float = None, cec_optimum = None, cec_error_value = None):
        """
        Harmony Search Algorithm
        :param iterations: number of iterations during optimization
        :type iterations: int
        :param dimensions: number of dimensions of optimization function
        :type dimensions: int
        :param boundaries: lower and higher limit of the range of every gene
        :type boundaries: tuple of floats
        :param maximize: True for maximization, False for minimization
        :type maximize: bool
        :param hmcr: ranges from 0.0 to 1.0
        :type hmcr: float
        :param par: ranges from 0.0 to 1.0, it is optional parameter
        :type par: float
        """
        logging.basicConfig(filename='hs.log', filemode='a', format='%(message)s')

        super().__init__(iterations, dimensions, boundaries, maximize, cec_optimum, cec_error_value)
        self.hmcr = hmcr
        self.par = par
        self.max_par = 0.25


    def evaluation(self, population: list[np.ndarray], fitness_function: callable, child: np.ndarray):
        population = population + [child]

        # CEC version
        # best_population = sorted(list(zip(population, fitness_function(population))), key=lambda ind: ind[1],
        #                          reverse=self.maximize).copy()[:len(population) - 1]
        best_population = sorted([(ind, fitness_function(ind)) for ind in population], key=lambda ind: ind[1],
                                 reverse=self.maximize).copy()[:len(population) - 1]
        return best_population

    def reproduction(self, population: list[np.ndarray]) -> np.ndarray:
        child = np.empty(self.dimensions, dtype=float)
        for ind in range(self.dimensions):
            if self.hmcr is not None:
                if random.random() > self.hmcr:
                    child[ind] = random.uniform(self.boundaries[0], self.boundaries[1])
                else:
                    child[ind] = random.choice(population)[ind]
            if self.par is not None:
                if random.random() < self.par:
                    if random.random() < 0.5:
                        child[ind] -= (child[ind] - self.boundaries[0]) * self.max_par * random.random()
                    else:
                        child[ind] += (self.boundaries[0] - child[ind]) * self.max_par * random.random()
        return child

    def optimize(self, population: list[np.ndarray], optimize_function: callable):
        # logging.warning(f"NEW VARIANT\niterations: {self.iterations}  dimensions: {self.dimensions}  "
        #                 f"population_size: {len(population)}  hmcr: {self.hmcr}  par: {self.par}  "
        #                 f"optimize_function: {optimize_function.__name__} ")
        best_individual = None

        for i in range(self.iterations):
        # for i in tqdm(range(self.iterations)):


            child = self.reproduction(population)
            evaluated_population = self.evaluation(population, optimize_function, child)

            if best_individual is None:
                best_individual = evaluated_population[0]

                # print(f"new best: {best_individual[0]} -> {best_individual[1]}")
                # logging.warning(f"new best: {best_individual[0]} -> {best_individual[1]}")
            elif ((evaluated_population[0][1] > best_individual[1]) if self.maximize
                  else (evaluated_population[0][1] < best_individual[1])):
                best_individual = evaluated_population[0]
                # print(f"new best: {best_individual[0]} -> {best_individual[1]}")
                # logging.warning(f"new best: {best_individual[0]} -> {best_individual[1]}")
            population = [ind[0] for ind in evaluated_population]
            if self.cec_optimum is not None:
                diff = best_individual[1] - self.cec_optimum
                if diff < self.cec_error_value:
                    logging.warning(f'{i+1}')
                    return best_individual
                if i+1 >= self.k_FES[self.k]:
                    logging.warning(f'{diff}')
                    self.k = self.k + 1
        return best_individual


if __name__ == '__main__':
    boundaries = (-5.12, 5.12)
    optimizer = HS(10000, 6, boundaries, True, hmcr=0.9)
    population = optimizer.init_population(100)
    optimizer.optimize(population, rastrigins_function)
