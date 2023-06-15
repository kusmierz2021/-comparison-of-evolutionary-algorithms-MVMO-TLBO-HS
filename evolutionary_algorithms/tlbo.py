from evolutionary_algorithm import EvolutionaryAlgorithm
import numpy as np
from random import random, choice, seed, randint
from tqdm import tqdm
from optimization_comparison.optimization_functions import rastrigins_function


class TLBO(EvolutionaryAlgorithm):
    def __init__(self, iterations: int, dimensions: int, boundaries: tuple[float, float], maximize: bool, set_seed=None):
        """
        Teaching Learning Based Optimization Algorithm
        :param iterations: number of iterations during optimization
        :type iterations: int
        :param dimensions: number of dimensions of optimization function
        :type dimensions: int
        :param boundaries: lower and higher limit of the range of every gene
        :type boundaries: tuple of floats
        :param maximize: True for maximization, False for minimization
        :type maximize: bool
        """
        if set_seed is not None:
            seed(seed)
        super().__init__(iterations, dimensions, boundaries, maximize)

    def optimize(self, population: list[np.ndarray], optimize_function: callable):
        """
        Searches for the best solution for a given number of iterations
        :param population: initial population
        :type population: list[np.ndarray]
        :param optimize_function:
        :type optimize_function: callable
        :return: best from found solutions
        :rtype: numpy.ndarray
        """
        best_individual = self.evaluation(population, optimize_function)[1]
        print(f"new best: {best_individual} -> {optimize_function(best_individual)}")

        for _ in tqdm(range(self.iterations)):
            evaluated_mutated_population = self.mutation(population, optimize_function)
            population = self.crossover(evaluated_mutated_population, optimize_function)
            potential_best_individual = self.evaluation(population, optimize_function)[1]

            if (optimize_function(potential_best_individual) > optimize_function(best_individual)) if self.maximize \
                    else (optimize_function(potential_best_individual) < optimize_function(best_individual)):
                best_individual = potential_best_individual
                print(f"new best: {best_individual} -> {optimize_function(best_individual)}")
        return best_individual

    # TODO: before changes, it seems to be some problem with mutagen
    # def mutation(self, population: list[np.ndarray], fitness_function: callable):
    #     """
    #     Mutates every individual in the population
    #     :param population: population to be mutated
    #     :type population: list[numpy.ndarray]
    #     :param fitness_function: function to evaluate how close a given solution is to the optimum solution
    #     :type fitness_function: callable
    #     :return: evaluated mutated population
    #     :rtype: list[tuple[numpy.ndarray, float]]
    #     """
    #     evaluated_population, best_individual, mean_individual = self.evaluation(population, fitness_function)
    #     mutation_rate = round((random()+1))
    #     mutagen = np.array([random() * (best - mutation_rate * mean) for (best, mean)
    #                         in zip(best_individual, mean_individual)])
    #     mutated_population = list(map(lambda ind: ind + mutagen, population))
    #     mutated_population = self.ensure_boundaries_population(mutated_population)
    #     evaluated_mutated_population = self.evaluation(mutated_population, fitness_function)[0]
    #
    #     if self.maximize:
    #         return [mutated_ind if mutated_ind[1] > ind[1] else ind for mutated_ind, ind
    #                 in zip(evaluated_mutated_population, evaluated_population)]
    #     else:
    #         return [mutated_ind if mutated_ind[1] < ind[1] else ind for mutated_ind, ind
    #                 in zip(evaluated_mutated_population, evaluated_population)]

    # TODO: now it will work better, take care of optimization results
    def mutation(self, population: list[np.ndarray], fitness_function: callable):
        """
        Mutates every individual in the population
        :param population: population to be mutated
        :type population: list[numpy.ndarray]
        :param fitness_function: function to evaluate how close a given solution is to the optimum solution
        :type fitness_function: callable
        :return: evaluated mutated population
        :rtype: list[tuple[numpy.ndarray, float]]
        """
        evaluated_population, best_individual, mean_individual = self.evaluation(population, fitness_function)
        # mutation_rate = round((random()+1))
        # r = random()
        mutagen_pop = [np.array([random() for _ in range(len(mean_individual))]) *
                       (best_individual - np.array([randint(1, 2) for _ in range(len(mean_individual))]) *
                        mean_individual) for _ in range(len(population))]

        mutated_population = [ind + mutagen for (ind, mutagen) in zip(population, mutagen_pop)]
        mutated_population = self.ensure_boundaries_population(mutated_population)
        evaluated_mutated_population = self.evaluation(mutated_population, fitness_function)[0]

        if self.maximize:
            return [mutated_ind if mutated_ind[1] > ind[1] else ind for mutated_ind, ind
                    in zip(evaluated_mutated_population, evaluated_population)]
        else:
            return [mutated_ind if mutated_ind[1] < ind[1] else ind for mutated_ind, ind
                    in zip(evaluated_mutated_population, evaluated_population)]

    def evaluation(self, population: list[np.ndarray], fitness_function: callable):
        """
        Counts fitness function value for every individual
        :param population: population to be evaluated
        :type population: list[numpy.ndarray]
        :param fitness_function: function to evaluate how close a given solution is to the optimum solution
        :type fitness_function: callable
        :return: evaluated population, best individual, mean individual
        :rtype: tuple[list[tuple[numpy.ndarray, float]], numpy.ndarray, numpy.ndarray]
        """
        evaluated_population = [(ind, fitness_function(ind)) for ind in population]
        best_individual = sorted(evaluated_population, key=lambda ind: ind[1], reverse=self.maximize)[0][0]
        mean_individual = np.mean(population, axis=0)

        return evaluated_population, best_individual, mean_individual

    def ensure_boundaries_individual(self, new_ind: np.ndarray) -> np.ndarray:
        """
        Sets every gene value to lower/higher boundary value if it crosses the given range
        :param new_ind: individual to be validated
        :type new_ind: numpy.ndarray
        :return: validated individual
        :rtype: numpy.ndarray
        """
        return np.array([self.boundaries[0] if gene < self.boundaries[0] else
                        (self.boundaries[1] if gene > self.boundaries[1] else gene) for gene in new_ind])

    def ensure_boundaries_population(self, new_pop: list[np.ndarray]) -> list[np.ndarray]:
        """
        For the entire population sets every gene value to lower/higher boundary value if it crosses the given range
        :param new_pop: population which individuals are to validation
        :type new_pop: list[numpy.ndarray]
        :return: validated population
        :rtype: list[numpy.ndarray]
        """
        return [self.ensure_boundaries_individual(new_ind) for new_ind in new_pop]

    def crossover(self, evaluated_population: list[tuple[np.ndarray, float]], fitness_function: callable)\
            -> list[np.ndarray]:
        """
        For every individual draws other individual with other fitness function value and crosses them
        :param evaluated_population: evaluated population
        :type evaluated_population: list[tuple[numpy.ndarray, float]]
        :param fitness_function: function to evaluate how close a given solution is to the optimum solution
        :type fitness_function: callable
        :return: population after crossover
        :rtype: list[numpy.ndarray]
        """
        crossed_population: list[np.ndarray] = []
        r = random()
        for ind in evaluated_population:
            to_choose = list(filter(lambda individual: individual[1] != ind[1], evaluated_population))
            if len(to_choose) == 0:
                return [individual[0] for individual in evaluated_population]
            else:
                ind_to_cross = choice(to_choose)

            if self.maximize:
                if ind[1] > ind_to_cross[1]:
                    # TODO: ten random to chyba powinien być jeden dla całego osobnika, a jest dla każdego genu
                    #  oddzielny
                    #  ten random to w ogóle chyba powinien być jeden na całą iterację
                    #  a z Tf to trochę nie mogę doczytać, wygląda jakby miał być jeden na całe działanie algorytmu,
                    #  ale wtedy byłby parametrem, więc raczej faktycznie musi być losowany częściej,
                    #  niech będzie tak jak tutaj z random(), że jest losowany dla każdego genu oddzielnie
                    new_ind = np.array([g1 + r * (g1 - g2) for g1, g2 in zip(ind[0], ind_to_cross[0])])
                else:
                    new_ind = np.array([g1 + r * (g2 - g1) for g1, g2 in zip(ind[0], ind_to_cross[0])])
            else:
                if ind[1] < ind_to_cross[1]:
                    new_ind = np.array([g1 + r * (g1 - g2) for g1, g2 in zip(ind[0], ind_to_cross[0])])
                else:
                    new_ind = np.array([g1 + r * (g2 - g1) for g1, g2 in zip(ind[0], ind_to_cross[0])])

            new_ind = self.ensure_boundaries_individual(new_ind)
            new_ind = new_ind if ((fitness_function(new_ind) > ind[1] and self.maximize)
                                  or (fitness_function(new_ind) < ind[1] and not self.maximize)) else ind[0]

            crossed_population.append(new_ind)
        return crossed_population


if __name__ == '__main__':
    # boundaries = (-100, 100)
    # optimizer = TLBO(1, 2, boundaries, False)
    # population = optimizer.init_population(5)
    # optimizer.optimize(population, lambda x: x[0]**2 + x[1]**2)

    boundaries = (-5.12, 5.12)
    optimizer = TLBO(10000, 6, boundaries, True)
    population = optimizer.init_population(100)
    optimizer.optimize(population, rastrigins_function)
