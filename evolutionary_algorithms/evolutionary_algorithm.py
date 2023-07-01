from numpy.random import uniform
import numpy as np

class EvolutionaryAlgorithm:

    def __init__(self, iterations: int, dimensions: int, boundaries: tuple[float, float], maximize: bool, cec_optimum = None, cec_error_value = None):
        self.iterations = iterations
        self.dimensions = dimensions
        self.boundaries = boundaries
        self.maximize = maximize
        self.cec_optimum = cec_optimum
        self.cec_error_value = cec_error_value
        if cec_optimum is not None:
            self.k_FES = {k: dimensions ** (k / 5 - 3) * iterations for k in range(16)}
            self.k = 0

    def init_population(self, size: int = 2) -> list[np.ndarray]:
        """
        Initialize population of given size with individuals of given dimension and constraints
        :param size: size of initialized population
        :return: population (list) of individuals (numpy arrays)
        """
        return [uniform(low=self.boundaries[0], high=self.boundaries[1],
                                  size=(self.dimensions,)) for _ in range(size)]