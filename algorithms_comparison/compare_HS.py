from pyharmonysearch import ObjectiveFunctionInterface, harmony_search
from math import pow
import random


class ObjectiveFunction(ObjectiveFunctionInterface):

    """
        This is a toy objective function that contains only continuous variables. A random seed is used, so the same result
        will be generated every time.

        Goal:

            maximize -(x^2 + (y+1)^2) + 4
            The maximum is 4 at (0, -1).

        Note that since all variables are continuous, we don't actually need to implement get_index() and get_num_discrete_values().

        Warning: Stochastically solving a linear system is dumb. This is just a toy example.
    """

    def __init__(self):
        self._lower_bounds = [-1000, -1000]
        self._upper_bounds = [1000, 1000]
        self._variable = [True, True]

        # define all input parameters
        self._maximize = True  # do we maximize or minimize?
        self._max_imp = 50000  # maximum number of improvisations
        self._hms = 100  # harmony memory size
        self._hmcr = 0.75  # harmony memory considering rate
        self._par = 0.5  # pitch adjusting rate
        self._mpap = 0.25  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
        self._mpai = 2  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only
        self._random_seed = 8675309  # optional random seed for reproducible results

    def get_fitness(self, vector):
        """
            maximize -(x^2 + (y+1)^2) + 4
            The maximum is 4 at (0, -1).
        """
        return -(pow(vector[0], 2) + pow(vector[1] + 1, 2)) + 4

    def get_value(self, i, index=None):
        """
            Values are returned uniformly at random in their entire range. Since both parameters are continuous, index can be ignored.
        """
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

    def get_lower_bound(self, i):
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        return self._upper_bounds[i]

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        # all variables are continuous
        return False

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return hasattr(self, '_random_seed') and self._random_seed

    def get_random_seed(self):
        return self._random_seed

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def maximize(self):
        return self._maximize

if __name__ == '__main__':
    obj_fun = ObjectiveFunction()
    num_processes = 1
    num_iterations = 1  # because random_seed is defined, there's no point in running this multiple times
    initial_harmonies = [[random.randint(-1000, 1000) for _ in range(2)] for _ in range(100)]
    results = harmony_search(obj_fun, num_processes, num_iterations, initial_harmonies=initial_harmonies)
    print('Elapsed time: {}\nBest harmony: {}\nBest fitness: {}'.format(results.elapsed_time, results.best_harmony, results.best_fitness))
