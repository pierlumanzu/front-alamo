import tensorflow as tf

from problems.lap.lap_class import LAP

'''
For more details about the LAP problems, the user is referred to 

Cocchi, G., Lapucci, M. 
An augmented Lagrangian algorithm for multi-objective optimization. 
Comput Optim Appl 77, 29–56 (2020). 
https://doi.org/10.1007/s10589-020-00204-z
'''


class LAP_1(LAP):

    def __init__(self, n: int):
        assert n == 2

        LAP.__init__(self, n)

        self.objectives = [
            tf.exp(0.5 * (-self._z[0] - self._z[1])) + 0.25 * self._z[0] ** 2 - 1 / 3 * self._z[0] * self._z[1],
            self._z[0] ** 2 + self._z[1] ** 2 - 2 * self._z[0] + 4 * self._z[1] + 6 * self._z[0] * self._z[1]
        ]

        self.general_constraints = [
            self._z[0] ** 2 + self._z[1] ** 2 - 1,
            self._z[1] - 2 * self._z[0] - 1,
            1 / 3 * self._z[0] ** 2 - self._z[1] - 1 / 3
        ]

    @staticmethod
    def name():
        return 'LAP_1'

    @staticmethod
    def family_name():
        return 'LAP_1'


class LAP_2(LAP):

    def __init__(self, n: int):
        assert n >= 1
        LAP.__init__(self, n)

        self.objectives = [
            tf.reduce_sum([(i + 1) * self._z[i] for i in range(n)]),
            tf.reduce_sum([0.1 * tf.exp(-self._z[i]) for i in range(n)])
        ]

        self.general_constraints = [
            tf.reduce_sum([(i + 1) * self._z[i] ** 4 for i in range(n)]) - self.n ** 2
        ]

    @staticmethod
    def name():
        return 'LAP_2'
