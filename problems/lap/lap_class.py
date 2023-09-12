from abc import ABC

import numpy as np
from nsma.problems.problem import Problem

'''
For more details about the LAP problems, the user is referred to 

Cocchi, G., Lapucci, M. 
An augmented Lagrangian algorithm for multi-objective optimization. 
Comput Optim Appl 77, 29–56 (2020). 
https://doi.org/10.1007/s10589-020-00204-z
'''


class LAP(Problem, ABC):

    def __init__(self, n: int):
        Problem.__init__(self, n)

    def generate_feasible_random_point(self):
        return np.array([[0] * self.n], dtype=float)

    @staticmethod
    def family_name():
        return 'LAP'