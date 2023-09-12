import numpy as np

from nsma.problems.problem import Problem


class AlamoProblem(Problem):

    def __init__(self, problem: Problem, mu_0: np.array, tau_0: float):
        Problem.__init__(self, problem.n)

        self.__problem = problem

        self.__mu = mu_0
        self.__tau = tau_0

    def evaluate_functions(self, x: np.array):
        F = self.__problem.evaluate_functions(x)
        C = self.__problem.evaluate_constraints(x)

        return np.array([f + self.__tau * sum([max(0, c + mu / self.__tau)**2 for (c, mu) in zip(C, self.__mu)]) for f in F])

    def evaluate_functions_jacobian(self, x: np.array):
        jacobian_f = self.__problem.evaluate_functions_jacobian(x)
        jacobian_c = self.__problem.evaluate_constraints_jacobian(x)

        C = self.__problem.evaluate_constraints(x)
        M = [max(0, c + mu / self.__tau) for (c, mu) in zip(C, self.__mu)]

        return jacobian_f + self.__tau * sum(np.dot(np.diag(M), jacobian_c))

    def evaluate_constraints(self, x: np.array):
        return np.empty(0)

    def evaluate_constraints_jacobian(self, x: np.array):
        return np.empty(0)

    def check_point_feasibility(self, x: np.array):
        return True

    @Problem.objectives.setter
    def objectives(self, objectives: list):
        raise RuntimeError

    @Problem.general_constraints.setter
    def general_constraints(self, general_constraints: list):
        raise RuntimeError

    @Problem.lb.setter
    def lb(self, lb: np.array):
        raise RuntimeError

    @Problem.ub.setter
    def ub(self, ub: np.array):
        raise RuntimeError

    @property
    def n(self):
        return self.__problem.n

    @property
    def m(self):
        return self.__problem.m

    @staticmethod
    def name():
        return "Alamo Problem"

    @staticmethod
    def family_name():
        return "Alamo Problem"

    @property
    def mu(self):
        return self.__mu

    @property
    def tau(self):
        return self.__tau

    @tau.setter
    def tau(self, value: float):
        self.__tau = value
