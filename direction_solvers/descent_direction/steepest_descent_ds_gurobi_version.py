import numpy as np
from gurobipy import Model, GRB

from nsma.direction_solvers.descent_direction.dds import DDS
from nsma.direction_solvers.gurobi_settings import GurobiSettings

from nsma.problems.problem import Problem


class SteepestDescentDSGurobiVersion(DDS, GurobiSettings):

    def __init__(self, gurobi_method: int, gurobi_verbose: bool):
        DDS.__init__(self)
        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)

    def compute_direction(self, problem: Problem, Jac: np.array, x_p: np.array= None):
        assert x_p is None

        m, n = Jac.shape

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(n), 0

        try:
            model = Model("Steepest Descent Direction")
            model.setParam("OutputFlag", self._gurobi_verbose)
            model.setParam("Method", self._gurobi_method)

            d = model.addMVar(n, lb=-np.inf, ub=np.inf, name="d")
            beta = model.addMVar(1, lb=-np.inf, ub=0., name="beta")

            obj = beta + 1/2 * (d @ d)
            model.setObjective(obj)

            for j in range(m):
                model.addConstr(Jac[j, :] @ d <= beta, name='Jacobian constraint n°{}'.format(j))

            model.update()

            for i in range(n):
                d[i].start = 0.
            beta.start = 0.

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = model.getVars()

                d_p = np.array([s.x for s in sol][:n])
                theta_p = model.getObjective().getValue()

            else:
                return np.zeros(n), 0

        except AttributeError:
            return np.zeros(n), 0

        return d_p, theta_p
