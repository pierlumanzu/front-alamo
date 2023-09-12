import numpy as np

from nsma.problems.problem import Problem

from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm


class MOSD(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 theta_tol: float,
                 gurobi_method: int, gurobi_verbose: bool,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float,
                 max_iter: int = None, max_time: float = None, max_f_evals: int = None):

        ExtendedGradientBasedAlgorithm.__init__(self,
                                                max_iter, max_time, max_f_evals, False, 0, False, False, 0,
                                                theta_tol,
                                                gurobi_method, gurobi_verbose, ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha,
                                                name_DDS='Steepest_Descent_DS', name_ALS='MOALS')

        self.__current_theta = -np.inf
        ExtendedGradientBasedAlgorithm.add_stopping_condition(self, 'theta_tolerance', theta_tol, self.__current_theta, equal_required=True)

        self.__current_alpha = 1.
        ExtendedGradientBasedAlgorithm.add_stopping_condition(self, 'min_alpha', 0, self.__current_alpha, smaller_value_required=True, equal_required=True)

    def search(self, x: np.array, f: np.array, problem: Problem):

        n, m = len(x), len(f)

        while not self.evaluate_stopping_conditions():

            J = problem.evaluate_functions_jacobian(x)
            self.add_to_stopping_condition_current_value('max_f_evals', n)

            if self.evaluate_stopping_conditions():
                break

            v, theta = self._direction_solver.compute_direction(problem, J)
            self.__current_theta = theta
            self.update_stopping_condition_current_value('theta_tolerance', self.__current_theta)

            if theta < self._theta_tol:

                new_x, new_f, alpha, f_eval_ls = self._line_search.search(problem, x, f, v, theta)
                self.add_to_stopping_condition_current_value('max_f_evals', f_eval_ls)

                self.__current_alpha = alpha
                self.update_stopping_condition_current_value('min_alpha', self.__current_alpha)

                if new_x is not None:
                    x = np.copy(new_x)
                    f = np.copy(new_f)

                    self.__current_theta = -np.inf
                    self.update_stopping_condition_current_value('theta_tolerance', self.__current_theta)

                    self.__current_alpha = 1.
                    self.update_stopping_condition_current_value('min_alpha', self.__current_alpha)

            self.add_to_stopping_condition_current_value('max_iter', 1)

        return x, f, self.__current_theta

    def reset_stopping_conditions_current_values(self, theta_tol: float):

        self.update_stopping_condition_current_value('max_iter', 0)

        self._theta_tol = theta_tol
        self.__current_theta = -np.inf
        self.update_stopping_condition_current_value('theta_tolerance', self.__current_theta)
        self.update_stopping_condition_reference_value('theta_tolerance', theta_tol)

        self.__current_alpha = 1.
        self.update_stopping_condition_current_value('min_alpha', self.__current_alpha)
