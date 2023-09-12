import time
from typing import Union
import numpy as np

from nsma.algorithms.genetic.genetic_utils.general_utils import calc_crowding_distance
from nsma.general_utils.pareto_utils import pareto_efficient
from nsma.problems.problem import Problem

from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from algorithms.gradient_based.local_search.local_search_factory import LocalSearchFactory
from line_searches.line_search_factory import LineSearchFactory
from problems.alamo_problem import AlamoProblem


class FrontAlamo(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 mu_max: float, sigma: float, ro: float, theta_tol: float, theta_dec_factor: float, qth_quantile: float, MOSD_max_iter: int,
                 gurobi_method: int, gurobi_verbose: bool,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):

        ExtendedGradientBasedAlgorithm.__init__(self,
                                                max_iter, max_time, max_f_evals, verbose, verbose_interspace,
                                                plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                                theta_tol,
                                                gurobi_method, gurobi_verbose, ALS_alpha_0, ALS_delta,
                                                ALS_beta, ALS_min_alpha,
                                                name_DDS='Steepest_Descent_DS',
                                                name_ALS='Boundconstrained_Front_ALS')

        self.__tau0 = 1
        self.__mu_max = mu_max
        self.__sigma = sigma
        self.__ro = ro
        self.__theta_dec_factor = theta_dec_factor
        self.__qth_quantile = qth_quantile

        self.__mosd = LocalSearchFactory.get_algorithm('MOSD', {'theta_tol': theta_tol, 'gurobi_method': gurobi_method, 'gurobi_verbose': gurobi_verbose, 'ALS_alpha_0': ALS_alpha_0, 'ALS_delta': ALS_delta, 'ALS_beta': ALS_beta, 'ALS_min_alpha': ALS_min_alpha, 'MOSD_max_iter': MOSD_max_iter, 'max_time': max_time, 'max_f_evals': max_f_evals})
        self.__single_point_line_search = LineSearchFactory.get_line_search('MOALS', ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)

    def search(self, p_list: np.array, f_list: np.array, problem: Problem):
        self.update_stopping_condition_current_value('max_time', time.time())

        n_points, n = p_list.shape
        m = f_list.shape[1]

        alamo_problem = AlamoProblem(problem, np.zeros(problem.n_total_constraints), self.__tau0)

        f_list = np.zeros((n_points, m))
        for p in range(n_points):
            f_list[p, :] = alamo_problem.evaluate_functions(p_list[p, :])

        self.show_figure(p_list, f_list)

        V = np.zeros(problem.n_total_constraints)
        old_V_norm = 0

        crowding_quantile = np.inf

        while not self.evaluate_stopping_conditions():
            self.output_data(f_list, crowding_quantile=crowding_quantile, max_mu=np.max(alamo_problem.mu), tau=alamo_problem.tau)
            self.add_to_stopping_condition_current_value('max_iter', 1)

            efficient_point_idx = pareto_efficient(f_list)
            f_list = f_list[efficient_point_idx, :]
            p_list = p_list[efficient_point_idx, :]

            p_list_prev = np.copy(p_list)
            f_list_prev = np.copy(f_list)

            crowding_list = calc_crowding_distance(f_list_prev)
            is_finite_idx = np.isfinite(crowding_list)

            if len(crowding_list[is_finite_idx]) > 0:
                crowding_quantile = np.quantile(crowding_list[is_finite_idx], self.__qth_quantile)
            else:
                crowding_quantile = np.inf

            sorted_idx = np.flip(np.argsort(crowding_list))

            p_list_prev = p_list_prev[sorted_idx, :]
            f_list_prev = f_list_prev[sorted_idx, :]
            crowding_list = crowding_list[sorted_idx]

            p = 0

            while not self.evaluate_stopping_conditions() and p < len(p_list_prev):

                x_k = p_list_prev[p, :]
                f_k = f_list_prev[p, :]

                J_k = alamo_problem.evaluate_functions_jacobian(x_k)
                self.add_to_stopping_condition_current_value('max_f_evals', n)

                power_set = self.objectives_powerset(m)

                for I_k in power_set:

                    if self.evaluate_stopping_conditions():
                        break

                    if self.exists_dominating_point(f_k, f_list) or crowding_list[p] < crowding_quantile:
                        break
                        
                    d_k, theta_k = self._direction_solver.compute_direction(alamo_problem, J_k[I_k, ])

                    if not self.evaluate_stopping_conditions() and theta_k < self._theta_tol:
                        new_p, new_f, _, f_eval_ls = self._line_search.search(alamo_problem, x_k, f_list, d_k, 0, np.arange(m))
                        self.add_to_stopping_condition_current_value('max_f_evals', f_eval_ls)

                        if not self.evaluate_stopping_conditions() and new_p is not None:

                            self.__mosd.reset_stopping_conditions_current_values(self._theta_tol)
                            new_p, new_f, theta_new_p = self.__mosd.search(new_p, new_f, alamo_problem)
                            self.update_stopping_condition_current_value('max_f_evals', self.__mosd.get_stopping_condition_current_value('max_f_evals'))

                            efficient_point_idx = self.fast_non_dominated_filter(f_list, new_f.reshape((1, m)))

                            p_list = np.concatenate((p_list[efficient_point_idx, :], new_p.reshape((1, n))), axis=0)
                            f_list = np.concatenate((f_list[efficient_point_idx, :], new_f.reshape((1, m))), axis=0)

                d_k, theta_k = self._direction_solver.compute_direction(alamo_problem, J_k)

                if not self.evaluate_stopping_conditions() and theta_k < self._theta_tol:
                    new_p, new_f, _, f_eval_ls = self.__single_point_line_search.search(alamo_problem, x_k, f_k, d_k, theta_k)
                    self.add_to_stopping_condition_current_value('max_f_evals', f_eval_ls)

                    if not self.evaluate_stopping_conditions() and new_p is not None:

                        self.__mosd.reset_stopping_conditions_current_values(self._theta_tol)
                        new_p, new_f, theta_new_p = self.__mosd.search(new_p, new_f, alamo_problem)
                        self.update_stopping_condition_current_value('max_f_evals', self.__mosd.get_stopping_condition_current_value('max_f_evals'))

                        if not self.exists_dominating_point(new_f, f_list):

                            efficient_point_idx = self.fast_non_dominated_filter(f_list, new_f.reshape((1, m)))

                            p_list = np.concatenate((p_list[efficient_point_idx, :], new_p.reshape((1, n))), axis=0)
                            f_list = np.concatenate((f_list[efficient_point_idx, :], new_f.reshape((1, m))), axis=0)

                p += 1

            self._theta_tol *= self.__theta_dec_factor

            n_points = p_list.shape[0]
            C = np.zeros((n_points, len(alamo_problem.mu)))
            increase_tau = False
            for p in range(n_points):
                C[p, :] = problem.evaluate_constraints(p_list[p, :])
                for index_c in range(C.shape[1]):
                    if C[p, index_c] < 0 and alamo_problem.mu[index_c] + alamo_problem.tau * C[p, index_c] > 0:
                        increase_tau = True

            for index_c in range(len(alamo_problem.mu)):
                V[index_c] = min(min([-C[p, index_c] for p in range(n_points)]), alamo_problem.mu[index_c] / alamo_problem.tau)
                alamo_problem.mu[index_c] = max(0, min(self.__mu_max, alamo_problem.mu[index_c] + alamo_problem.tau * max([C[p, index_c] for p in range(n_points)])))

            new_V_norm = np.linalg.norm(V)
            if new_V_norm > self.__sigma * old_V_norm or increase_tau:
                alamo_problem.tau *= self.__ro
            old_V_norm = new_V_norm

            for p in range(n_points):
                f_list[p, :] = alamo_problem.evaluate_functions(p_list[p, :])

            self.show_figure(p_list, f_list)

        self.close_figure()
        self.output_data(f_list, crowding_quantile=crowding_quantile, max_mu=np.max(alamo_problem.mu), tau=alamo_problem.tau)

        return p_list, f_list, time.time() - self.get_stopping_condition_current_value('max_time')

    def update_stopping_condition_current_value(self, condition_name: str, current_value: Union[float, int, np.float64, np.ndarray]):
        ExtendedGradientBasedAlgorithm.update_stopping_condition_current_value(self, condition_name, current_value)
        if condition_name in ['max_time', 'max_f_evals']:
            self.__mosd.update_stopping_condition_current_value(condition_name, current_value)
