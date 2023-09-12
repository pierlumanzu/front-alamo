import numpy as np

from nsma.general_utils.pareto_utils import pareto_efficient
from nsma.problems.problem import Problem


def points_postprocessing(p_list: np.array, f_list: np.array, problem: Problem):
    assert len(p_list) == len(f_list)
    n_points, _ = p_list.shape

    for p in range(n_points):
        f_list[p, :] = problem.evaluate_functions(p_list[p, :])

    feasible = [True] * n_points
    infeasible_points = 0
    for p in range(n_points):
        constraints = problem.evaluate_constraints(p_list[p, :])
        if not np.linalg.norm(constraints[constraints > 0]) < 0.5e-3:
            feasible[p] = False
            infeasible_points += 1
    if infeasible_points > 0:
        print('Warning: found {} infeasible points'.format(infeasible_points))

    p_list = p_list[feasible, :]
    f_list = f_list[feasible, :]

    efficient_points_idx = pareto_efficient(f_list)
    p_list = p_list[efficient_points_idx, :]
    f_list = f_list[efficient_points_idx, :]

    print('Result: found {} non-dominated points'.format(len(p_list)))
    print()

    return p_list, f_list
