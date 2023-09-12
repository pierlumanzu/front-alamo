import argparse
import sys


def get_args():

    parser = argparse.ArgumentParser(description='algorithms for Multi-Objective optimization')

    parser.add_argument('--algs', type=str, help='algorithms', nargs='+', choices=['FRONT-ALAMO'])

    parser.add_argument('--probs', help='problems to evaluate', nargs='+', choices=['LAP'])

    parser.add_argument('--multiple_points', help='If activated (not recommended unless the problem has box constraints only: it may require some time), the algorithm starts with n initial feasible points, with n being the problem dimension. If deactivated, a single initial feasible point is used', action='store_true', default=False)

    parser.add_argument('--max_iter', help='Maximum number of iterations', default=None, type=int)

    parser.add_argument('--max_time', help='Maximum number of elapsed minutes per problem', default=None, type=float)

    parser.add_argument('--max_f_evals', help='Maximum number of function evaluations', default=None, type=int)

    parser.add_argument('--verbose', help='Verbose during the iterations', action='store_true', default=False)

    parser.add_argument('--verbose_interspace', help='Used interspace in the verbose (Requirements: verbose activated)', default=20, type=int)

    parser.add_argument('--plot_pareto_front', help='Plot Pareto front', action='store_true', default=False)

    parser.add_argument('--plot_pareto_solutions', help='Plot Pareto solutions (Requirements: plot_pareto_front activated; n in [2, 3])', action='store_true', default=False)

    parser.add_argument('--general_export', help='Export fronts (including plots), execution times and arguments files', action='store_true', default=False)

    parser.add_argument('--export_pareto_solutions', help='Export pareto solutions, including the plots if n in [2, 3] (Requirements: general_export activated)', action='store_true', default=False)

    parser.add_argument('--plot_dpi', help='DPI of the saved plots (Requirements: general_export activated)', default=100, type=int)

    ####################################################
    ### ONLY FOR Gurobi ###
    ####################################################

    parser.add_argument('--gurobi_method', help='Gurobi parameter -- Used method', default=1, type=int)

    parser.add_argument('--gurobi_verbose', help='Gurobi parameter -- Verbose during the Gurobi iterations', action='store_true', default=False)

    ####################################################
    ### ONLY FOR FRONT-ALAMO ###
    ####################################################

    parser.add_argument('--FALAMO_mu_max', help='FRONT-ALAMO parameter -- Maximum value for the Lagrange multipliers', default=10000, type=int)

    parser.add_argument('--FALAMO_sigma', help='FRONT-ALAMO parameter -- Sigma', default=0.9, type=float)

    parser.add_argument('--FALAMO_ro', help='FRONT-ALAMO parameter -- Ro', default=2, type=float)

    parser.add_argument('--FALAMO_theta_tol', help='FRONT-ALAMO parameter -- Theta tol', default=-1.0e-5, type=float)

    parser.add_argument('--FALAMO_theta_dec_factor', help='FRONT-ALAMO parameter -- Theta decreasing factor', default=0.95, type=float)

    parser.add_argument('--FALAMO_qth_quantile', help='FRONT-ALAMO parameter -- q-th quantile', default=0.75, type=float)

    parser.add_argument('--FALAMO_MOSD_max_iter', help='FRONT-ALAMO parameter -- Number of maximum iterations for MOSD', default=1, type=int)

    ####################################################
    ### ONLY FOR ArmijoTypeLineSearch ###
    ####################################################

    parser.add_argument('--ALS_alpha_0', help='ALS parameter -- Initial step size', default=1, type=float)

    parser.add_argument('--ALS_delta', help='ALS parameter -- Coefficient for step size contraction', default=0.5, type=float)

    parser.add_argument('--ALS_beta', help='ALS parameter -- Beta', default=1.0e-4, type=float)

    parser.add_argument('--ALS_min_alpha', help='ALS parameter -- Min alpha', default=1.0e-7, type=float)

    return parser.parse_args(sys.argv[1:])

