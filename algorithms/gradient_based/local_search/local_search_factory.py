from algorithms.gradient_based.local_search.mosd import MOSD


class LocalSearchFactory:
    @staticmethod
    def get_algorithm(algorithm_name: str, args_algorithm: dict):
            
        if algorithm_name == 'MOSD':

            local_search_algorithm = MOSD(args_algorithm['theta_tol'],
                                          args_algorithm['gurobi_method'],
                                          args_algorithm['gurobi_verbose'],
                                          args_algorithm['ALS_alpha_0'],
                                          args_algorithm['ALS_delta'],
                                          args_algorithm['ALS_beta'],
                                          args_algorithm['ALS_min_alpha'],
                                          args_algorithm['MOSD_max_iter'] if args_algorithm['MOSD_max_iter'] is not None else None,
                                          args_algorithm['max_time'] if args_algorithm['max_time'] is not None else None,
                                          args_algorithm['max_f_evals'] if args_algorithm['max_f_evals'] is not None else None)

        else:
            raise NotImplementedError

        return local_search_algorithm
