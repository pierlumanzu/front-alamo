from algorithms.gradient_based.front_alamo import FrontAlamo


class AlgorithmFactory:

    @staticmethod
    def get_algorithm(algorithm_name: str, **kwargs):

        general_settings = kwargs['general_settings']
        algorithms_settings = kwargs['algorithms_settings']

        if algorithm_name == 'FRONT-ALAMO':
            FALAMO_settings = algorithms_settings[algorithm_name]

            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']

            return FrontAlamo(general_settings['max_iter'],
                              general_settings['max_time'],
                              general_settings['max_f_evals'],
                              general_settings['verbose'],
                              general_settings['verbose_interspace'],
                              general_settings['plot_pareto_front'],
                              general_settings['plot_pareto_solutions'],
                              general_settings['plot_dpi'],
                              FALAMO_settings['mu_max'],
                              FALAMO_settings['sigma'],
                              FALAMO_settings['ro'],
                              FALAMO_settings['theta_tol'],
                              FALAMO_settings['theta_dec_factor'],
                              FALAMO_settings['qth_quantile'],
                              FALAMO_settings['MOSD_max_iter'],
                              DDS_settings['gurobi_method'],
                              DDS_settings['gurobi_verbose'],
                              ALS_settings['alpha_0'],
                              ALS_settings['delta'],
                              ALS_settings['beta'],
                              ALS_settings['min_alpha'])

        else:
            raise NotImplementedError
