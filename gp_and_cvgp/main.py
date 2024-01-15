import os.path

from library import Library
import argparse
from program import Program
import regress_task
from const import ScipyMinimize
from scibench.symbolic_data_generator import *
from scibench.symbolic_equation_evaluator_public import Equation_evaluator
from functions import create_tokens
from cvgp import ExpandingGeneticProgram
from gp import GeneticProgram
from genetic_operations import GPHelper

import numpy as np
import random
import time

averaged_var_y = 10  # 569

config = {
    'neg_mse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': 0.01},
    'neg_nmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': 0.01 / averaged_var_y},
    'neg_nrmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': np.sqrt(0.01 / averaged_var_y)},
    'neg_rmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': 0.1},
    'inv_mse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': -1 / (1 + 0.01)},
    'inv_nmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': -1 / (1 + 0.01 / averaged_var_y)},
    'inv_nrmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': -1 / (1 + np.sqrt(0.01 / averaged_var_y))},
}


def run_VSR_GP(equation_name, metric_name, noise_type, noise_scale, optimizer, memray_output_bin, track_memory=False):
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name)
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()

    regress_batchsize = 256
    opt_num_expr = 5

    expr_obj_thres = 1e-6  # data_query_oracle.expr_obj_thres
    expr_consts_thres = config[metric_name]['expr_consts_thres']

    # gp parameters
    cxpb = 0.8
    mutpb = 0.8
    maxdepth = 2
    tour_size = 3
    hof_size = 50  # 0

    population_size = 100
    n_generations = 20

    # get all the functions and variables ready
    all_tokens = create_tokens(nvar, data_query_oracle.operators_set, protected=True)
    protected_library = Library(all_tokens)

    protected_library.print_library()

    # get program ready
    Program.library = protected_library
    Program.opt_num_expr = opt_num_expr
    Program.expr_obj_thres = expr_obj_thres
    Program.expr_consts_thres = expr_consts_thres

    Program.set_execute(True)  # protected = True

    # set const_optimizer
    Program.optimizer = optimizer
    Program.const_optimizer = ScipyMinimize()
    Program.noise_std = noise_scale

    # set the task
    allowed_input_tokens = np.zeros(nvar, dtype=np.int32)  # set it for now. Will change in gp.run
    Program.task = regress_task.RegressTaskV1(regress_batchsize,
                                              allowed_input_tokens,
                                              dataXgen,
                                              data_query_oracle)

    # set gp helper
    gp_helper = GPHelper()
    gp_helper.library = protected_library

    # set GP
    ExpandingGeneticProgram.library = protected_library
    ExpandingGeneticProgram.gp_helper = gp_helper
    vsr_gp = ExpandingGeneticProgram(cxpb, mutpb, maxdepth, population_size,
                                               tour_size, hof_size, n_generations, nvar)

    # run GP
    if track_memory:
        import memray
        if os.path.isfile(memray_output_bin):
            os.remove(memray_output_bin)
        with memray.Tracker(memray_output_bin):
            start = time.time()
            vsr_gp.run()
            end_time = time.time() - start

    else:
        start = time.time()
        vsr_gp.run()
        end_time = time.time() - start
    # print
    print('final hof=')
    vsr_gp.print_hof()
    print("VSR_GP {} mins".format(np.round(end_time / 60, 3)))


def run_GP(equation_name, metric_name, noise_type, noise_scale, optimizer, memray_output_bin, track_memory=False):
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name)
    temp = data_query_oracle.get_vars_range_and_types()
    dataXgen = DataX(temp)
    nvar = data_query_oracle.get_nvars()

    regress_batchsize = 256
    opt_num_expr = 1  # currently do not need to re-run the experiments multiple times.

    # gp parameters
    cxpb = 0.8
    mutpb = 0.8
    maxdepth = 2
    population_size = 100  # 00
    tour_size = 3
    hof_size = 50
    n_generations = 20* nvar  # 00

    # get all the functions and variables ready
    all_tokens = create_tokens(nvar, data_query_oracle.operators_set, protected=True)
    protected_library = Library(all_tokens)

    protected_library.print_library()

    # everything is allowed.
    allowed_input_tokens = np.ones(nvar, dtype=np.int32)
    protected_library.set_allowed_input_tokens(allowed_input_tokens)

    # get program ready
    Program.library = protected_library
    Program.opt_num_expr = opt_num_expr
    Program.set_execute(True)  # protected = True

    # set const_optimizer
    Program.optimizer = optimizer
    Program.const_optimizer = ScipyMinimize()
    Program.noise_std = noise_scale

    # set the task
    Program.task = regress_task.RegressTaskV1(regress_batchsize,
                                              allowed_input_tokens,
                                              dataXgen,
                                              data_query_oracle)

    # set gp helper
    gp_helper = GPHelper()
    gp_helper.library = protected_library

    # set GP
    GeneticProgram.library = protected_library
    GeneticProgram.gp_helper = gp_helper
    gp = GeneticProgram(cxpb, mutpb, maxdepth, population_size, tour_size, hof_size, n_generations)

    # run GP
    if track_memory:
        import memray
        if os.path.isfile(memray_output_bin):
            os.remove(memray_output_bin)
        with memray.Tracker(memray_output_bin):
            start = time.time()
            gp.run()
            end_time = time.time() - start

    else:
        start = time.time()
        gp.run(verbose=False)
        end_time = time.time() - start
    # print
    print('final hof=')
    gp.print_hof()
    print("GP {} mins".format(np.round(end_time / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation_name", help="the filename of the true program.")
    parser.add_argument("--metric_name", type=str, help="The name of the metric for loss.")
    parser.add_argument("--noise_type", type=str, help="The name of the noises.")
    parser.add_argument('--optimizer',
                        nargs='?',
                        choices=['BFGS', 'L-BFGS-B', 'Nelder-Mead', 'CG', 'basinhopping', 'dual_annealing', 'shgo', 'direct'],
                        help='list servers, storage, or both (default: %(default)s)')
    parser.add_argument("--expr_obj_thres", type=float, default=1e-6, help="Threshold")
    parser.add_argument("--noise_scale", type=float, default=0.0, help="This parameter adds the standard deviation of the noise")
    parser.add_argument("--memray_output_bin", type=str, help="memory profile")
    parser.add_argument("--track_memory", action="store_true",
                        help="whether run memery track evaluation.")
    parser.add_argument("--cvgp", action="store_true", help="whether run normal gp (expand_gp=False) or expand_gp.")

    args = parser.parse_args()

    seed = int(time.perf_counter() * 10000) % 1000007
    random.seed(seed)
    print('random seed=', seed)

    seed = int(time.perf_counter() * 10000) % 1000007
    np.random.seed(seed)
    print('np.random seed=', seed)

    if args.cvgp:
        run_VSR_GP(args.equation_name, args.metric_name, args.noise_type, args.noise_scale, args.optimizer, args.memray_output_bin,
                   args.track_memory)
    else:
        run_GP(args.equation_name, args.metric_name, args.noise_type, args.noise_scale, args.optimizer, args.memray_output_bin,
               args.track_memory)
