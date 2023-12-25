"""Parallelized, single-point launch script to run DSO on a set of benchmarks."""


import time
import multiprocessing
from copy import deepcopy
import click
from cvdso.grammar.production_rules import get_production_rules

from deep_symbolic_optimizer import CVDeepSymbolicOptimizer
from utils import load_config, create_uniform_generations

from scibench.symbolic_data_generator import *
from scibench.symbolic_equation_evaluator_public import Equation_evaluator


def train_cvdso(config, function_set, dataX, data_query_oracle, config_filename):
    """Trains DSO and returns dict of reward, expressions"""

    # Train the model
    model = CVDeepSymbolicOptimizer(deepcopy(config), function_set, dataX, data_query_oracle, config_filename)
    start = time.time()
    print("training start.....")
    # Setup the model
    model.setup()
    result = model.train()
    used_time = time.time() - start
    return result, used_time


@click.command()
@click.argument('config_template', default="")
@click.option('--equation_name', default=None, type=str, help="Name of equation")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--n_cores_task', default=1, help="Number of cores to spread out across tasks")
@click.option('--num_per_episodes', default=20, help="Number of iterations per rounds")
def main(config_template, equation_name, metric_name, noise_type, noise_scale, n_cores_task, num_per_rounds=30):
    """Runs DSO in parallel across multiple seeds using multiprocessing."""
    config = load_config(config_template)
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name=metric_name)
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()
    function_set = data_query_oracle.operators_set

    production_rules = get_production_rules(0, function_set)
    num_iterations = create_uniform_generations(num_per_rounds, nvar)

    # Overwrite config seed, if specified



    # Start training
    # Farm out the work
    for round_idx in range(len(num_iterations)):
        print('++++++++++++ ROUND {}  ++++++++++++'.format(round_idx))
        program.set_vf(round_idx)
        task.set_allowed_inputs(program.get_vf())
        if round_idx < len(num_iterations):
            grammars += get_var_i_production_rules(round_idx, function_set)
        print("grammars:", grammars)
        print("aug grammars:", aug_grammars)
        print("aug ntn nodes:", aug_nt_nodes)
        print("num_rollouts:", num_rollouts)
    if n_cores_task > 1:
        pool = multiprocessing.Pool(n_cores_task)
        for i, (result, used_time) in enumerate(
                pool.imap_unordered(train_cvdso, config, function_set, dataXgen, data_query_oracle, config_template)):
            print("cvDSO time {:.0f} s".format(i + 1, used_time))
    else:
        for i, config in enumerate(config):
            result, used_time = train_cvdso(config, function_set, dataXgen, data_query_oracle, config_template)

            print("cvDSO time {:.0f} s".format(i + 1, used_time))


if __name__ == "__main__":
    main()
