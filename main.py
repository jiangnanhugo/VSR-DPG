"""Parallelized, single-point launch script to run DSO on a set of benchmarks."""

import time
import multiprocessing
from copy import deepcopy
import click

from scibench.symbolic_data_generator import *
from scibench.symbolic_equation_evaluator_public import Equation_evaluator

from cvdso.grammar.grammar import ContextSensitiveGrammar
from cvdso.grammar.grammar_regress_task import RegressTask
from cvdso.grammar.production_rules import get_production_rules, get_var_i_production_rules
from cvdso.grammar.grammar_program import grammarProgram
from deep_symbolic_optimizer import CVDeepSymbolicOptimizer
from utils import load_config, create_uniform_generations, create_reward_threshold


def train_cvdso(config, config_filename):
    """Trains DSO and returns dict of reward, expressions"""

    # Train the model
    model = CVDeepSymbolicOptimizer(deepcopy(config),  config_filename)
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
@click.option('--optimizer', default=None, type=str, help="optimizer for the expressions")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--n_cores_task', default=1, help="Number of cores to spread out across tasks")
@click.option('--num_per_episodes', default=20, help="Number of iterations per rounds")
def main(config_template, optimizer, equation_name, metric_name, noise_type, noise_scale, n_cores_task, num_per_rounds=30):
    """Runs DSO in parallel across multiple seeds using multiprocessing."""
    config = load_config(config_template)
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name=metric_name)
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()
    function_set = data_query_oracle.get_operators_set()

    num_iterations = create_uniform_generations(num_per_rounds, nvar)
    ContextSensitiveGrammar.program = grammarProgram(nvar, optimizer)
    ContextSensitiveGrammar.program.evalaute_loss = data_query_oracle.compute_metric
    # Overwrite config seed, if specified

    regress_dataset_size = 2048
    ContextSensitiveGrammar.task = RegressTask(regress_dataset_size,
                                               nvar,
                                               dataXgen,
                                               data_query_oracle)
    # get basic production rules
    production_rules = get_production_rules(0, function_set)
    grammars = production_rules
    # print("The production rules are:", production_rules)
    reward_thresh = create_reward_threshold(10, len(num_iterations))
    nt_nodes = ['A']
    eta = 0.999
    max_len = 100
    aug_grammars, aug_nt_nodes = [], []

    # Start training
    # Farm out the work
    for round_idx in range(len(num_iterations)):
        print('++++++++++++ ROUND {}  ++++++++++++'.format(round_idx))
        grammar_model = ContextSensitiveGrammar(base_grammars=grammars,
                                                aug_grammars=aug_grammars,
                                                non_terminal_nodes=nt_nodes,
                                                aug_nt_nodes=aug_nt_nodes,
                                                max_length=max_len,
                                                eta=eta,
                                                hof_size=100)
        grammar_model.task.set_vf(round_idx)
        grammar_model.task.set_allowed_inputs(grammar_model.program.get_vf())
        if round_idx < len(num_iterations):
            grammars += get_var_i_production_rules(round_idx, function_set)
        print("grammars:", grammars)
        print("aug grammars:", aug_grammars)
        print("aug ntn nodes:", aug_nt_nodes)
        if n_cores_task > 1:
            pool = multiprocessing.Pool(n_cores_task)
            for i, (result, used_time) in enumerate(
                    pool.imap_unordered(train_cvdso, config_template)):
                print("cvDSO time {:.0f} s".format(i + 1, used_time))
        else:
            for i, config in enumerate(config):
                result, used_time = train_cvdso(config, config_template)

                print("cvDSO time {:.0f} s".format(i + 1, used_time))


if __name__ == "__main__":
    main()
