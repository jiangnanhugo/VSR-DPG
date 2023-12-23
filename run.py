"""Parallelized, single-point launch script to run DSO on a set of benchmarks."""

import os
import sys
import time
import multiprocessing
from copy import deepcopy
from datetime import datetime
import random
import click
from cvdso.grammar.production_rules import get_production_rules
from deep_symbolic_optimizer import CVDeepSymbolicOptimizer
from cvdso.logeval import LogEval
from cvdso.utils import load_config
from cvdso.utils import safe_update_summary

from scibench.symbolic_data_generator import *
from scibench.symbolic_equation_evaluator_public import Equation_evaluator


def train_cvdso(config, function_set, dataX, data_query_oracle, config_filename):
    """Trains DSO and returns dict of reward, expression, and traversal"""

    print("\n== TRAINING SEED {} START ============".format(config["experiment"]["seed"]))

    # Train the model
    model = CVDeepSymbolicOptimizer(deepcopy(config), function_set, dataX, data_query_oracle, config_filename)
    start = time.time()
    # Setup the model
    model.setup()
    result = model.train()
    result["t"] = time.time() - start
    result.pop("program")

    save_path = model.config_experiment["save_path"]
    summary_path = os.path.join(save_path, "summary.csv")

    print("== TRAINING SEED {} END ==============".format(config["experiment"]["seed"]))

    return result, summary_path


def print_summary(config, runs, messages):
    text = '\n== EXPERIMENT SETUP START ===========\n'
    text += 'Starting seed        : {}\n'.format(config["experiment"]["seed"])
    text += 'Runs                 : {}\n'.format(runs)
    if len(messages) > 0:
        text += 'Additional context   :\n'
        for message in messages:
            text += "      {}\n".format(message)
    text += '== EXPERIMENT SETUP END ============='
    print(text)


@click.command()
@click.argument('config_template', default="")
@click.option('--equation_name', '--e', default=None, type=str, help="Name of equation")
@click.option('--noise_type', '--nt', default='normal', type=str, help="")
@click.option('--noise_scale', '--ns', default=0.0, type=float, help="")
@click.option('--runs', '--r', default=1, type=int, help="Number of independent runs with different seeds")
@click.option('--n_cores_task', '--n', default=1, help="Number of cores to spread out across tasks")
@click.option('--logdir', '--l', default="log", type=str, help="logdir folder")
def main(config_template, equation_name, noise_type, noise_scale, runs, n_cores_task, logdir):
    """Runs DSO in parallel across multiple seeds using multiprocessing."""

    messages = []

    # Load the experiment config
    config_template = config_template if config_template != "" else None
    config = load_config(config_template)
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name='inv_nrmse')
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()
    function_set= data_query_oracle.operators_set

    operators_set = data_query_oracle.get_operators_set()

    production_rules = get_production_rules(0, operators_set)

    # Overwrite config seed, if specified
    seed = int(time.perf_counter() * 10000) % 1000007
    random.seed(seed)
    print('random seed=', seed)
    seed = int(time.perf_counter() * 10000) % 1000007
    np.random.seed(seed)
    config["experiment"]["seed"] = seed
    config["experiment"]["logdir"] = logdir
    config["experiment"]["cmd"] = " ".join(sys.argv)

    # Set timestamp once to be used by all workers
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    config["experiment"]["timestamp"] = timestamp

    # Fix incompatible configurations
    if n_cores_task == -1:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task > runs:
        messages.append(
            "INFO: Setting 'n_cores_task' to {} because there are only {} runs.".format(
                runs, runs))
        n_cores_task = runs
    if config["training"]["verbose"] and n_cores_task > 1:
        messages.append(
            "INFO: Setting 'verbose' to False for parallelized run.")
        config["training"]["verbose"] = False
    if config["training"]["n_cores_batch"] != 1 and n_cores_task > 1:
        messages.append(
            "INFO: Setting 'n_cores_batch' to 1 to avoid nested child processes.")
        config["training"]["n_cores_batch"] = 1

    # Start training
    print_summary(config, runs, messages)

    # Generate configs (with incremented seeds) for each run
    configs = [deepcopy(config) for _ in range(runs)]
    for i, config in enumerate(configs):
        config["experiment"]["seed"] += i

    # Farm out the work
    if n_cores_task > 1:
        pool = multiprocessing.Pool(n_cores_task)
        for i, (result, summary_path) in enumerate(
                pool.imap_unordered(train_cvdso, configs, function_set, dataXgen, data_query_oracle, config_template)):
            if not safe_update_summary(summary_path, result):
                print("Warning: Could not update summary stats at {}".format(summary_path))
            print("INFO: Completed run {} of {} in {:.0f} s".format(i + 1, runs, result["t"]))
    else:
        for i, config in enumerate(configs):
            result, summary_path = train_cvdso(config, function_set, dataXgen, data_query_oracle, config_template)
            if not safe_update_summary(summary_path, result):
                print("Warning: Could not update summary stats at {}".format(summary_path))
            print("INFO: Completed run {} of {} in {:.0f} s".format(i + 1, runs, result["t"]))

    # Evaluate the log files
    print("\n== POST-PROCESS START =================")
    log = LogEval(config_path=os.path.dirname(summary_path))
    log.analyze_log(
        show_count=config["postprocess"]["show_count"],
        show_hof=config["training"]["hof"] is not None and config["training"]["hof"] > 0,
        show_pf=config["training"]["save_pareto_front"],
        save_plots=config["postprocess"]["save_plots"])
    print("== POST-PROCESS END ===================")


if __name__ == "__main__":
    main()
