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
from deep_symbolic_optimizer import VSRDeepSymbolicRegression
from utils import load_config, create_uniform_generations, create_reward_threshold

averaged_var_y = 10
threshold_values = {
    'neg_mse': {'expr_consts_thres': 1e-6, 'expr_obj_thres': 0.01},
    'neg_nmse': {'expr_consts_thres': 1e-6, 'expr_obj_thres': 0.01 / averaged_var_y},
    'neg_nrmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': np.sqrt(0.01 / averaged_var_y)},
    'neg_rmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': 0.1},
    'inv_mse': {'expr_consts_thres': -1 / (1 + 1e-6), 'expr_obj_thres': -1 / (1 + 0.01)},
    'inv_nmse': {'expr_consts_thres': -1 / (1 + 1e-6), 'expr_obj_thres': -1 / (1 + 0.01 / averaged_var_y)},
    'inv_nrmse': {'expr_consts_thres': -1 / (1 + 1e-6), 'expr_obj_thres': -1 / (1 + np.sqrt(0.01 / averaged_var_y))},
}


def train_cvdso(config, config_filename, grammar_model):
    """Trains DSO and returns dict of reward, expressions"""

    # Train the model
    model = VSRDeepSymbolicRegression(deepcopy(config), config_filename, grammar_model)
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
@click.option('--optimizer', default='BFGS', type=str, help="optimizer for the expressions")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--max_len', default=10, help="max length of the sequence from the decoder")
@click.option('--num_per_rounds', default=20, help="Number of iterations per rounds")
def main(config_template, optimizer, equation_name, metric_name, noise_type, noise_scale, max_len, num_per_rounds):
    """Runs DSO in parallel across multiple seeds using multiprocessing."""
    config = load_config(config_template)
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name=metric_name)
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()
    function_set = data_query_oracle.get_operators_set()

    num_iterations = create_uniform_generations(num_per_rounds, nvar)
    program = grammarProgram(optimizer=optimizer)
    program.evaluate_loss = data_query_oracle.compute_metric

    regress_dataset_size = 2048
    task = RegressTask(regress_dataset_size,
                       nvar,
                       dataXgen,
                       data_query_oracle)

    # get basic production rules
    production_rules = get_production_rules(0, function_set)
    # print("The production rules are:", production_rules)
    reward_thresh = create_reward_threshold(10, len(num_iterations))
    nt_nodes = ['A']
    eta = 0.999
    start_symbols = ['A']
    # Start training
    stand_alone_constants = []
    # Farm out the work
    for round_idx in range(len(num_iterations)):
        print('++++++++++++ ROUND {}  ++++++++++++'.format(round_idx))

        if round_idx < len(num_iterations):
            production_rules += get_var_i_production_rules(round_idx, function_set)
        print("grammars:", production_rules)
        print("start_symbols:", start_symbols)
        grammar_model = ContextSensitiveGrammar(
            nvars=nvar,
            production_rules=production_rules,
            start_symbols=start_symbols[0],
            non_terminal_nodes=nt_nodes,
            max_length=max_len,
            eta=eta,
            hof_size=100,
            reward_threhold=reward_thresh[round_idx]
        )
        grammar_model.expr_obj_thres = threshold_values[metric_name]['expr_obj_thres']
        grammar_model.task = task
        grammar_model.program = program
        grammar_model.task.set_vf(round_idx)
        grammar_model.task.set_allowed_inputs(grammar_model.task.get_vf())
        best_expressions, used_time = train_cvdso(config, config_template, grammar_model)

        # print("cvDSO time {:.0f} s".format(used_time))
        # print(f"best expression is:", best_expressions[-1])
        # from cvdso.grammar.grammar_program import SymbolicExpression
        # eq = SymbolicExpression(
        #     'f->A;A->(A+A);A->C*X0;A->C;A->C;A->(A-A);A->C;A->(A-A);A->(A+A);A->C;A->C;A->(A+A);A->(A-A);A->C;A->(A+A);A->(A+A);A->C*sin(X0);A->(A+A);A->A*A;A->C*cos(X0);A->(A-A)')
        # eq.reward = 0.9999999395508058
        # eq.expr_template = '(C*X0+C)'
        # eq.fitted_eq = "0.07876558963752647*X0 - 0.5078745192475905"
        # best_expressions = [eq]
        if round_idx < len(num_iterations) - 1:
            # the last round does not need freeze
            start_symbols, _, stand_alone_constants = grammar_model.freeze_equations(best_expressions,
                                                                                     stand_alone_constants,
                                                                                     round_idx + 1)

            print("The discovered expression template (with control variable {})".format(grammar_model.task.vf))
            print(start_symbols)

            production_rules = [gi for gi in production_rules if str(round_idx) not in gi]


if __name__ == "__main__":
    main()
