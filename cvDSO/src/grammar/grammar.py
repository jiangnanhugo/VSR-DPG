import copy
import numpy as np
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr

from grammar.grammar_program import execute, SymbolicExpression, optimize
from grammar.grammar_utils import pretty_print_expr, expression_to_template, nth_repl


class ContextSensitiveGrammar(object):
    # will link to regression_task
    task = None
    # will link to grammarProgram
    program = None
    # threshold for deciding constants as summary or standalone constant
    opt_num_expriments = 5  # number of experiments done for multi-trail control variable experiments
    expr_obj_thres = 1e-6
    expr_consts_thres = 1e-3

    noise_std = 0.0

    """
       A Task in which the search space is a binary tree. Observations include
       the previous action, the parent, the sibling, and/or the number of dangling
       (unselected) nodes.
    """

    OBS_DIM = 4  # action, parent, sibling, dangling

    def __init__(self, nvars, production_rules, start_symbols, non_terminal_nodes,
                 max_length,
                 hof_size, reward_threhold):
        # number of input variables
        self.nvars = nvars
        # input variable symbols
        self.input_var_Xs = [Symbol('X' + str(i)) for i in range(self.nvars)]
        self.production_rules = production_rules

        self.start_symbol = 'f->' + start_symbols
        self.non_terminal_nodes = non_terminal_nodes
        self.max_length = max_length
        self.hof_size = hof_size
        self.reward_threhold = reward_threhold
        self.hall_of_fame = []
        self.allowed_grammar = np.ones(len(self.production_rules), dtype=bool)
        # those rules has terminal symbol on the right-hand side
        self.terminal_rules = [g for g in self.production_rules if
                               sum([nt in g[3:] for nt in self.non_terminal_nodes]) == 0]
        self.print_grammar_vocabulary()
        print(f"rules with only terminal symbols: {self.terminal_rules}")

        # used for output vocabulary
        self.n_action_inputs = self.output_vocab_size + 1  # Library tokens + empty token
        self.n_parent_inputs = self.output_vocab_size + 1  # - len(self.terminal_rules)  # Parent sub-lib tokens + empty token
        self.n_sibling_inputs = self.output_vocab_size + 1  # Library tokens + empty token
        self.EMPTY_ACTION = self.n_action_inputs - 1
        self.EMPTY_PARENT = self.n_parent_inputs - 1
        self.EMPTY_SIBLING = self.n_sibling_inputs - 1

    def allowed_grammar_indices(self, vc: set) -> list:
        """
        return the list of indices for all grammars that do not have the rules for controlled variables.
        :param vc: the set of controlled variables
        """
        filtered_grammars = []
        for idx, g in enumerate(self.production_rules):
            if sum([vi in g for vi in vc]) == 0:
                filtered_grammars.append(idx)
        return filtered_grammars

    @property
    def output_vocab_size(self):
        return len(self.production_rules)

    def print_grammar_vocabulary(self):
        print('============== GRAMMAR Vocabulary ==============')
        print('{0: >8} {1: >20}'.format('ID', 'NAME'))
        for i in range(len(self.production_rules)):
            print('{0: >8} {1: >20}'.format(i + 1, self.production_rules[i]))
        print('========== END OF GRAMMAR Vocabulary ===========')

    def valid_production_rules(self, Node):
        # Get index of all possible production rules starting with a given node
        return [self.production_rules.index(x) for x in self.production_rules if x.startswith(Node)]

    def get_non_terminal_nodes(self, prod) -> list:

        # Get all the non-terminal nodes from right-hand side of a production rule grammar
        return [i for i in prod[3:] if i in self.non_terminal_nodes]

    def complete_rules(self, list_of_rules):
        """
        complete all non-terminal symbols in rules.

        given one sequence of rules, either cut the sequence for the position where Number_of_Non_Terminal_Symbols=0,
        or add several rules with non only terminal symbols
        """
        ntn_counts = 0
        for one_rule in list_of_rules:
            ntn_counts += len(self.get_non_terminal_nodes(one_rule)) - 1
            if ntn_counts == 0:
                return list_of_rules
        # print(f"trying to complete all non-terminal in {list_of_rules} ==>", end="\t")
        #
        # for _ in range(ntn_counts):
        #     list_of_rules.append(np.random.choice(self.terminal_rules))
        # print(list_of_rules)
        return list_of_rules

    def construct_expression(self, many_seq_of_rules):
        filtered_many_rules = []
        for one_seq_of_rules in many_seq_of_rules:
            one_seq_of_rules = [self.start_symbol] + [self.production_rules[li] for li in one_seq_of_rules]

            one_list_of_rules = self.complete_rules(one_seq_of_rules)
            filtered_many_rules.append(one_list_of_rules)
            # print("pruned list_of_rules:", one_list_of_rules)
        self.task.rand_draw_data_with_X_fixed()
        y_true = self.task.evaluate()
        many_expressions = self.program.fitting_new_expressions_in_parallel(filtered_many_rules, self.task.X, y_true,
                                                                            self.input_var_Xs)
        for one_expression in many_expressions:
            if one_expression.reward != -np.inf:
                one_expression.all_metrics = self.print_reward_function_all_metrics(one_expression.fitted_eq)
        return many_expressions

    def freeze_equations(self, best_expressions, stand_alone_constants, next_free_variable):
        """
        in the proposed control variable experiment, we need to decide summary constants and stand alone constants.
        the threshold dependent on the evaluation metric.
        """
        print("---------Freeze Equation----------")
        freezed_exprs = []
        aug_nt_nodes = []
        new_stand_alone_constants = stand_alone_constants
        # only use the best
        fitted_expr = best_expressions[-1].fitted_eq
        optimized_constants = []
        optimized_obj = []
        expr_template = expression_to_template(parse_expr(fitted_expr), stand_alone_constants)
        print('expr template is:', expr_template)
        for _ in range(self.opt_num_expriments):
            self.task.rand_draw_X_fixed()
            self.task.rand_draw_data_with_X_fixed()
            y_true = self.task.evaluate()
            _, eq, opt_consts, opt_obj = optimize(
                expr_template,
                self.task.X,
                y_true,
                self.input_var_Xs,
                self.program.evaluate_loss,
                self.program.max_open_constants,
                2000,
                self.program.optimizer)
            ##
            optimized_constants.append(opt_consts)
            optimized_obj.append(opt_obj)
        optimized_constants = np.asarray(optimized_constants)
        # optimized_obj = np.asarray(optimized_obj)
        print("optimized_obj: ", optimized_obj)
        num_changing_consts = expr_template.count('C')
        is_summary_constants = np.zeros(num_changing_consts, dtype=int)
        if np.max(optimized_obj) <= self.expr_obj_thres:
            for ci in range(num_changing_consts):
                print("std", np.std(optimized_constants[:, ci]), end="\t")
                if abs(np.mean(optimized_constants[:, ci])) < 1e-5:
                    print(f'c{ci} is a noisy minial constant')
                    is_summary_constants[ci] = 2
                elif np.std(optimized_constants[:, ci]) <= self.expr_consts_thres:
                    print(f'c{ci} {np.mean(optimized_constants[:, ci])} is a stand-alone constant')
                else:
                    print(f'c{ci}  is a summary constant')
                    is_summary_constants[ci] = 1
            ####
            # summary constant vs controlled variable
            ####
            for ci in range(num_changing_consts):
                if is_summary_constants[ci] != 1:
                    continue
                print(expr_template)
                new_expr_template = nth_repl(copy.copy(expr_template), 'C', str(optimized_constants[-1, ci]), ci + 1)
                print(new_expr_template, ci, np.mean(optimized_constants[:, ci]))
                # optimized_constants = []
                optimized_cond_obj = []
                print('expr template is"', new_expr_template)
                for _ in range(self.opt_num_expriments * 3):
                    self.task.rand_draw_X_fixed_with_index(next_free_variable)
                    y_true = self.task.evaluate()
                    _, eq, opt_consts, opt_obj = optimize(
                        expr_template,
                        self.task.X,
                        y_true,
                        self.input_var_Xs,
                        self.program.evaluate_loss,
                        self.program.max_open_constants,
                        2000,
                        self.program.optimizer)
                    ##
                    # optimized_constants.append(opt_consts)
                    optimized_cond_obj.append(opt_obj)
                if np.max(optimized_cond_obj) <= self.expr_obj_thres:
                    is_summary_constants[ci] = 3
                else:
                    print(f'summary constant c{ci} will be a summary constant in the next round')
            # print all the information together
            for i, ci in enumerate(is_summary_constants):
                if ci == 0:
                    print(f"c{i} is a real stand-alone constant")
                elif ci == 1:
                    print(f'c{i}  is a summary constant')
                elif ci == 2:
                    print(f'c{i} is a noisy minial constant')
                elif ci == 3:
                    print(f'summary constant c{i} will still be a constant in the next round')
            ####
            cidx = 0
            new_expr_template = ''
            for ti in expr_template:
                if ti == 'C' and is_summary_constants[cidx] == 1:
                    # real summary constant in the next round
                    new_expr_template += '(A)'
                    cidx += 1
                elif ti == "C" and is_summary_constants[cidx] == 0:
                    # standalone constant
                    est_c = np.mean(optimized_constants[:, cidx])
                    if abs(est_c) < 1e-5:
                        est_c = 0.0
                    new_expr_template += str(est_c)
                    if len(new_stand_alone_constants) == 0 or min(
                            [abs(est_c - fi) for fi in new_stand_alone_constants]) < 1e-5:
                        new_stand_alone_constants.append(est_c)
                    cidx += 1
                elif ti == 'C' and is_summary_constants[cidx] == 2:
                    # noise values
                    new_expr_template += '0.0'
                    cidx += 1
                elif ti == 'C' and is_summary_constants[cidx] == 3:
                    # is a summary constant but will still be constant in the next round
                    new_expr_template += 'C'
                    cidx += 1
                else:
                    new_expr_template += ti
            freezed_exprs.append(new_expr_template)
            aug_nt_nodes.append(['A', ] * sum([1 for ti in new_expr_template if ti == 'A']))
            return freezed_exprs, aug_nt_nodes, new_stand_alone_constants

        print("No available expression is found....trying to add the current best guessed...")
        fitted_expr = best_expressions[-1].fitted_eq
        expr_template = expression_to_template(parse_expr(fitted_expr), stand_alone_constants)
        cidx = 0
        new_expr_template = ''
        for ti in expr_template:
            if ti == 'C':
                # summary constant
                new_expr_template += '(A)'
                cidx += 1
            else:
                new_expr_template += ti
        freezed_exprs.append(new_expr_template)
        aug_nt_nodes.append(['A', ] * sum([1 for ti in new_expr_template if ti == 'A']))
        expri, ntnodei = freezed_exprs[0], aug_nt_nodes[0]
        countA = expri.count('(A)')
        # diversify the number of A
        new_freezed_exprs = [expri, ]
        new_aug_nt_nodes = [['A', ] * countA, ]

        if countA >= 3:
            ti = 0
            while ti < 2:
                mask = np.random.randint(2, size=countA)
                while np.sum(mask) == 0 or np.sum(mask) == countA:
                    mask = np.random.randint(2, size=countA)
                countAi = 0
                expri_new = ""
                for i in range(len(expri)):
                    if expri[i] == 'A' and mask[countAi] == 0:
                        expri_new += 'C'
                    else:
                        expri_new += expri[i]
                    countAi += (expri[i] == 'A')
                if expri_new not in new_freezed_exprs:
                    new_freezed_exprs.append(expri_new)
                    new_aug_nt_nodes.append(['A', ] * (np.sum(mask)))
                    ti += 1
        else:
            new_freezed_exprs.append(expri)
            new_aug_nt_nodes.append(ntnodei)
        # only generate at most 3 template for the next round, otherwise it will be too time consuming
        ret_frezze_exprs, ret_aug_nt_nodes = [], []
        for x, y in zip(new_freezed_exprs, new_aug_nt_nodes):
            if x not in ret_frezze_exprs:
                ret_frezze_exprs.append(x)
                ret_aug_nt_nodes.append(y)
        return ret_frezze_exprs, ret_aug_nt_nodes, new_stand_alone_constants

    def update_hall_of_fame(self, one_fitted_expression: SymbolicExpression):

        if one_fitted_expression.traversal.count(';') <= self.max_length:
            if not self.hall_of_fame:
                self.hall_of_fame = [one_fitted_expression]
            elif one_fitted_expression.traversal not in [x.traversal for x in self.hall_of_fame]:
                if len(self.hall_of_fame) < self.hof_size:
                    self.hall_of_fame.append(one_fitted_expression)
                    # sorting the list in ascending order
                    self.hall_of_fame = sorted(self.hall_of_fame, key=lambda x: x.reward, reverse=False)
                else:
                    if one_fitted_expression.reward > self.hall_of_fame[-1].reward:
                        # sorting the list in ascending order
                        self.hall_of_fame = sorted(self.hall_of_fame[1:] + [one_fitted_expression],
                                                   key=lambda x: x.reward, reverse=False)

    def print_hofs(self, mode: str, verbose=False):
        """
        mode: if global, then we rank on no variable controlled.
        """
        if mode == 'global':
            old_vf = copy.copy(self.task.get_vf())
            self.program.vf = [1, ] * self.nvars
            self.task.set_allowed_inputs(self.task.get_vf())
            print("new vf for HOF ranking", self.task.get_vf(), self.task.fixed_column)
        self.task.rand_draw_data_with_X_fixed()
        print(f"PRINT HOF (free variables={self.task.fixed_column})")
        print("=" * 20)
        for pr in self.hall_of_fame[:self.hof_size]:
            if verbose:
                print('        ', pr, end="\n")
                pr.print_all_metrics()
            else:
                print('        ', pr, end="\n")
        print("=" * 20)
        if mode == 'global':
            self.task.vf = old_vf
            self.task.set_allowed_inputs(old_vf)
            print("reset old vf", self.task.get_vf(), self.task.fixed_column)

    def print_reward_function_all_metrics(self, expr_str, verbose=False):
        """used for print the error for all metrics between the predicted program `p` and true program."""
        y_hat = execute(expr_str, self.task.X.T, self.input_var_Xs)
        dict_of_result = self.task.data_query_oracle._evaluate_all_losses(self.task.X, y_hat)
        dict_of_result['tree_edit_distance'] = self.task.data_query_oracle.compute_normalized_tree_edit_distance(
            expr_str)
        if verbose:
            print('-' * 30)
            for mertic_name in dict_of_result:
                print(f"{mertic_name} {dict_of_result[mertic_name]}")
            print('-' * 30)
        return dict_of_result
