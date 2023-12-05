import copy
import sys
import numpy as np
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr
from production_rules import production_rules_to_expr
from cvdso.grammar_program import execute
from cvdso.grammar_utils import pretty_print_expr, expression_to_template, nth_repl


class ContextFreeGrammar(object):
    """
    hall_of_fame: ranked good expressions.
    """
    task = None  # Task
    program = None
    # constants
    opt_num_expr = 1  # number of experiments done for optimization
    expr_obj_thres = 1e-6
    expr_consts_thres = 1e-3

    noise_std = 0.0

    def __init__(self, base_grammars, aug_grammars,
                 non_terminal_nodes, aug_nt_nodes,
                 max_len, max_module,
                 aug_grammars_allowed, max_opt_iter=500):
        # number of input variables
        self.nvars = self.task.data_query_oracle.get_nvars()
        self.input_var_Xs = [Symbol('X' + str(i)) for i in range(self.nvars)]
        self.base_grammars = base_grammars
        self.aug_grammars = aug_grammars
        self.grammars = base_grammars + [x for x in aug_grammars if x not in base_grammars]
        self.aug_nt_nodes = aug_nt_nodes
        self.non_terminal_nodes = non_terminal_nodes
        self.max_len = max_len
        self.max_module = max_module
        self.max_aug = aug_grammars_allowed
        self.hall_of_fame = []
        self.eta = 0.99
        self.max_opt_iter = max_opt_iter

    def valid_production_rules(self, Node):
        # Get index of all possible production rules starting with a given node
        return [self.grammars.index(x) for x in self.grammars if x.startswith(Node)]

    def valid_non_terminal_production_rules(self, Node):
        # Get index of all possible production rules starting with a given node
        valid_rules = []
        for i, x in enumerate(self.grammars):
            if x.startswith(Node) and np.sum([y in x[3:] for y in self.non_terminal_nodes]):
                valid_rules.append(i)
        return valid_rules
        # return [self.grammars.index(x) for x in self.grammars if x.startswith(Node) ]

    def get_non_terminal_nodes(self, prod) -> list:
        # Get all the non-terminal nodes from right-hand side of a production rule grammar
        return [i for i in prod[3:] if i in self.non_terminal_nodes]

    def step(self, state, action_idx, ntn):
        """
        state:      all production rules
        action_idx: index of grammar starts from the current Non-terminal Node
        tree:       the current tree
        ntn:        all remaining non-terminal nodes

        This defines one step of Parse Tree traversal
        return tree (next state), remaining non-terminal nodes, reward, and if it is done
        """
        action = self.grammars[action_idx]
        state = state + ',' + action
        ntn = self.get_non_terminal_nodes(action) + ntn

        if not ntn:
            self.task.rand_draw_data_with_X_fixed()
            y_true = self.task.evaluate()
            expr_template = production_rules_to_expr(state.split(','))
            reward, eq, _, _ = self.program.optimize(expr_template,
                                                     len(state.split(',')),
                                                     self.task.X,
                                                     y_true,
                                                     self.input_var_Xs,
                                                     max_opt_iter=self.max_opt_iter)

            return state, ntn, reward, True, eq
        else:
            return state, ntn, 0, False, None

    def freeze_equations(self, list_of_grammars, opt_num_expr, stand_alone_constants, next_free_variable):
        # decide summary constants and stand alone constants.
        print("---------Freeze Equation----------")
        freezed_exprs = []
        aug_nt_nodes = []
        new_stand_alone_constants = stand_alone_constants
        # only use the best
        state, _, expr = list_of_grammars[-1]
        optimized_constants = []
        optimized_obj = []
        expr_template = expression_to_template(parse_expr(expr), stand_alone_constants)
        print('expr template is"', expr_template)
        for _ in range(opt_num_expr):
            self.task.rand_draw_X_fixed()
            self.task.rand_draw_data_with_X_fixed()
            y_true = self.task.evaluate()
            _, eq, opt_consts, opt_obj = self.program.optimize(expr_template,
                                                               len(state.split(',')),
                                                               self.task.X,
                                                               y_true,
                                                               self.input_var_Xs,
                                                               eta=self.eta,
                                                               max_opt_iter=1000)
            ##
            optimized_constants.append(opt_consts)
            optimized_obj.append(opt_obj)
        optimized_constants = np.asarray(optimized_constants)
        optimized_obj = np.asarray(optimized_obj)
        print(optimized_obj)
        num_changing_consts = expr_template.count('C')
        is_summary_constants = np.zeros(num_changing_consts)
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
                for _ in range(opt_num_expr * 3):
                    self.task.rand_draw_X_fixed_with_index(next_free_variable)
                    y_true = self.task.evaluate()
                    _, eq, opt_consts, opt_obj = self.program.optimize(new_expr_template,
                                                                       len(state.split(',')),
                                                                       self.task.X,
                                                                       y_true,
                                                                       self.input_var_Xs,
                                                                       eta=self.eta,
                                                                       max_opt_iter=1000)
                    ##
                    # optimized_constants.append(opt_consts)
                    optimized_cond_obj.append(opt_obj)
                if np.max(optimized_cond_obj) <= self.expr_obj_thres:
                    print(f'summary constant c{ci} will still be a constant in the next round')
                    is_summary_constants[ci] = 3
                else:
                    print(f'summary constant c{ci} will be a summary constant in the next round')

            ####
            cidx = 0
            new_expr_template = 'B->'
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
        state, _, expr = list_of_grammars[-1]
        expr_template = expression_to_template(parse_expr(expr), stand_alone_constants)
        cidx = 0
        new_expr_template = 'B->'
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
        # only generate at most 3 template for the next round, otherwise it will be too time counsuming
        ret_frezze_exprs, ret_aug_nt_nodes = [], []
        for x, y in zip(new_freezed_exprs, new_aug_nt_nodes):
            if x not in ret_frezze_exprs:
                ret_frezze_exprs.append(x)
                ret_aug_nt_nodes.append(y)
        return ret_frezze_exprs, ret_aug_nt_nodes, new_stand_alone_constants

    def rollout(self, num_play, state_initial, ntn_initial):
        """
        Perform `num_play` simulation, get the maximum reward
        """
        best_eq = ''
        reward = -100
        next_state = None
        eq = ''
        best_r = -100
        idx = 0
        while idx < num_play:
            done = False
            state = state_initial
            ntn = ntn_initial

            while not done:
                valid_index = self.valid_production_rules(ntn[0])
                action = np.random.choice(valid_index)
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn[1:])
                state = next_state
                ntn = ntn_next

                if state.count(',') >= self.max_len:  # tree depth shall be less than max_len
                    break

            if done:
                idx += 1
                if reward > best_r:
                    # save the current expression into hall-of-fame
                    self.update_hall_of_fame(next_state, reward, eq)
                    best_eq = eq
                    best_r = reward

        return best_r, best_eq



    def update_hall_of_fame(self, state, reward, eq):
        """
        If we pass by a concise solution with high score, we store it as an
        single action for future use.
        """
        module = state
        if state.count(',') <= self.max_module:
            if not self.hall_of_fame:
                self.hall_of_fame = [(module, reward, eq)]
            elif eq not in [x[2] for x in self.hall_of_fame]:
                if len(self.hall_of_fame) < self.max_aug:
                    self.hall_of_fame = sorted(self.hall_of_fame + [(module, reward, eq)], key=lambda x: x[1])
                else:
                    if reward > self.hall_of_fame[0][1]:
                        self.hall_of_fame = sorted(self.hall_of_fame[1:] + [(module, reward, eq)], key=lambda x: x[1])


    def print_hofs(self, flag, verbose=False):
        if flag == -1:
            old_vf = copy.copy(self.program.get_vf())
            self.program.vf = [1, ] * self.nvars
            self.task.set_allowed_inputs(self.program.get_vf())
            print("new vf for HOF ranking", self.program.get_vf(), self.task.fixed_column)
        self.task.rand_draw_data_with_X_fixed()
        print(f"PRINT HOF (free variables={self.task.fixed_column})")
        print("=" * 20)
        for pr in self.hall_of_fame[-len(self.hall_of_fame):]:
            if verbose:
                print('        ' + str(get_state(pr)), end="\n")
                self.print_reward_function_all_metrics(pr[2])
            else:
                print('        ' + str(get_state(pr)), end="\n")
        print("=" * 20)
        if flag == -1:
            self.program.vf = old_vf
            self.task.set_allowed_inputs(old_vf)
            print("reset old vf", self.program.get_vf(), self.task.fixed_column)

    def print_reward_function_all_metrics(self, expr_str):
        """used for print the error for all metrics between the predicted program `p` and true program."""
        y_hat = execute(expr_str, self.task.X.T, self.input_var_Xs)
        dict_of_result = self.task.data_query_oracle._evaluate_all_losses(self.task.X, y_hat)
        dict_of_result['tree_edit_distance'] = self.task.data_query_oracle.compute_normalized_tree_edit_distance(
            expr_str)
        print('-' * 30)
        for mertic_name in dict_of_result:
            print(f"{mertic_name} {dict_of_result[mertic_name]}")
        print('-' * 30)


def get_state(pr):
    state_dict = {
        'reward': pr[1],
        'pretty-eq': pretty_print_expr(pr[2]),
        'rules': pr[0],
    }
    return state_dict
