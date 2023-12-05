"""Class for symbolic expression object or program."""

import array
import warnings

import numpy as np

from cvdso.functions import PlaceholderConstant
from cvdso.const import make_const_optimizer
from cvdso.utils import cached_property
import cvdso.utils as U


def _finish_tokens(tokens):
    """
    Complete a possibly unfinished string of tokens.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal.

    Returns
    tokens : list of ints
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    """

    n_objects = 1

    arities = np.array([Program.library.arities[t] for t in tokens])
    # Number of dangling nodes, returns the cumsum up to each point
    # Note that terminal nodes are -1 while functions will be >= 0 since arities - 1
    dangling = 1 + np.cumsum(arities - 1)

    if -n_objects in (dangling - 1):
        # Chop off tokens once the cumsum reaches 0, This is the last valid point in the tokens
        expr_length = 1 + np.argmax((dangling - 1) == -n_objects)
        tokens = tokens[:expr_length]
    else:
        # Extend with valid variables until string is valid
        # NOTE: This only appends onto the end of a set of tokens, even in the multi-object case!
        assert n_objects == 1, "Is max length constraint turned on? Max length constraint required when n_objects > 1."
        tokens = np.append(tokens, np.random.choice(Program.library.input_tokens, size=dangling[-1]))

    return tokens


def from_tokens(tokens, skip_cache=False, on_policy=True, finish_tokens=True):
    """
    Memoized function to generate a Program from a list of tokens.

    Since some tokens are nonfunctional, this first computes the corresponding
    traversal. If that traversal exists in the cache, the corresponding Program
    is returned. Otherwise, a new Program is returned.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    skip_cache : bool
        Whether to bypass the cache when creating the program (used for
        previously learned symbolic actions in DSP).
        
    finish_tokens: bool
        Do we need to finish this token. There are instances where we have
        already done this. Most likely you will want this to be True. 

    Returns
    _______
    program : Program
        The Program corresponding to the tokens, either pulled from memoization
        or generated from scratch.
    """

    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''

    if finish_tokens:
        tokens = _finish_tokens(tokens)

    # For stochastic Tasks, there is no cache; always generate a new Program.
    # For deterministic Programs, if the Program is in the cache, return it;
    # otherwise, create a new one and add it to the cache.
    if skip_cache:
        p = Program(tokens, on_policy=on_policy)
    else:
        key = tokens.tostring()
        try:
            p = Program.cache[key]
            if on_policy:
                p.on_policy_count += 1
            else:
                p.off_policy_count += 1
        except KeyError:
            p = Program(tokens, on_policy=on_policy)
            Program.cache[key] = p

    return p


class Program(object):
    """
    The executable program representing the symbolic expression.

    The program comprises unary/binary operators, constant placeholders
    (to-be-optimized), input variables, and hard-coded constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. "Dangling"
        programs are completed with repeated "x1" until the expression
        completes.

    Attributes
    ----------
    traversal : list
        List of operators (type: Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarry (dtype: int)
        Array of integers whose values correspond to indices

    const_pos : list of int
        A list of indicies of constant placeholders along the traversal.

    float_pos : list of float
        A list of indices of constants placeholders or floating-point constants
        along the traversal.

    sympy_expr : str
        The (lazily calculated) SymPy expression corresponding to the program.
        Used for pretty printing _only_.

    complexity : float
        The (lazily calcualted) complexity of the program.

    r : float
        The (lazily calculated) reward of the program.

    count : int
        The number of times this Program has been sampled.

    str : str
        String representation of tokens. Useful as unique identifier.
    """

    # Static variables
    task = None  # Task
    library = None  # Library
    const_optimizer = None  # Function to optimize constants
    cache = {}
    n_objects = 1  # Number of executable objects per Program instance

    # Cython-related static variables
    have_cython = None  # Do we have cython installed
    execute = None  # Link to execute. Either cython or python
    cyfunc = None  # Link to cyfunc lib since we do an include inline

    def __init__(self, tokens=None, on_policy=True):
        """
        Builds the Program from a list of of integers corresponding to Tokens.
        """

        # Can be empty if we are unpickling 
        if tokens is not None:
            self._init(tokens, on_policy)

    def _init(self, tokens, on_policy=True):

        self.traversal = [Program.library[t] for t in tokens]
        self.const_pos = [i for i, t in enumerate(self.traversal) if isinstance(t, PlaceholderConstant)]
        self.len_traversal = len(self.traversal)

        if self.have_cython and self.len_traversal > 1:
            self.is_input_var = array.array('i', [t.input_var is not None for t in self.traversal])

        self.invalid = False
        self.str = tokens.tostring()
        self.tokens = tokens

        self.on_policy_count = 1 if on_policy else 0
        self.off_policy_count = 0 if on_policy else 1
        self.originally_on_policy = on_policy  # Note if a program was created on policy

    def __getstate__(self):

        have_r = "r" in self.__dict__
        have_evaluate = "evaluate" in self.__dict__
        possible_const = have_r or have_evaluate

        state_dict = {'tokens': self.tokens,
                      # string rep comes out different if we cast to array, so we can get cache misses.
                      'have_r': bool(have_r),
                      'r': float(self.r) if have_r else float(-np.inf),
                      'have_evaluate': bool(have_evaluate),
                      'evaluate': self.evaluate if have_evaluate else float(-np.inf),
                      'const': array.array('d', self.get_constants()) if possible_const else float(-np.inf),
                      'on_policy_count': bool(self.on_policy_count),
                      'off_policy_count': bool(self.off_policy_count),
                      'originally_on_policy': bool(self.originally_on_policy),}

        # In the future we might also return sympy_expr and complexity if we ever need to compute in parallel 

        return state_dict

    def __setstate__(self, state_dict):

        # Question, do we need to init everything when we have already run, or just some things?
        self._init(state_dict['tokens'], state_dict['originally_on_policy'])

        have_run = False

        if state_dict['have_r']:
            setattr(self, 'r', state_dict['r'])
            have_run = True

        if state_dict['have_evaluate']:
            setattr(self, 'evaluate', state_dict['evaluate'])
            have_run = True

        if have_run:
            self.set_constants(state_dict['const'].tolist())
            self.invalid = state_dict['invalid']
            self.error_node = state_dict['error_node'].tounicode()
            self.error_type = state_dict['error_type'].tounicode()
            self.on_policy_count = state_dict['on_policy_count']
            self.off_policy_count = state_dict['off_policy_count']

    def execute(self, X):
        """
        Execute program on input X.

        Parameters
        ==========

        X : np.array
            Input to execute the Program over.

        Returns
        =======

        result : np.array or list of np.array
            In a single-object Program, returns just an array. In a multi-object Program, returns a list of arrays.
        """
        if not Program.protected:
            result = Program.execute_function(self.traversal, X)
        else:
            result = Program.execute_function(self.traversal, X)
        return result

    def optimize(self):
        """
        Optimizes PlaceholderConstant tokens against the reward function. The
        optimized values are stored in the traversal.
        """

        if len(self.const_pos) == 0:
            return

        # Define the objective function: negative reward
        def f(consts):
            self.set_constants(consts)
            r = self.task.reward_function(self)
            obj = -r  # Constant optimizer minimizes the objective function

            # Need to reset to False so that a single invalid call during
            # constant optimization doesn't render the whole Program invalid.
            self.invalid = False

            return obj

        self.task.rand_draw_data_with_X_fixed()
        # Do the optimization
        x0 = np.ones(len(self.const_pos))  # Initial guess
        optimized_constants = Program.const_optimizer(f, x0)

        # Set the optimized constants
        self.set_constants(optimized_constants)

    def get_constants(self):
        """Returns the values of a Program's constants."""

        return [t.value for t in self.traversal if isinstance(t, PlaceholderConstant)]

    def set_constants(self, consts):
        """Sets the program's constants to the given values"""

        for i, const in enumerate(consts):
            assert U.is_float, "Input to program constants must be of a floating point type"
            # Create a new instance of PlaceholderConstant instead of changing
            # the "values" attribute, otherwise all Programs will have the same
            # instance and just overwrite each other's value.
            self.traversal[self.const_pos[i]] = PlaceholderConstant(const)

    @classmethod
    def clear_cache(cls):
        """Clears the class' cache"""
        cls.cache = {}

    @classmethod
    def set_task(cls, task):
        """Sets the class' Task"""

        Program.task = task
        Program.library = task.library

    @classmethod
    def set_const_optimizer(cls, **kwargs):
        """Sets the class' constant optimizer"""

        const_optimizer = make_const_optimizer(**kwargs)
        Program.const_optimizer = const_optimizer

    @classmethod
    def set_complexity(cls, name):
        """Sets the class' complexity function"""

        all_functions = {
            # No complexity
            None: lambda p: 0.0,

            # Length of sequence
            "length": lambda p: len(p.traversal),

            # Sum of token-wise complexities
            "token": lambda p: sum([t.complexity for t in p.traversal]),

        }

        assert name in all_functions, "Unrecognzied complexity function name."

        Program.complexity_function = lambda p: all_functions[name](p)

    @classmethod
    def set_execute(cls, protected):
        """Sets which execute method to use"""

        # Check if cython_execute can be imported; if not, fall back to python_execute
        from cvdso.execute import cython_execute
        execute_function = cython_execute
        Program.have_cython = True

        if protected:
            Program.protected = True
            Program.execute_function = execute_function
        else:
            Program.protected = False

            # Define closure for execute function
            def unsafe_execute(traversal, X):
                """This is a wrapper for execute_function. If a floating-point error
                would be hit, a warning is logged instead, p.invalid is set to True,
                and the appropriate nan/inf value is returned. It's up to the task's
                reward function to decide how to handle nans/infs."""

                return execute_function(traversal, X)

            Program.execute_function = unsafe_execute

    @cached_property
    def r(self):
        """Evaluates and returns the reward of the program"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Optimize any PlaceholderConstants
            self.optimize()

            # Return final reward after optimizing
            return self.task.reward_function(self)

    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_function(self)

    @cached_property
    def evaluate(self):
        """Evaluates and returns the evaluation metrics of the program."""

        # Program must be optimized before computing evaluate
        if "r" not in self.__dict__:
            print("WARNING: Evaluating Program before computing its reward." \
                  "Program will be optimized first.")
            self.optimize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return self.task.evaluate(self)

    @cached_property
    def sympy_expr(self):
        """
        Returns the attribute self.sympy_expr.

        This is actually a bit complicated because we have to go: traversal -->
        tree --> serialized tree --> SymPy expression
        """

        exprs = []
        for i in range(len(self.traversals)):
            tree = self.traversals[i].copy()
            exprs.append(tree)
        return exprs

    def print_stats(self):
        """Prints the statistics of the program
        
            We will print the most honest reward possible when using validation.
        """

        print("\tReward: {}".format(self.r))
        print("\tCount Off-policy: {}".format(self.off_policy_count))
        print("\tCount On-policy: {}".format(self.on_policy_count))
        print("\tOriginally on Policy: {}".format(self.originally_on_policy))
        print("\tInvalid: {}".format(self.invalid))
        print("\tTraversal: {}".format(self))
        self.task.rand_draw_data_with_X_fixed()
        self.task.print_reward_function_all_metrics(self)

    def __repr__(self):
        """Prints the program's traversal"""
        return ','.join([repr(t) for t in self.traversal])
