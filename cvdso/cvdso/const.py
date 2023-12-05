"""Constant optimizer used for deep symbolic optimization."""

from functools import partial

import numpy as np
from scipy.optimize import minimize


def make_const_optimizer(**kwargs):
    """Returns a ConstOptimizer given a name and keyword arguments"""
    return ScipyMinimize(**kwargs)


class ScipyMinimize(object):
    """SciPy's non-linear optimizer"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, f, x0):
        """
        Optimizes an objective function from an initial guess.

               The objective function is the negative of the base reward (reward
               without penalty) used for training. Optimization excludes any penalties
               because they are constant w.r.t. to the constants being optimized.

               Parameters
               ----------
               f : function mapping np.ndarray to float
                   Objective function (negative base reward).

               x0 : np.ndarray
                   Initial guess for constant placeholders.

               Returns
               -------
               x : np.ndarray
                   Vector of optimized constants.
        """
        with np.errstate(divide='ignore'):
            opt_result = partial(minimize, **self.kwargs)(f, x0)

        x = opt_result['x']
        return x
