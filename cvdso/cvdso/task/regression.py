import numpy as np
import pandas as pd
import scipy

"""Factory functions for generating symbolic search tasks."""

import numpy as np

from cvdso.program import Program
from cvdso.library import Library
from cvdso.functions import create_tokens
from cvdso.subroutines import parents_siblings


class Task(object):
    """
    A Task in which the search space is a binary tree. Observations include
    the previous action, the parent, the sibling, and/or the number of dangling
    (unselected) nodes.
    """

    OBS_DIM = 4  # action, parent, sibling, dangling

    def __init__(self):
        pass

    def get_next_obs(self, actions, obs):
        dangling = obs[:, 3]  # Shape of obs: (?, 4)
        action = actions[:, -1]  # Current action
        lib = self.library

        # Compute parents and siblings
        parent, sibling = parents_siblings(actions,
                                           arities=lib.arities,
                                           parent_adjust=lib.parent_adjust,
                                           empty_parent=lib.EMPTY_PARENT,
                                           empty_sibling=lib.EMPTY_SIBLING)

        # Update dangling with (arity - 1) for each element in action
        dangling += lib.arities[action] - 1

        prior = self.prior(actions, parent, sibling, dangling)  # (?, n_choices)

        # Reset initial values when tree completes
        if Program.n_objects > 1:  # NOTE: do this to save computuational cost only when n_objects > 1
            finished = (dangling == 0)
            dangling[finished] = 1
            action[finished] = lib.EMPTY_ACTION
            parent[finished] = lib.EMPTY_PARENT
            sibling[finished] = lib.EMPTY_SIBLING
            prior[finished] = self.prior.initial_prior()

        next_obs = np.stack([action, parent, sibling, dangling], axis=1)  # (?, 4)
        next_obs = next_obs.astype(np.float32)
        return next_obs, prior

    def reset_task(self, prior):
        """
        Returns the initial observation: empty action, parent, and sibling, and
        dangling is 1.
        """

        self.prior = prior

        # Order of observations: action, parent, sibling, dangling
        initial_obs = np.array([self.library.EMPTY_ACTION,
                                self.library.EMPTY_PARENT,
                                self.library.EMPTY_SIBLING,
                                1],
                               dtype=np.float32)
        return initial_obs


def set_task(config_task):
    """Helper function to make set the Program class Task and execute function
    from task config."""

    # Use of protected functions is the same for all tasks, so it's handled separately
    protected = config_task["protected"] if "protected" in config_task else False

    Program.set_execute(protected)
    print(config_task)
    task = RegressionTask(**config_task)

    Program.set_task(task)


class RegressionTask(Task):
    """
    Class for the symbolic regression task. Discrete objects are expressions,
    which are evaluated based on their fitness to a specified dataset.
    """

    task_type = "regression"

    def __init__(self,
                 batchsize, allowed_input, dataX, data_query_oracle,
                 metric="inv_nrmse",
                 metric_params=(1.0,), threshold=1e-12,
                 normalize_variance=False, protected=False):
        """
        Parameters
        ----------

        metric : str
            Name of reward function metric to use.

        metric_params : list
            List of metric-specific parameters.

        threshold : float
            Threshold of NMSE on noiseless data used to determine success.

        normalize_variance : bool
            If True and reward_noise_type=="r", reward is multiplied by
            1 / sqrt(1 + 12*reward_noise**2) (We assume r is U[0,1]).

        protected : bool
            Whether to use protected functions.

        """

        super(Task).__init__()

        """
        Configure (X, y) train/test data. There are four supported use cases:
        (1) named benchmark, (2) benchmark config, (3) filename, and (4) direct
        (X, y) data.
        """
        self.X_test = self.y_test = self.y_test_noiseless = None

        self.batchsize = batchsize
        self.allowed_input = allowed_input
        self.n_input = allowed_input.size
        self.dataX = dataX
        self.data_query_oracle = data_query_oracle

        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]
        self.X_fixed = np.random.rand(self.n_input)

        """
        Configure train/test reward metrics.
        """
        self.threshold = threshold
        self.invalid_reward, self.max_reward = make_regression_metric(metric, *metric_params)

        """
        Configure reward noise.
        """
        self.normalize_variance = normalize_variance
        self.rng = None
        self.scale = None

        # Set the Library
        tokens = create_tokens(n_input_var=self.data_query_oracle.get_nvars(),
                               function_set=self.data_query_oracle.operators_set,
                               protected=protected)
        self.library = Library(tokens)

    def set_allowed_inputs(self, allowed_inputs):
        self.allowed_input = np.copy(allowed_inputs)
        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]

    def set_allowed_input(self, i, flag):
        self.allowed_input[i] = flag
        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]

    def rand_draw_X_non_fixed(self):
        self.X = self.dataX.randn(sample_size=self.batchsize).T

    def rand_draw_X_fixed(self):
        self.X_fixed = np.squeeze(self.dataX.randn(sample_size=1))

    def rand_draw_data_with_X_fixed(self):
        self.X = self.dataX.randn(sample_size=self.batchsize).T
        if len(self.fixed_column):
            self.X[:, self.fixed_column] = self.X_fixed[self.fixed_column]

    def reward_function(self, p):
        # Compute estimated values
        y_hat = p.execute(self.X)

        # For invalid expressions, return invalid_reward
        if p.invalid:
            return self.invalid_reward

        # Compute metric
        return self.data_query_oracle._evaluate_loss(self.X, y_hat)

    def print_reward_function_all_metrics(self, p):
        """used for print the error for all metrics between the predicted program `p` and true program."""
        y_hat = p.execute(self.X)
        dict_of_result = self.data_query_oracle._evaluate_all_losses(self.X, y_hat)
        print('-' * 30)
        for mertic_name in dict_of_result:
            print(f"{mertic_name} {dict_of_result[mertic_name]}")
        print('-' * 30)

    def evaluate(self, p):

        # Compute predictions on test data
        y_hat = p.execute(self.X)
        if p.invalid:
            nmse_test = None
            success = False

        else:
            # NMSE on test data (used to report final error)
            nmse_test = self.data_query_oracle._evaluate_loss(self.X, y_hat)  # np.mean((self.y_test - y_hat) ** 2) / self.var_y_test

            # Success is defined by NMSE on noiseless test data below a threshold
            success = False

        info = {
            "nmse_test": nmse_test,
            "nmse_test_noiseless": nmse_test,
            "success": success
        }

        return info


def make_regression_metric(name, y_train, *args):
    """
    Factory function for a regression metric. This includes a closures for
    metric parameters and the variance of the training data.

    Parameters
    ----------

    name : str
        Name of metric. See all_metrics for supported metrics.

    args : args
        Metric-specific parameters

    Returns
    -------

    metric : function
        Regression metric mapping true and estimated values to a scalar.

    invalid_reward: float or None
        Reward value to use for invalid expression. If None, the training
        algorithm must handle it, e.g. by rejecting the sample.

    max_reward: float
        Maximum possible reward under this metric.
    """

    var_y = np.var(y_train)

    # For negative MSE-based rewards, invalid reward is the value of the reward function when y_hat = mean(y)
    # For inverse MSE-based rewards, invalid reward is 0.0
    # For non-MSE-based rewards, invalid reward is the minimum value of the reward function's range
    all_invalid_rewards = {
        "neg_mse": -var_y,
        "neg_rmse": -np.sqrt(var_y),
        "neg_nmse": -1.0,
        "neg_nrmse": -1.0,
        "neglog_mse": -np.log(1 + var_y),
        "inv_mse": 0.0,  # 1/(1 + args[0]*var_y),
        "inv_nmse": 0.0,  # 1/(1 + args[0]),
        "inv_nrmse": 0.0,  # 1/(1 + args[0]),
        "fraction": 0.0,
        "pearson": 0.0,
        "spearman": 0.0
    }
    invalid_reward = all_invalid_rewards[name]

    all_max_rewards = {
        "neg_mse": 0.0,
        "neg_rmse": 0.0,
        "neg_nmse": 0.0,
        "neg_nrmse": 0.0,
        "neglog_mse": 0.0,
        "inv_mse": 1.0,
        "inv_nmse": 1.0,
        "inv_nrmse": 1.0,
        "fraction": 1.0,
        "pearson": 1.0,
        "spearman": 1.0
    }
    max_reward = all_max_rewards[name]

    return invalid_reward, max_reward
