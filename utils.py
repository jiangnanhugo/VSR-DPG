"""Utility functions used in deep symbolic optimization."""

import collections
import copy
import functools
import numpy as np
import time
import importlib
import re
import pandas as pd


import os

import commentjson as json

def is_float(s):
    """Determine whether the input variable can be cast to float."""

    try:
        float(s)
        return True
    except ValueError:
        return False


# Adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points given an array of costs.

    Parameters
    ----------

    costs : np.ndarray
        Array of shape (n_points, n_costs).

    Returns
    -------

    is_efficient_maek : np.ndarray (dtype:bool)
        Array of which elements in costs are pareto-efficient.
    """

    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


class cached_property(object):
    """
    Decorator used for lazy evaluation of an object attribute. The property
    should be non-mutable, since it replaces itself.
    """

    def __init__(self, getter):
        self.getter = getter

        functools.update_wrapper(self, getter)

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = self.getter(obj)
        setattr(obj, self.getter.__name__, value)
        return value


def weighted_quantile(values, weights, q):
    """
    Computes the weighted quantile, equivalent to the exact quantile of the
    empirical distribution.

    Given ordered samples x_1 <= ... <= x_n, with corresponding weights w_1,
    ..., w_n, where sum_i(w_i) = 1.0, the weighted quantile is the minimum x_i
    for which the cumulative sum up to x_i is greater than or equal to 1.

    Quantile = min{ x_i | x_1 + ... + x_i >= q }
    """

    sorted_indices = np.argsort(values)
    sorted_weights = weights[sorted_indices]
    sorted_values = values[sorted_indices]
    cum_sorted_weights = np.cumsum(sorted_weights)
    i_quantile = np.argmax(cum_sorted_weights >= q)
    quantile = sorted_values[i_quantile]

    # NOTE: This implementation is equivalent to (but much faster than) the
    # following:
    # from scipy import stats
    # empirical_dist = stats.rv_discrete(name='empirical_dist', values=(values, weights))
    # quantile = empirical_dist.ppf(q)

    return quantile


# Entropy computation in batch
def empirical_entropy(labels):

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.
    # Compute entropy
    for i in probs:
        ent -= i * np.log(i)

    return ent


def get_duration(start_time):
    return get_human_readable_time(time.time() - start_time)


def get_human_readable_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return "{:02d}:{:02d}:{:02d}:{:05.2f}".format(int(d), int(h), int(m), s)


def safe_merge_dicts(base_dict, update_dict):
    """Merges two dictionaries without changing the source dictionaries.

    Parameters
    ----------
        base_dict : dict
            Source dictionary with initial values.
        update_dict : dict
            Dictionary with changed values to update the base dictionary.

    Returns
    -------
        new_dict : dict
            Dictionary containing values from the merged dictionaries.
    """
    if base_dict is None:
        return update_dict
    base_dict = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if isinstance(value, collections.abc.Mapping):
            base_dict[key] = safe_merge_dicts(base_dict.get(key, {}), value)
        else:
            base_dict[key] = value
    return base_dict


def safe_update_summary(csv_path, new_data):
    """Updates a summary csv file with new rows. Adds new columns
    in existing data if necessary. New rows are distinguished by
    the run seed.

    Parameters
    ----------
        csv_path : str
            String with the path to the csv file.
        new_data : dict
            Dictionary containing values to be saved in the csv file.

    Returns
    -------
        bool
            Boolean value to indicate if saving the data to file worked.
    """
    try:
        new_data_pd = pd.DataFrame(new_data, index=[0])
        new_data_pd.set_index('seed', inplace=True)
        if os.path.isfile(csv_path):
            old_data_pd = pd.read_csv(csv_path)
            old_data_pd.set_index('seed', inplace=True)
            merged_df = pd.concat([old_data_pd, new_data_pd], axis=0, ignore_index=False)
            merged_df.to_csv(csv_path, header=True, mode='w+', index=True)
        else:
            new_data_pd.to_csv(csv_path, header=True, mode='w+', index=True)
        return True
    except:
        return False





##### load configure files
def get_base_config():
    # Load base config
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config', "config_common.json"), encoding='utf-8') as f:
        base_config = json.load(f)

    # Load task specific config
    task_config_file = "config_regression.json"
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config', task_config_file), encoding='utf-8') as f:
        task_config = json.load(f)

    return safe_merge_dicts(base_config, task_config)


def load_config(config=None):
    # Load user config
    print("user config file:",config)
    if isinstance(config, str):
        with open(config, encoding='utf-8') as f:
            user_config = json.load(f)
    elif isinstance(config, dict):
        user_config = config
    else:
        assert config is None, "Config must be None, str, or dict."
        user_config = {}

    # Load task-specific base config
    base_config = get_base_config()

    # Return combined configs
    return safe_merge_dicts(base_config, user_config)

def create_reward_threshold(highest_threhold, nvar, ratio=0.95):
    return [highest_threhold * ratio ** i for i in range(nvar)]



def create_geometric_generations(n_generations, nvar, ratio=1.2):
    gens = [0] * nvar
    round = 0
    total_ratios = sum([ratio ** it for it in range(nvar)])
    for it in range(nvar):
        gens[it] += int(n_generations * ratio ** it / total_ratios)

    # gens[0] = n_generations
    for it in range(0, nvar):
        if gens[it] < 20:
            gens[it] = 20
    gens = gens
    print('generation #:', gens, 'sum=', sum(gens))
    return gens


def create_uniform_generations(n_generations, nvar):
    gens = [0] * nvar
    for it in range(nvar):
        gens[it] = n_generations
    print('generation #:', gens, 'sum=', sum(gens))
    return gens
