"""Core deep symbolic optimizer construct."""
import copy
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import random
from time import time
from datetime import datetime

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import commentjson as json

from cvdso.task.regression import set_task
from cvdso.expression_decoder import ExpressionDecoder
from train import learn
from cvdso.prior import make_prior
from cvdso.program import Program, ScipyMinimize
from cvdso.utils import load_config
from cvdso.tf_state_manager import make_state_manager as manager_make_state_manager


class CVDeepSymbolicOptimizer(object):
    """
    Control variable for Deep symbolic optimization model. Includes model hyperparameters and
    training configuration.
    """

    def __init__(self, config=None, function_set=None, dataX=None, data_query_oracle=None, config_filename=None):
        """config : dict or str. Config dictionary or path to JSON."""
        self.config_filename = config_filename
        self.set_config(config)
        self.sess = None
        self.dataX = dataX
        self.data_query_oracle = data_query_oracle

        self.config_task['batchsize'] = self.config_training['batch_size']
        self.function_set = function_set
        print(f"self.function_set ={self.function_set}")
        nvar = data_query_oracle.get_nvars()
        self.allowed_input_tokens = np.ones(nvar, dtype=np.int32)

        self.task_name = "_".join(['regression', self.config_filename.split("/")[-1],
                                   self.data_query_oracle._get_eq_name().split('/')[-1],
                                   self.data_query_oracle.noise_type,
                                   str(self.data_query_oracle.noise_scale)])

    def setup(self):
        # Clear the cache and reset the compute graph
        Program.clear_cache()
        tf.compat.v1.reset_default_graph()

        # Generate objects needed for training and set seeds
        self.pool = self.make_pool_and_set_task()
        self.set_seeds()  # Must be called _after_ resetting graph and _after_ setting task
        self.sess = tf.compat.v1.Session()

        # Save complete configuration file
        self.output_file = self.make_output_file()
        self.save_config()

        # Prepare training parameters
        self.prior = make_prior(Program.library, self.config_prior)
        self.state_manager = manager_make_state_manager(self.config_state_manager)
        self.expression_decoder = ExpressionDecoder(self.sess,
                                                    self.prior,
                                                    self.state_manager,
                                                    **self.config_expression_decoder)


    def train(self):
        # Train the model
        result = {"seed": self.config_experiment["seed"]}  # Seed listed first
        result_dict = learn(self.sess,
                            self.expression_decoder,
                            self.pool,
                            self.output_file,
                            **self.config_training)
        result.update(result_dict)
        return result

    def set_config(self, config):
        config = load_config(config)

        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_prior = self.config["prior"]
        self.config_training = self.config["training"]
        self.config_state_manager = self.config["state_manager"]
        self.config_expression_decoder = self.config["controller"]
        self.config_experiment = self.config["experiment"]

    def save_config(self):
        # Save the config file
        if self.output_file is not None:
            path = os.path.join(self.config_experiment["save_path"], "config.json")
            # With run.py, config.json may already exist. To avoid race
            # conditions, only record the starting seed. Use a backup seed
            # in case this worker's seed differs.
            backup_seed = self.config_experiment["seed"]
            if not os.path.exists(path):
                if "starting_seed" in self.config_experiment:
                    self.config_experiment["seed"] = self.config_experiment["starting_seed"]
                    del self.config_experiment["starting_seed"]
                with open(path, 'w') as f:
                    cp_config = copy.copy(self.config)
                    json.dump(cp_config, f, indent=4)
            self.config_experiment["seed"] = backup_seed

    def set_seeds(self):
        """
        Set the tensorflow, numpy, and random module seeds based on the seed
        specified in config.  a time-based seed is used.
        """

        seed = round(time() * 1000) % int(1e9)
        self.config_experiment["seed"] = seed
        shifted_seed = seed

        # Set the seeds using the shifted seed
        tf.compat.v1.random.set_random_seed(shifted_seed)
        np.random.seed(shifted_seed)
        random.seed(shifted_seed)

    def make_pool_and_set_task(self):
        # Create the pool and set the Task for each worker

        # Set complexity and const optimizer here so pool can access them
        # Set the complexity function
        Program.set_complexity(self.config_training["complexity"])

        # Set the constant optimizer
        Program.const_optimizer = ScipyMinimize

        pool = None
        n_cores_batch = self.config_training.get("n_cores_batch")
        if n_cores_batch is not None:
            if n_cores_batch == -1:
                n_cores_batch = cpu_count()
            if n_cores_batch > 1:
                pool = Pool(n_cores_batch,
                            initializer=set_task,
                            initargs=(self.config_task,))

        # Set the Task for the parent process
        set_task(self.function_set, self.allowed_input_tokens, self.dataX, self.data_query_oracle, self.config_task)

        return pool

    def make_output_file(self):
        """Generates an output filename"""

        # If logdir is not provided (e.g. for pytest), results are not saved
        if self.config_experiment.get("logdir") is None:
            print("WARNING: logdir not provided. Results will not be saved to file.")
            return None

        # When using run.py, timestamp is already generated
        timestamp = self.config_experiment.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            self.config_experiment["timestamp"] = timestamp

        # Generate save path

        save_path = self.config_experiment["logdir"] + "_log/"
        self.config_experiment["task_name"] = self.task_name
        self.config_experiment["save_path"] = save_path
        os.makedirs(save_path, exist_ok=True)

        seed = self.config_experiment["seed"]
        output_file = os.path.join(save_path,
                                   "cvdso_{}_{}.csv".format(self.task_name, seed))

        return output_file
