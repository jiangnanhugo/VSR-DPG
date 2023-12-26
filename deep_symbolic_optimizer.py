"""Core deep symbolic optimizer construct."""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


from collections import defaultdict
from multiprocessing import Pool, cpu_count
import random
from time import time

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# from cvdso.task.regression import set_task
from expression_decoder import ExpressionDecoder
from train import learn
from utils import load_config
from state_manager import make_state_manager as manager_make_state_manager


class CVDeepSymbolicOptimizer(object):
    """
    Control variable for Deep symbolic optimization model. Includes model hyperparameters and
    training configuration.
    """

    def __init__(self, config=None, config_filename=None):
        """config : dict or str. Config dictionary or path to JSON."""
        # set config
        self.config_filename = config_filename
        self.set_config(config)
        self.sess = None


        self.config_task['batchsize'] = self.config_training['batch_size']



    def setup(self):
        # Clear the cache and reset the compute graph
        tf.compat.v1.reset_default_graph()
        # Generate objects needed for training and set seeds
        self.pool = self.make_pool_and_set_task()
        # set seeds
        seed = int(time.perf_counter() * 10000) % 1000007
        random.seed(seed)
        print('random seed=', seed)
        seed = int(time.perf_counter() * 10000) % 1000007
        np.random.seed(seed)
        seed = int(time.perf_counter() * 10000) % 1000007
        tf.compat.v1.random.set_random_seed(seed)

        self.sess = tf.compat.v1.Session()
        # Prepare training parameters
        self.state_manager = manager_make_state_manager(self.config_state_manager)
        self.expression_decoder = ExpressionDecoder(self.sess,
                                                    self.state_manager,
                                                    **self.config_expression_decoder)

    def train(self):
        # Train the model
        result = {"seed": self.config_experiment["seed"]}  # Seed listed first
        result_dict = learn(self.sess,
                            self.expression_decoder,
                            self.pool,
                            **self.config_training)
        result.update(result_dict)
        return result

    def set_config(self, config):
        config = load_config(config)

        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_training = self.config["training"]
        self.config_state_manager = self.config["state_manager"]
        self.config_expression_decoder = self.config["controller"]
        self.config_experiment = self.config["experiment"]


    def make_pool_and_set_task(self):
        # Create the pool and set the Task for each worker


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
        set_task(self.config_task)

        return pool
