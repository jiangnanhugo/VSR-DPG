"""Core deep symbolic optimizer construct."""

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from collections import defaultdict
from multiprocessing import Pool, cpu_count
import random
import time

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from expression_decoder import NeuralExpressionDecoder
from train import learn
from utils import load_config
from state_manager import make_state_manager as manager_make_state_manager


class VSRDeepSymbolicRegression(object):
    """
    Control variable for Deep symbolic optimization model. Includes model hyperparameters and
    training configuration.
    """

    def __init__(self, config, config_filename, cfg):
        """config : dict or str. Config dictionary or path to JSON.
        cfg: context-sensitive-grammar
        """
        # set config
        self.config_filename = config_filename
        self.set_config(config)
        self.sess = None
        self.cfg = cfg
        self.config_task['batchsize'] = self.config_training['batch_size']

    def setup(self):
        # Clear the cache and reset the compute graph
        tf.compat.v1.reset_default_graph()
        # Generate objects needed for training and set seeds
        self.pool = None  # self.make_pool_and_set_task()
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

        self.expression_decoder = NeuralExpressionDecoder(self.cfg,
                                                          self.sess,
                                                          self.state_manager,
                                                          **self.config_expression_decoder)

    def train(self):
        # Train the model
        print("extra arguments:\n {}".format(self.config_training))

        result_dict = learn(self.cfg,
                            self.sess,
                            self.expression_decoder,
                            self.pool,
                            **self.config_training)

        return result_dict

    def set_config(self, config):
        config = load_config(config)
        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_training = self.config["training"]
        self.config_state_manager = self.config["state_manager"]
        self.config_expression_decoder = self.config["expression_decoder"]

