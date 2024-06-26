"""Defines main training loop for deep symbolic optimization."""

import os

from itertools import compress

import tensorflow as tf
import numpy as np

from grammar.grammar import ContextSensitiveGrammar
from grammar.utils import empirical_entropy, weighted_quantile
from grammar.memory import Batch, make_queue
from grammar.variance import quantile_variance
import sys
# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


"""
  
    sess : tf.Session. TensorFlow Session object.
    expression_decoder : ExpressionDecoder. ExpressionDecoder object used to generate Programs.

    n_epochs : int or None, optional. Number of epochs to train when n_samples is None.

    n_samples : int or None, optional
        Total number of expressions to sample when n_epochs is None. In this
        case, n_epochs = int(n_samples / batch_size).
    batch_size : int, optional. Number of sampled expressions per epoch.
    complexity : str, optional. Complexity function name, used computing Pareto front.
    alpha : float, optional.  Coefficient of exponentially-weighted moving average of baseline.
    epsilon : float or None, optional.  Fraction of top expressions used for training. None (or equivalently, 1.0) turns off risk-seeking.


    verbose : bool, optional. Whether to print progress.
    save_summary : bool, optional. Whether to write TensorFlow summaries.
    save_all_epoch : bool, optional. Whether to save all rewards for each iteration.

    baseline : str, optional
        Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
        Choices:
        (1) "ewma_R" : b = EWMA(<R>)
        (2) "R_e" : b = R_e
        (3) "ewma_R_e" : b = EWMA(R_e)
        (4) "combined" : b = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the (1-epsilon)-quantile estimate.

    b_jumpstart : bool, optional
        Whether EWMA part of the baseline starts at the average of the first
        iteration. If False, the EWMA starts at 0.0.

    early_stopping : bool, optional. Whether to stop early if stopping criteria is reached.
    hof : int or None, optional. If not None, number of top Programs to evaluate after training.

    eval_all : bool, optional
        If True, evaluate all Programs. While expensive, this is useful for
        noisy data when you can't be certain of success solely based on reward.
        If False, only the top Program is evaluated each iteration.

    save_pareto_front : bool, optional. If True, compute and save the Pareto front at the end of training.
    debug : int. Debug level, also passed to Controller. 0: No debug. 1: Print initial parameter means. 2: Print parameter means each step.
    use_memory : bool, optional. If True, use memory queue for reward quantile estimation.
    memory_capacity : int. Capacity of memory queue.

    warm_start : int or None
        Number of samples to warm start the memory queue. If None, uses
        batch_size.

    memory_threshold : float or None
        If not None, run quantile variance/bias estimate experiments after
        memory weight exceeds memory_threshold.

    save_positional_entropy : bool, optional.  Whether to save evolution of positional entropy for each iteration.

    save_top_samples_per_batch : float, optional. Whether to store X% top-performer samples in every batch.
    save_cache : bool. Whether to save the str, count, and r of each Program in the cache.

    save_cache_r_min : float or None
        If not None, only keep Programs with r >= r_min when saving cache.

    save_freq : int or None. Statistics are flushed to file every save_freq epochs (default == 1). If < 0, uses save_freq = inf
    save_token_count : bool. Whether to save token counts each batch.

    Return : dict. A dict describing the best-fit expression (determined by reward).
    """


def learn(grammar_model: ContextSensitiveGrammar,
          sess, expression_decoder,
          n_epochs=12, dataset_size=None, batch_size=1000,
          reward_threshold=0.999999,
          alpha=0.5, epsilon=0.05, verbose=True, baseline="R_e",
          b_jumpstart=False, early_stopping=True,
          debug=0, use_memory=False, memory_capacity=1e3,
          warm_start=None, memory_threshold=None, save_positional_entropy=False,
          save_top_samples_per_batch=0):
    """
      Executes the main training loop.
    """
    print(dataset_size, n_epochs)
    # Initialize compute graph
    sess.run(tf.compat.v1.global_variables_initializer())

    # Create the priority queue
    k = expression_decoder.pqt_k
    if expression_decoder.pqt and k is not None and k > 0:
        priority_queue = make_queue(priority=True, capacity=k)
    else:
        priority_queue = None

    # Create the memory queue
    if use_memory:
        assert epsilon is not None and epsilon < 1.0, "Memory queue is only used with risk-seeking."
        memory_queue = make_queue(expression_decoder=expression_decoder, priority=False,
                                  capacity=int(memory_capacity))

        # Warm start the queue
        warm_start = warm_start if warm_start is not None else batch_size
        actions, obs = expression_decoder.sample(warm_start)
        print("sampled actions:", actions)
        # construct program based on the input token indices


        grammar_expressions = grammar_model.construct_expression(actions)
        rewards = np.array([p.r for p in grammar_expressions])
        expr_lengths = np.array([len(p.traversal.split(";")) for p in grammar_expressions])
        sampled_batch = Batch(actions=actions, obs=obs,
                              lengths=expr_lengths, rewards=rewards)
        memory_queue.push_batch(sampled_batch, grammar_expressions)
    else:
        memory_queue = None

    if debug >= 1:
        print("\nInitial parameter means:")
        print_var_means(sess)

    # Main training loop
    p_final = None
    r_best = -np.inf
    prev_r_best = None
    ewma = None if b_jumpstart else 0.0  # EWMA portion of baseline
    nevals = 0  # Total number of sampled expressions (from RL)
    positional_entropy = np.zeros(shape=(n_epochs, expression_decoder.max_length), dtype=np.float32)

    top_samples_per_batch = list()
    print("-- RUNNING EPOCHS START -------------")
    sys.stdout.flush()
    for epoch in range(n_epochs):
        # Set of str representations for all Programs ever seen
        # s_history = set(grammar_model.program.cache.keys())

        # Sample batch of expressions from the expression_decoder
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: [(batch_size, max_length)] * 3
        actions, obs = expression_decoder.sample(batch_size)
        # if verbose:
        #     print("sampled actions:", actions)
        grammar_expressions = grammar_model.construct_expression(actions)
        nevals += batch_size
        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.reward for p in grammar_expressions])
        # if verbose:
        #     print("rewards:", r)
        r_train = r

        # Need for Vanilla Policy Gradient (epsilon = null)
        p_train = grammar_expressions

        if save_positional_entropy:
            positional_entropy[epoch] = np.apply_along_axis(empirical_entropy, 0, actions)

        if save_top_samples_per_batch > 0:
            # sort in descending order: larger rewards -> better solutions
            sorted_idx = np.argsort(r)[::-1]
            one_perc = int(len(grammar_expressions) * float(save_top_samples_per_batch))
            for idx in sorted_idx[:one_perc]:
                top_samples_per_batch.append([epoch, r[idx], repr(grammar_expressions[idx])])

        # Update HOF
        for p in grammar_expressions:
            if not p.reward:
                continue
            grammar_model.update_hall_of_fame(p)

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_max = np.max(r)
        r_best = max(r_max, r_best)

        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with lesser reward.
        """
        if epsilon is not None and epsilon < 1.0:
            # Compute reward quantile estimate
            if use_memory:  # Memory-augmented quantile
                # Get subset of Programs not in buffer
                unique_programs = [p for p in grammar_expressions if p.traversal not in memory_queue.unique_items]
                N = len(unique_programs)

                # Get rewards
                memory_r = memory_queue.get_rewards()
                sample_r = [p.r for p in unique_programs]
                combined_r = np.concatenate([memory_r, sample_r])

                # Compute quantile weights
                memory_w = memory_queue.compute_probs()
                if N == 0:
                    print("WARNING: Found no unique samples in batch!")
                    combined_w = memory_w / memory_w.sum()  # Renormalize
                else:
                    sample_w = np.repeat((1 - memory_w.sum()) / N, N)
                    combined_w = np.concatenate([memory_w, sample_w])

                # Quantile variance/bias estimates
                if memory_threshold is not None:
                    print("Memory weight:", memory_w.sum())
                    if memory_w.sum() > memory_threshold:
                        quantile_variance(memory_queue, expression_decoder, batch_size, epsilon, epoch)

                # Compute the weighted quantile
                quantile = weighted_quantile(values=combined_r, weights=combined_w, q=1 - epsilon)

            else:  # Empirical quantile
                quantile = np.quantile(r, 1 - epsilon, interpolation="higher")

            '''
                Here we get the returned as well as stored programs and properties.

            '''

            keep = r >= quantile

            r_train = r = r[keep]
            p_train = grammar_expressions = list(compress(grammar_expressions, keep))

            '''
                get the action, observation status of all programs returned to the controller.
            '''
            actions = actions[keep, :]
            obs = obs[keep, :, :]

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)
        r_train = np.clip(r_train, -1e6, 1e6)
        print("r_train shape:", r_train.shape)
        sys.stdout.flush()
        # Compute baseline
        # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
        if baseline == "ewma_R":
            ewma = np.mean(r_train) if ewma is None else alpha * np.mean(r_train) + (1 - alpha) * ewma
            b_train = ewma
        elif baseline == "R_e":  # Default
            ewma = -1
            b_train = quantile
        elif baseline == "ewma_R_e":
            ewma = np.min(r_train) if ewma is None else alpha * quantile + (1 - alpha) * ewma
            b_train = ewma
        elif baseline == "combined":
            ewma = np.mean(r_train) - quantile if ewma is None else alpha * (np.mean(r_train) - quantile) + (1 - alpha) * ewma
            b_train = quantile + ewma

        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), expression_decoder.max_length) for p in p_train], dtype=np.int32)

        # Create the Batch
        sampled_batch = Batch(actions=actions, obs=obs,
                              lengths=lengths, rewards=r_train)

        # Update and sample from the priority queue
        if priority_queue is not None:
            priority_queue.push_best(sampled_batch, grammar_expressions)
            pqt_batch = priority_queue.sample_batch(expression_decoder.pqt_batch_size)
        else:
            pqt_batch = None
        summaries = expression_decoder.train_step(b_train, sampled_batch, pqt_batch)

        # Update the memory queue
        if memory_queue is not None:
            memory_queue.push_batch(sampled_batch, grammar_expressions)

        # Update new best expression
        new_r_best = False

        if prev_r_best is None or r_max > prev_r_best:
            new_r_best = True
            p_r_best = grammar_expressions[np.argmax(r)]

        prev_r_best = r_best

        # Print new best expression
        if verbose and new_r_best:
            print(" Training epoch {}/{}, current best R: {}".format(epoch + 1, n_epochs, prev_r_best))
            print(f"{p_r_best}")

            print("\t** New best")

        if early_stopping and p_r_best.reward > reward_threshold:
            print("Early stopping criteria met; breaking early.")
            break

        if verbose and (epoch + 1) % 10 == 0:
            print("Training epoch {}/{}, current best R: {:.4f}".format(epoch + 1, n_epochs, prev_r_best))

        if debug >= 2:
            print("\nParameter means after epoch {} of {}:".format(epoch + 1, n_epochs))
            print_var_means(sess)

        if verbose and (epoch + 1) == n_epochs:
            print("Ending training after epoch {}/{}, current best R: {:.4f}".format(epoch + 1, n_epochs,
                                                                                     prev_r_best))

        grammar_model.print_hofs(mode='global', verbose=True)

    # Print the priority queue at the end of training
    if verbose and priority_queue is not None:
        for i, item in enumerate(priority_queue.iter_in_order()):
            print("\nPriority queue entry {}:".format(i))
            p = grammar_model.program.cache[item[0]]
            p.print_stats()

    # Return the best expression
    p = p_final if p_final is not None else p_r_best
    print(p)
    sys.stdout.flush()
    return [p]


def print_var_means(sess):
    tvars = tf.compat.v1.trainable_variables()
    tvars_vals = sess.run(tvars)
    for var, val in zip(tvars, tvars_vals):
        print(var.name, "mean:", val.mean(), "var:", val.var())
        # print(val)
