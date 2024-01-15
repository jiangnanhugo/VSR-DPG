import time
import numpy as np
from operator import attrgetter

from program import Program
from utils import print_prs


class GeneticProgram(object):
    """
    Parameters
    ----------
    cxpb: probability of mate
    mutpb: probability of mutations
    maxdepth: the maxdepth of the tree during mutation
    population_size: the size of the selected populations (at the end of each generation)
    tour_size: the size of the tournament for selection
    hof_size: the size of the best programs retained

    Variables
    ---------
    population: the current list of programs
    hof: list of the best programs
    timer_log: list of times
    gen_num: number of generations, starting from 0.

    """

    # static variables
    library = None
    gp_helper = None

    def __init__(self, cxpb, mutpb, maxdepth, population_size, tour_size, hof_size,
                 n_generations):
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.maxdepth = maxdepth
        self.population_size = population_size
        self.tour_size = tour_size
        self.hof_size = hof_size
        self.n_generations = n_generations

        self.hof = []
        self.timer_log = []
        self.gen_num = 0
        self.create_init_population()

    def run(self, print_freq=5, verbose=True):
        # run for n generations
        most_recent_timestamp = time.perf_counter()
        for i in range(1, self.n_generations + 1):
            print(f'++++++++++++++++++ ITERATION {i} ++++++++++++++++++')
            self.one_generation(verbose)

            now_time_stamp = time.perf_counter()
            if now_time_stamp - most_recent_timestamp >= 900:  # 15 min
                print('running {} more than 15mins'.format(i))
                # self.print_hof()
                # print("")
                most_recent_timestamp = now_time_stamp
            if i % print_freq == 0:
                self.print_hof()

    def one_generation(self, verbose=True):
        """
        One step of the genetic algorithm.
        This wraps selection, mutation, crossover and hall of fame computation
        over all the individuals in the population for this epoch/step.

        Parameters
        ----------
        iter : int
            The current iteration used for logging purposes.

        """
        t1 = time.perf_counter()

        # Selection the next generation individuals
        offspring = self.selectTournament(self.population_size, self.tour_size)
        if verbose:
            print('offspring after select=')
            print_prs(offspring)
            print("")

        # Vary the pool of individuals
        # the crossover and mutation.
        offspring = self._var_and(offspring)
        if verbose:
            print('offspring after _var_and=')
            print_prs(offspring)
            print("")

        # Replace the current population by the offspring
        self.population = offspring + self.hof

        # Update hall of fame
        self.update_hof()

        timer = time.perf_counter() - t1

        self.timer_log.append(timer)
        self.gen_num += 1

    def update_hof(self):
        new_hof = sorted(self.population, reverse=True, key=attrgetter('r'))

        self.hof = []
        for i in range(self.hof_size):
            new_hofi = new_hof[i].clone()
            self.hof.append(new_hofi)

    def update_population(self):
        new_population = sorted(self.population, reverse=True, key=attrgetter('r'))
        self.population = []
        for i in range(self.population_size):
            self.population.append(new_population[i].clone())

    def selectTournament(self, population_size, tour_size):
        offspring = []
        # higher fitness score has higher chance to be survive in the next generation.
        for pp in range(population_size):
            # random sample  tor_size number of individual
            if len(self.population) <= tour_size:
                spr = self.population
            else:
                spr = np.random.choice(self.population, tour_size)
            # select the guys has the highest fit
            maxspr = max(spr, key=attrgetter('r'))
            # maxspri = copy.deepcopy(maxspr)
            # if "expr_objs" in maxspr.__dict__:
            #     maxspri.expr_objs = np.copy(maxspr.expr_objs)
            # if "expr_consts" in maxspr.__dict__:
            #     maxspri.expr_consts = np.copy(maxspr.expr_consts)
            maxspri = maxspr.clone()
            offspring.append(maxspri)
            # offspring may have duplicates,
        return offspring

    def _var_and(self, offspring):
        """
        Apply crossover AND mutation to each individual in a population
        given a constant probability.
        """
        # offspring = [copy.deepcopy(pr) for pr in self.population]
        np.random.shuffle(offspring)
        # Apply crossover on the offspring
        for i in range(1, len(offspring), 2):
            if np.random.random() < self.cxpb:
                self.gp_helper.mate(offspring[i - 1],
                                    offspring[i])

        # Apply mutation on the offspring
        for i in range(len(offspring)):
            if np.random.random() < self.mutpb:
                # for everyone you randomly mutate them.
                self.gp_helper.multi_mutate(offspring[i], self.maxdepth)

        return offspring

    def create_init_population(self):
        """
           create the initial population; look for every token in library, fill in
           the leaves with constants or inputs.
        """
        self.population = []
        for i, t in enumerate(self.library.tokens):
            if self.library.allowed_tokens[i]:
                # otherwise (not allowed) do not need to do anything
                tree = [i]
                for j in range(t.arity):
                    t_idx = np.random.choice(self.library.tokens_of_arity[0])
                    while self.library.allowed_tokens[t_idx] == 0:
                        t_idx = np.random.choice(self.library.tokens_of_arity[0])
                    tree.append(t_idx)
                tree = np.array(tree)

                pr = Program(tree, np.ones(tree.size, dtype=np.int32))
                self.population.append(pr)

        # self.hof = [copy.deepcopy(pr) for pr in self.population]
        self.hof = []
        for pr in self.population:
            new_pr = pr.clone()
            self.hof.append(new_pr)

    def print_population(self):
        for pr in self.population:
            print(pr.__getstate__())

    def print_hof(self):
        new_hof = sorted(self.hof, reverse=False, key=attrgetter('r'))
        for pr in new_hof:
            print(pr.__getstate__())
            pr.task.rand_draw_X_non_fixed()
            print('validate r=', pr.task.reward_function(pr))
            pr.task.print_reward_function_all_metrics(pr)
            pr.print_expression()
