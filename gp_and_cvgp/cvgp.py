# import random
import time
import numpy as np
from operator import attrgetter

from program import Program


from utils import print_prs

def create_geometric_generations(n_generations, nvar):
    gens = [0] * nvar
    for it in range(nvar - 1, 0, -1):
        gens[it] = n_generations // 2
        n_generations -= gens[it]
    gens[0] = n_generations
    for it in range(0, nvar):
        if gens[it] < 50:
            gens[it] = 50
    print('generation #:', gens, 'sum=', sum(gens))
    return gens


def create_uniform_generations(n_generations, nvar):
    gens = [0] * nvar
    each_gen = n_generations // nvar
    for it in range(nvar - 1, 0, -1):
        gens[it] = each_gen
        n_generations -= each_gen
    gens[0] = n_generations
    print('generation #:', gens, 'sum=', sum(gens))
    return gens


class ExpandingGeneticProgram(object):
    # the main idea.
    """
    Parameters
    ----------
    cxpb: probability of mate
    mutpb: probability of mutations
    maxdepth: the maxdepth of the tree during mutation
    population_size: the size of the selected populations (at the end of each generation)
    tour_size: the size of the tournament for selection
    hof_size: the size of the best programs retained
    nvar: number of variables

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
                 n_generations, nvar):
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.maxdepth = maxdepth
        self.population_size = population_size
        self.tour_size = tour_size
        self.hof_size = hof_size

        # self.n_generations = create_geometric_generations(n_generations, nvar)
        self.n_generations = create_uniform_generations(n_generations, nvar + 1)

        self.hof = []
        self.population = []

        self.timer_log = []
        self.gen_num = 0

        self.nvar = nvar
        assert self.library != None
        assert Program.task != None

        start_allowed_input_tokens = np.zeros(nvar, dtype=np.int32)
        start_allowed_input_tokens[0] = 1
        self.library.set_allowed_input_tokens(start_allowed_input_tokens)

        Program.task.set_allowed_inputs(start_allowed_input_tokens)

        # self.create_init_population(nvar=0)

    def run(self):
        # for
        # most_recent_timestamp = time.perf_counter()
        for var in range(self.nvar + 1):
            self.create_init_population()

            # consider one variable at a time.
            for pr in self.population:
                # once we fix it, we want to remove r so that it won't change.
                pr.remove_r_evaluate()
            for pr in self.hof:
                pr.remove_r_evaluate()

            for pr in self.population:
                # a cached property in python (evaluated once)
                # forth the function to evaluate a new r
                thisr = pr.r  # how good you fit. the inverse of the residual.
            for pr in self.hof:
                thisr = pr.r

            # for fixed variable, do n generation
            for i in range(self.n_generations[var]):
                print('++++++++++++ VAR {} ITERATION {} ++++++++++++'.format(var, i))
                self.one_generation()
            self.update_population()

            for i, pr in enumerate(self.population):
                print('{}-th in self.population'.format(i))
                # evaluate r again, just incase it has not been evaluated.
                this_r = pr.r
                if len(pr.const_pos) == 0 or pr.num_changing_consts == 0:
                    # only expand at those constant node. if there are no constant node,then we are done
                    # if we do not want num_changing_consts, then we also quit.
                    print('there are no constant node. we are done...')

                else:
                    if not ("expr_objs" in pr.__dict__ and "expr_consts" in pr.__dict__):
                        print('WARNING: pr.expr_objs NOT IN DICT: pr=' + str(pr.__getstate__()))
                        pr.remove_r_evaluate()
                        this_r = pr.r
                        print('pr.expr_objs=', pr.expr_objs)
                        print('pr.expr_consts=', pr.expr_consts)
                # whether you get very different value for different constant.
                pr.freeze_equation()
                pr.remove_r_evaluate()

                print('pr.r=', pr.r)
                print('pr=', pr.__getstate__())

                pr.print_expression()

            for i, pr in enumerate(self.hof):
                print('{}-th in self.hof'.format(i))
                # evaluate r again, just incase it has not been evaluated.
                pr.remove_r_evaluate()
                print('pr.r=', pr.r)
                print('pr=', pr.__getstate__())

                pr.print_expression()

            if var < self.nvar - 1:
                # previous we only change x0,
                # the next round, we are not allow to change x0.
                self.library.set_allowed_input_token(var, 0)  # XYX commented out this on Jan 16; trying to run noisy experiments.
                self.library.set_allowed_input_token(var + 1, 1)
                # set the next variable to be free
                Program.task.set_allowed_input(var + 1, 1)

    def one_generation(self):
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

        # Select the next generation individuals
        offspring = self.selectTournament(self.population_size, self.tour_size)

        print('offspring after select=')
        print_prs(offspring)
        print("")

        # Vary the pool of individuals
        offspring = self._var_and(offspring)

        print('offspring after _var_and=')
        print_prs(offspring)
        print("")

        # Replace the current population by the offspring
        self.population = offspring + self.hof + self.population

        # Update hall of fame
        self.update_hof()
        print("after update hof after sorted=")
        print_prs(self.hof)
        timer = time.perf_counter() - t1

        self.timer_log.append(timer)
        self.gen_num += 1

    def update_hof(self):
        new_hof = sorted(self.population, reverse=True, key=attrgetter('r'))

        self.hof = []
        for i in range(self.hof_size):
            pr = new_hof[i]
            if pr.r == np.nan or pr.r == np.inf or pr.r == -np.inf:
                print("filter:", pr.r, pr.__getstate__(), end="\t")
                pr.print_expression()
                continue
            self.hof.append(pr.clone())

        # self.hof = [new_hof[i].clone() for i in range(self.hof_size)]

    def update_population(self):
        filtered_population = []
        for pr in self.population:
            if pr.r == np.nan or pr.r == np.inf or pr.r == -np.inf:
                print("filter:", pr.r, pr.__getstate__(), end="\t")
                pr.print_expression()
                continue
            filtered_population.append(pr)
        new_population = sorted(filtered_population, reverse=True, key=attrgetter('r'))
        self.population = []
        for i in range(min(self.population_size, len(filtered_population))):
            self.population.append(new_population[i].clone())

    def selectTournament(self, population_size, tour_size):
        offspring = []
        for pp in range(population_size):
            if len(self.population) <= tour_size:
                spr = self.population
            else:
                spr = np.random.choice(self.population, tour_size)
            maxspr = max(spr, key=attrgetter('r'))
            maxspri = maxspr.clone()
            offspring.append(maxspri)
        return offspring

    def _var_and(self, offspring):
        """
        Apply crossover AND mutation to each individual in a population 
        given a constant probability. 
        """

        # Apply crossover on the offspring
        np.random.shuffle(offspring)
        for i in range(1, len(offspring), 2):
            if np.random.random() < self.cxpb:
                self.gp_helper.mate(offspring[i - 1], offspring[i])

        # Apply mutation on the offspring
        for i in range(len(offspring)):
            if np.random.random() < self.mutpb:
                self.gp_helper.multi_mutate(offspring[i], self.maxdepth)

        return offspring

    def create_init_population(self):
        """
           create the initial population; look for every token in library, fill in
           the leaves with constants or inputs.
        """

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


