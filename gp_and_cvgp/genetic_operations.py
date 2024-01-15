import numpy as np

class GPHelper(object):
    """
    Function class for genetic programming.

    Parameters
    ----------

    Methods
    -------
    mate(a, b): apply cross over of two program trees; find two subtrees
       within allowed_change_tokens then swap them

    """

    # static variables
    library = None

    def mate(self, a, b):
        """
            a and b are two program objects.
        """
        a_allowed = a.allow_change_pos()
        b_allowed = b.allow_change_pos()

        if len(a_allowed) == 0 or len(b_allowed) == 0:
            return

        # a_start = random.sample(a_allowed, 1)[0]
        # b_start = random.sample(b_allowed, 1)[0]
        a_start = np.random.choice(a_allowed)
        b_start = np.random.choice(b_allowed)

        a_end = a.subtree_end(a_start)
        b_end = b.subtree_end(b_start)

        # print('a.tokens=', a.tokens, 'a_start=', a_start, 'a_end=', a_end)
        # print('b.tokens=', b.tokens, 'b_start=', b_start, 'b_end=', b_end)

        na_tokens = np.concatenate((a.tokens[:a_start],
                                    b.tokens[b_start:b_end],
                                    a.tokens[a_end:]))
        nb_tokens = np.concatenate((b.tokens[:b_start],
                                    a.tokens[a_start:a_end],
                                    b.tokens[b_end:]))

        na_allow = np.concatenate((a.allow_change_tokens[:a_start],
                                   b.allow_change_tokens[b_start:b_end],
                                   a.allow_change_tokens[a_end:]))
        nb_allow = np.concatenate((b.allow_change_tokens[:b_start],
                                   a.allow_change_tokens[a_start:a_end],
                                   b.allow_change_tokens[b_end:]))

        a.__init__(na_tokens, na_allow)
        b.__init__(nb_tokens, nb_allow)
        a.remove_r_evaluate()
        b.remove_r_evaluate()

    def gen_full(self, maxdepth):
        """
            generate a full program tree recursively (represented in token indicies in library)
        """
        if maxdepth == 1:
            # t_idx = np.random.choice(self.library.tokens_of_arity[0])
            # while self.library.allowed_tokens[t_idx] == 0:
            #    t_idx = np.random.choice(self.library.tokens_of_arity[0])

            # more efficient implementation
            allowed_pos = [t for t in self.library.tokens_of_arity[0] \
                           if self.library.allowed_tokens[t] > 0]
            t_idx = np.random.choice(allowed_pos)
            return [t_idx]
        else:
            # t_idx = random.randint(0, self.library.L-1)
            # while self.library.allowed_tokens[t_idx] == 0:
            #    t_idx = random.randint(0, self.library.L-1)

            # more efficient implementation
            allowed_pos = self.library.allowed_tokens_pos()
            t_idx = np.random.choice(allowed_pos)

            arity = self.library.tokens[t_idx].arity
            tree = [t_idx]
            for i in range(arity):
                tree.extend(self.gen_full(maxdepth - 1))
            return tree

    def multi_mutate(self, individual, maxdepth):
        """Randomly select one of four types of mutation."""

        v = np.random.randint(0, 5)

        if v == 0:
            self.mutUniform(individual, maxdepth)
        elif v == 1:
            self.mutNodeReplacement(individual)
        elif v == 2:
            self.mutInsert(individual, maxdepth)
        elif v == 3:
            self.mutShrink(individual)

    def mutUniform(self, p, maxdepth):
        """
            find a leaf node (which allow_change_tokens == 1), replace the node with a gen_full
            tree of maxdepth.
        """
        leaf_set = []
        for i, token in enumerate(p.traversal):
            if p.allow_change_tokens[i] > 0 and token.arity == 0:
                leaf_set.append(i)
        if len(leaf_set) == 0:
            return
        t_idx = np.random.choice(np.array(leaf_set))

        new_tree = np.array(self.gen_full(maxdepth))

        np_tokens = np.concatenate((p.tokens[:t_idx], new_tree, p.tokens[(t_idx + 1):]))
        np_allow = np.insert(p.allow_change_tokens, t_idx, \
                             np.ones(len(new_tree) - 1, dtype=np.int32))

        p.__init__(np_tokens, np_allow)
        p.remove_r_evaluate()

    def mutNodeReplacement(self, p):
        """
            find a node and replace it with a node of the same arity
        """
        allowed_pos = p.allow_change_pos()
        if len(allowed_pos) == 0:
            return
        a_idx = allowed_pos[np.random.randint(0, len(allowed_pos))]
        arity = p.traversal[a_idx].arity

        # t_idx = np.random.choice(self.library.tokens_of_arity[arity])
        # while self.library.allowed_tokens[t_idx] == 0:
        #    t_idx = np.random.choice(self.library.tokens_of_arity[arity])
        # more efficient implementation
        allowed_pos = [t for t in self.library.tokens_of_arity[arity] \
                       if self.library.allowed_tokens[t] > 0]
        t_idx = np.random.choice(allowed_pos)

        p.tokens[a_idx] = t_idx
        p.__init__(p.tokens, p.allow_change_tokens)
        p.remove_r_evaluate()

    def mutInsert(self, p, maxdepth):
        """
            insert a node at a random position, the original subtree at the location
            becomes one of its subtrees.
        """
        insert_pos = np.random.randint(0, len(p.tokens))
        subtree_start = insert_pos
        subtree_end = p.subtree_end(subtree_start)
        # print('subtree_start=', subtree_start, 'subtree_end=', subtree_end)

        # generate the new root node

        # t_idx = random.randint(0, self.library.L-1)
        # while self.library.allowed_tokens[t_idx] == 0 or self.library.tokens[t_idx].arity == 0:
        #    t_idx = random.randint(0, self.library.L-1)

        # more efficient implementation
        non_term_allowed = self.library.allowed_non_terminal_tokens_pos()
        t_idx = np.random.choice(non_term_allowed)

        root_arity = self.library.tokens[t_idx].arity

        which_old_tree = np.random.randint(0, root_arity)

        # generate other subtrees
        np_tokens = np.concatenate((p.tokens[:subtree_start], np.array([t_idx])))
        np_allow = np.concatenate((p.allow_change_tokens[:subtree_start], np.array([1])))

        for i in range(root_arity):
            if i == which_old_tree:
                np_tokens = np.concatenate((np_tokens,
                                            p.tokens[subtree_start:subtree_end]))
                np_allow = np.concatenate((np_allow,
                                           p.allow_change_tokens[subtree_start:subtree_end]))
            else:
                subtree = np.array(self.gen_full(maxdepth - 1))
                np_tokens = np.concatenate((np_tokens, subtree))
                np_allow = np.insert(np_allow, -1, np.ones(len(subtree), dtype=np.int32))

        # add the rest
        np_tokens = np.concatenate((np_tokens, p.tokens[subtree_end:]))
        np_allow = np.concatenate((np_allow, p.allow_change_tokens[subtree_end:]))

        p.__init__(np_tokens, np_allow)
        p.remove_r_evaluate()

    def mutShrink(self, p):
        """
            delete a node (which allow_change_tokens == 1), use one of its child to replace
            its position.
        """
        allowed_pos = p.allow_change_pos()
        if len(allowed_pos) == 0:
            return
        a_idx = allowed_pos[np.random.randint(0, len(allowed_pos))]
        arity = p.traversal[a_idx].arity

        # print('arity=', arity)
        if arity == 0:
            # replace it with another node arity == 0 (perhaps not this node; but it is OK).
            self.mutNodeReplacement(p)
        else:
            a_end = p.subtree_end(a_idx)

            # get all the subtrees
            subtrees_start = []
            subtrees_end = []
            k = a_idx + 1
            while k < a_end:
                k_end = p.subtree_end(k)
                subtrees_start.append(k)
                subtrees_end.append(k_end)
                k = k_end

            # pick one of the subtrees, and re-assemble
            sp = np.random.randint(0, len(subtrees_start))

            np_tokens = np.concatenate((p.tokens[:a_idx],
                                        p.tokens[subtrees_start[sp]:subtrees_end[sp]],
                                        p.tokens[a_end:]))
            np_allow = np.concatenate((p.allow_change_tokens[:a_idx],
                                       p.allow_change_tokens[subtrees_start[sp]:subtrees_end[sp]],
                                       p.allow_change_tokens[a_end:]))
            p.__init__(np_tokens, np_allow)
            p.remove_r_evaluate()
