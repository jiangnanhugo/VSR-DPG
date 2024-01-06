import numpy as np


class RegressTask(object):
    """
    used to handle input data 'X' for querying the data oracle.
    also used to set the controlled variables in input data `X`
    """

    def __init__(self, dataset_size, n_vars, dataX, data_query_oracle, protected=False):
        """
            dataset_size: size of dataset.
            allowed_input: 1 if the input variable is free. 0 if the input variable is controlled.
            dataX: generate the input data.
            data_query_oracle: compute the output.
        """
        self.dataset_size = dataset_size

        self.dataX = dataX
        self.data_query_oracle = data_query_oracle
        #
        self.n_vars = n_vars
        # set of free variables
        self.vf = [0, ] * n_vars
        self.fixed_column = []
        self.protected = protected
        self.X_fixed = self.rand_draw_X_fixed()
        print(f"X_fixed: {self.X_fixed}")

    def set_vf(self, xi: int):
        """set of xi to be free variable
            vf[xi]=1: xi is free variable
            vf[xi]=0: xi is controlled variable
        """

        if 0 <= xi < len(self.vf):
            self.vf[xi] = 1
            print('xi is:', xi, ', new vf is:', self.vf)

    def get_vf(self):
        ''' get the free variables.'''
        return self.vf

    def set_allowed_inputs(self, allowed_inputs):
        self.allowed_input = np.copy(allowed_inputs)
        self.fixed_column = [i for i in range(self.n_vars) if self.vf[i] == 0]

    def set_allowed_input(self, xi: int, flag: int):
        self.allowed_input[xi] = flag
        self.fixed_column = [i for i in range(self.n_vars) if self.vf[i] == 0]

    def rand_draw_X_non_fixed(self):
        self.X = self.dataX.randn(sample_size=self.dataset_size).T

    def rand_draw_X_fixed(self):
        self.X_fixed = np.squeeze(self.dataX.randn(sample_size=1))

    def rand_draw_X_fixed_with_index(self, xi):
        X_fixed = np.squeeze(self.dataX.randn(sample_size=1))
        self.X_fixed[xi] = X_fixed[xi]
        if len(self.fixed_column):
            self.X[:, self.fixed_column] = self.X_fixed[self.fixed_column]

    def rand_draw_data_with_X_fixed(self):
        self.X = self.dataX.randn(sample_size=self.dataset_size).T
        if self.X_fixed is None:
            self.rand_draw_X_fixed()
        if len(self.fixed_column) > 0:
            self.X[:, self.fixed_column] = self.X_fixed[self.fixed_column]

    def evaluate(self):
        return self.data_query_oracle.evaluate(self.X)


    def reward_function(self, y_hat):

        return self.data_query_oracle._evaluate_loss(self.X, y_hat)
