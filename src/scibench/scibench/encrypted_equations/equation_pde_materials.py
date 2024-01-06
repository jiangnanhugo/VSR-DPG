import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from base import KnownEquation, LogUniformSampling, IntegerUniformSampling, UniformSampling, LogUniformSampling2d

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
import os
from sympy import MatrixSymbol, Matrix, Symbol
from diff_ops import LaplacianOp, DifferentialOp

EQUATION_CLASS_DICT = OrderedDict()


def register_eq_class(cls):
    EQUATION_CLASS_DICT[cls.__name__] = cls
    return cls


def get_eq_obj(key, **kwargs):
    if key in EQUATION_CLASS_DICT:
        return EQUATION_CLASS_DICT[key](**kwargs)
    raise KeyError(f'`{key}` is not expected as a equation object key')


@register_eq_class
class Lorenz_dx(KnownEquation):
    _eq_name = 'Lorenz_dx'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6
    expr_consts_thres = None

    def __init__(self):
        self.sigma = 10
        self.beta = 8 / 3
        self.rho = 28
        self.dim = [1, 1, 1]

        vars_range_and_types = [LogUniformSampling(1e-2, 10.0, only_positive=True),
                                LogUniformSampling(1e-2, 10.0, only_positive=True),
                                LogUniformSampling(1e-2, 10.0, only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = self.sigma * (x[1] - x[0])


@register_eq_class
class Lorenz_dy(KnownEquation):
    _eq_name = 'Lorenz_dy'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.sigma = 10
        self.beta = 8 / 3
        self.rho = 28
        self.dim = [1, 1, 1]

        vars_range_and_types = [LogUniformSampling(1e-2, 10.0, only_positive=True),
                                LogUniformSampling(1e-2, 10.0, only_positive=True),
                                LogUniformSampling(1e-2, 10.0, only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = x[0] * (x[0] - self.rho - x[2])


@register_eq_class
class Lorenz_dz(KnownEquation):
    _eq_name = 'Lorenz_dz'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6
    expr_consts_thres = None

    def __init__(self):
        self.sigma = 10
        self.beta = 8 / 3
        self.rho = 28
        self.dim = [1, 1, 1]

        vars_range_and_types = [LogUniformSampling(1e-2, 10.0, only_positive=True),
                                LogUniformSampling(1e-2, 10.0, only_positive=True),
                                LogUniformSampling(1e-2, 10.0, only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=vars_range_and_types)
        x = self.x
        self.sympy_eq = x[0] * x[1] - self.beta * x[2]


@register_eq_class
class Glycolytic_oscillator_ds1(KnownEquation):
    _eq_name = 'Glycolytic_oscillator_ds1'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.J0 = 2.5
        self.k1 = 100
        self.k2 = 6
        self.k3 = 16
        self.k4 = 100
        self.k5 = 1.28
        self.k6 = 12
        self.K = 1.8
        self.k = 13
        self.q = 4
        self.K1 = 0.52
        self.phi = 0.1
        self.N = 1
        self.A = 4

        vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
                                LogUniformSampling(0.19, 2.16, only_positive=True),
                                LogUniformSampling(0.04, 0.20, only_positive=True),
                                LogUniformSampling(0.10, 0.35, only_positive=True),
                                LogUniformSampling(0.08, 0.30, only_positive=True),
                                LogUniformSampling(0.14, 2.67, only_positive=True),
                                LogUniformSampling(0.05, 0.10, only_positive=True)]
        super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = self.J0 - (self.k1 * x[0] * s[5]) / (1 + (self.x[5] / self.K1) ** self.q)


@register_eq_class
class Glycolytic_oscillator_ds1(KnownEquation):
    _eq_name = 'Glycolytic_oscillator_ds1'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.J0 = 2.5
        self.k1 = 100
        self.k2 = 6
        self.k3 = 16
        self.k4 = 100
        self.k5 = 1.28
        self.k6 = 12
        self.K = 1.8
        self.k = 13
        self.q = 4
        self.K1 = 0.52
        self.phi = 0.1
        self.N = 1
        self.A = 4

        vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
                                LogUniformSampling(0.19, 2.16, only_positive=True),
                                LogUniformSampling(0.04, 0.20, only_positive=True),
                                LogUniformSampling(0.10, 0.35, only_positive=True),
                                LogUniformSampling(0.08, 0.30, only_positive=True),
                                LogUniformSampling(0.14, 2.67, only_positive=True),
                                LogUniformSampling(0.05, 0.10, only_positive=True)]
        super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = self.J0 - (self.k1 * x[0] * s[5]) / (1 + (self.x[5] / self.K1) ** self.q)


@register_eq_class
class Glycolytic_oscillator_ds1(KnownEquation):
    _eq_name = 'Glycolytic_oscillator_ds1'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.J0 = 2.5
        self.k1 = 100
        self.k2 = 6
        self.k3 = 16
        self.k4 = 100
        self.k5 = 1.28
        self.k6 = 12
        self.K = 1.8
        self.k = 13
        self.q = 4
        self.K1 = 0.52
        self.phi = 0.1
        self.N = 1
        self.A = 4

        vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
                                LogUniformSampling(0.19, 2.16, only_positive=True),
                                LogUniformSampling(0.04, 0.20, only_positive=True),
                                LogUniformSampling(0.10, 0.35, only_positive=True),
                                LogUniformSampling(0.08, 0.30, only_positive=True),
                                LogUniformSampling(0.14, 2.67, only_positive=True),
                                LogUniformSampling(0.05, 0.10, only_positive=True)]
        super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = self.J0 - (self.k1 * x[0] * self.x[5]) / (1 + (self.x[5] / self.K1) ** self.q)


@register_eq_class
class Glycolytic_oscillator_ds2(KnownEquation):
    _eq_name = 'Glycolytic_oscillator_ds2'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.J0 = 2.5
        self.k1 = 100
        self.k2 = 6
        self.k3 = 16
        self.k4 = 100
        self.k5 = 1.28
        self.k6 = 12
        self.K = 1.8
        self.k = 13
        self.q = 4
        self.K1 = 0.52
        self.phi = 0.1
        self.N = 1
        self.A = 4

        vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
                                LogUniformSampling(0.19, 2.16, only_positive=True),
                                LogUniformSampling(0.04, 0.20, only_positive=True),
                                LogUniformSampling(0.10, 0.35, only_positive=True),
                                LogUniformSampling(0.08, 0.30, only_positive=True),
                                LogUniformSampling(0.14, 2.67, only_positive=True),
                                LogUniformSampling(0.05, 0.10, only_positive=True)]
        super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = 2 * (self.k1 * x[0] * self.x[5]) / (1 + (self.x[5] / self.K1) ** self.q) - self.k2 * self.x[1] * (self.N - self.x[4]) - \
                        self.k6 * self.x[1] * self.x[4]


@register_eq_class
class Glycolytic_oscillator_ds3(KnownEquation):
    _eq_name = 'Glycolytic_oscillator_ds3'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.J0 = 2.5
        self.k1 = 100
        self.k2 = 6
        self.k3 = 16
        self.k4 = 100
        self.k5 = 1.28
        self.k6 = 12
        self.K = 1.8
        self.k = 13
        self.q = 4
        self.K1 = 0.52
        self.phi = 0.1
        self.N = 1
        self.A = 4

        vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
                                LogUniformSampling(0.19, 2.16, only_positive=True),
                                LogUniformSampling(0.04, 0.20, only_positive=True),
                                LogUniformSampling(0.10, 0.35, only_positive=True),
                                LogUniformSampling(0.08, 0.30, only_positive=True),
                                LogUniformSampling(0.14, 2.67, only_positive=True),
                                LogUniformSampling(0.05, 0.10, only_positive=True)]
        super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = self.k2 * self.x[1] * (self.N - self.x[4]) - self.k3 * self.x[2] * (self.A - self.x[5])


@register_eq_class
class Glycolytic_oscillator_ds4(KnownEquation):
    _eq_name = 'Glycolytic_oscillator_ds4'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.J0 = 2.5
        self.k1 = 100
        self.k2 = 6
        self.k3 = 16
        self.k4 = 100
        self.k5 = 1.28
        self.k6 = 12
        self.K = 1.8
        self.kappa = 13
        self.q = 4
        self.K1 = 0.52
        self.phi = 0.1
        self.N = 1
        self.A = 4

        vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
                                LogUniformSampling(0.19, 2.16, only_positive=True),
                                LogUniformSampling(0.04, 0.20, only_positive=True),
                                LogUniformSampling(0.10, 0.35, only_positive=True),
                                LogUniformSampling(0.08, 0.30, only_positive=True),
                                LogUniformSampling(0.14, 2.67, only_positive=True),
                                LogUniformSampling(0.05, 0.10, only_positive=True)]
        super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = self.k3 * self.x[2] * (self.A - self.x[5]) - self.k4 * self.x[3] * self.x[4] - self.kappa * (self.x[3] - self.x[6])


@register_eq_class
class Glycolytic_oscillator_ds5(KnownEquation):
    _eq_name = 'Glycolytic_oscillator_ds5'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.J0 = 2.5
        self.k1 = 100
        self.k2 = 6
        self.k3 = 16
        self.k4 = 100
        self.k5 = 1.28
        self.k6 = 12
        self.K = 1.8
        self.k = 13
        self.q = 4
        self.K1 = 0.52
        self.phi = 0.1
        self.N = 1
        self.A = 4

        vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
                                LogUniformSampling(0.19, 2.16, only_positive=True),
                                LogUniformSampling(0.04, 0.20, only_positive=True),
                                LogUniformSampling(0.10, 0.35, only_positive=True),
                                LogUniformSampling(0.08, 0.30, only_positive=True),
                                LogUniformSampling(0.14, 2.67, only_positive=True),
                                LogUniformSampling(0.05, 0.10, only_positive=True)]
        super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = self.k2 * self.x[1] * (self.N - self.x[4]) - self.k4 * self.x[3] * self.x[4] - self.k6 * self.x[1] * self.x[4]


@register_eq_class
class Glycolytic_oscillator_ds6(KnownEquation):
    _eq_name = 'Glycolytic_oscillator_ds6'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.J0 = 2.5
        self.k1 = 100
        self.k2 = 6
        self.k3 = 16
        self.k4 = 100
        self.k5 = 1.28
        self.k6 = 12
        self.K = 1.8
        self.kappa = 13
        self.q = 4
        self.K1 = 0.52
        self.phi = 0.1
        self.N = 1
        self.A = 4

        vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
                                LogUniformSampling(0.19, 2.16, only_positive=True),
                                LogUniformSampling(0.04, 0.20, only_positive=True),
                                LogUniformSampling(0.10, 0.35, only_positive=True),
                                LogUniformSampling(0.08, 0.30, only_positive=True),
                                LogUniformSampling(0.14, 2.67, only_positive=True),
                                LogUniformSampling(0.05, 0.10, only_positive=True)]
        super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = -2 * (self.k1 * x[0] * self.x[5]) / (1 + (self.x[5] / self.K1) ** self.q) - 2 * self.k3 * self.x[2] * (
                self.A - self.x[5]) - self.k5 * self.x[5]


@register_eq_class
class Glycolytic_oscillator_ds7(KnownEquation):
    _eq_name = 'Glycolytic_oscillator_ds7'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6

    def __init__(self):
        self.J0 = 2.5
        self.k1 = 100
        self.k2 = 6
        self.k3 = 16
        self.k4 = 100
        self.k5 = 1.28
        self.k6 = 12
        self.K = 1.8
        self.kappa = 13
        self.q = 4
        self.K1 = 0.52
        self.phi = 0.1
        self.N = 1
        self.A = 4

        vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
                                LogUniformSampling(0.19, 2.16, only_positive=True),
                                LogUniformSampling(0.04, 0.20, only_positive=True),
                                LogUniformSampling(0.10, 0.35, only_positive=True),
                                LogUniformSampling(0.08, 0.30, only_positive=True),
                                LogUniformSampling(0.14, 2.67, only_positive=True),
                                LogUniformSampling(0.05, 0.10, only_positive=True)]
        super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
        x = self.x

        self.sympy_eq = self.phi * self.kappa * (self.x[3] - self.x[6]) - self.K * self.x[6]


@register_eq_class
class SpinodalDecomp64x64(KnownEquation):
    _eq_name = 'Spinodal_Decomposition_64x64'
    _function_set = ['add', 'sub', 'mul', 'div', 'clamp', 'laplacian', 'const']
    expr_obj_thres = 0.01
    expr_consts_thres = None
    simulated_exec = True

    def __init__(self):
        # super(SpinodalDecomp, self).__init__()
        # c is the input matrix; A, M, kappa is the constants in the expressions
        self.A = 1
        self.M = 1
        self.kappa = 0.5

        self.lap = LaplacianOp()
        self.diff = DifferentialOp()

        self.dx = 1
        self.dy = 1
        self.Nx = 64
        self.Ny = 64
        self.dim = [(self.Nx, self.Ny), ]

        self.dt = 1e-2

        vars_range_and_types = [LogUniformSampling2d(1e-3, 1.0, only_positive=True, dim=(self.Nx, self.Ny))]
        super().__init__(num_vars=1, vars_range_and_types=vars_range_and_types)
        self.x = [MatrixSymbol('X0', self.Nx, self.Ny)]
        c = self.x

        self.torch_func = self.forward
        self.sympy_eq = "EMPTY"
        ### consts = [0:1.0, 1:2.0, 2:A(1), 3:kappa(0.5), 4:dt(1e-2), 5:M(1)]
        consts = [1.0, 2.0, self.A, self.kappa, self.dt, self.M]

        ### tree1 = 2 * self.A * c * (1 - c) * (1 - 2*c)
        tree1 = [(2, "mul"), (0, 1), (2, "mul"), (0, 2), (2, "mul"), (1, 0), (2, "mul"), (2, "sub"), (0, 0), (1, 0), (2, "sub"),
                 (0, 0), (2, "mul"), (0, 1), (1, 0)]
        ### tree2 = self.kappa * self.lap(c, self.dx, self.dy)
        tree2 = [(2, "mul"), (0, 3), (2, "laplacian"), (1, 0)]
        # deltaF = 2 * self.A * c * (1-c) * (1-2*c) - self.kappa * self.lap(c, self.dx, self.dy)
        deltaF = [(2, "sub")]
        deltaF.extend(tree1)
        deltaF.extend(tree2)
        # dc = self.dt * self.lap(self.M*deltaF, self.dx, self.dy)
        dc = [(2, "mul"), (0, 4), (2, "laplacian"), (2, "mul"), (0, 5)]
        dc.extend(deltaF)
        # c_new = torch.clamp(c + dc, min=0.0001, max=0.9999)
        preorder_traversal = [(2, "clamp"), (2, "add"), (1, 0)]
        preorder_traversal.extend(dc)
        self.preorder_traversal = []
        for x in preorder_traversal:
            if x[0] == 0:
                self.preorder_traversal.append((consts[x[1]], "const"))
            elif x[0] == 1:
                self.preorder_traversal.append((str(c[x[1]]), "var"))
            elif x[0] == 2:
                if x[1] in ['mul', 'sub', 'div', 'add']:
                    self.preorder_traversal.append((x[1], "binary"))
                elif x[1] == "laplacian":
                    self.preorder_traversal.append(("laplacian", "unary"))
                elif x[1] == "clamp":
                    self.preorder_traversal.append((x[1], "unary"))

    def forward(self, c):
        # equation (4:18) + (4:17)
        deltaF = 2 * self.A * c * (1 - c) * (1 - 2 * c) - self.kappa * self.lap(c, self.dx, self.dy)
        # equation (4.16)
        dc = self.dt * self.lap(self.M * deltaF, self.dx, self.dy)
        # c_new = c+dc
        c_new = torch.clamp(c + dc, min=0, max=1)
        return c_new


# @register_eq_class
class GrainGrowth64x64(KnownEquation):
    _eq_name = 'Grain_Growth_64x64'
    _function_set = ['add', 'sub', 'mul', 'div', 'clamp', 'laplacian', 'const']
    expr_obj_thres = 0.01
    expr_consts_thres = None
    simulated_exec = True

    def __init__(self):
        self.A = 1  # np.random.randn(1)[0]
        self.M = 1  # np.random.randn(1)[0]
        # self.kappa = np.random.randn(1)[0]  # .to(device)

        self.lap = LaplacianOp()
        self.diff = DifferentialOp()

        self.dx = 0.5
        self.dy = 0.5
        self.Nx = 64
        self.Ny = 64
        self.dim = [(self.Nx, self.Ny), ]

        self.dt = 1e-2

        vars_range_and_types = [LogUniformSampling2d(1e-3, 1.0, only_positive=True, dim=(self.Nx, self.Ny)),
                                LogUniformSampling2d(1e-3, 1.0, only_positive=True, dim=(self.Nx, self.Ny))]
        self.x = [MatrixSymbol('X_0', self.Nx, self.Ny), MatrixSymbol('X_1', self.Nx, self.Ny)]
        super().__init__(num_vars=1, vars_range_and_types=vars_range_and_types)
        etas = self.x

        self.L = torch.tensor([5.0])  # nn.Parameter(torch.randn(1) * 5 + 0.1, requires_grad=True)
        self.kappa = torch.tensor(0.1)  # nn.Parameter(torch.randn(1) * 5 + 0.1, requires_grad=True)

        self.lap = LaplacianOp()

        # self.dtime = dt

        self.sympy_eq = "EMPTY"

    def forward(self, etas):
        n_grain = len(etas)
        all_new_etas = None
        sum_eta_2 = sum(eta ** 2 for eta in etas)

        etas_new = [None] * n_grain
        absL = torch.abs(self.L)
        absKappa = torch.abs(self.kappa)
        for i in range(0, n_grain):
            dfdeta = -self.A * etas[i] + self.B * (etas[i]) ** 3

            sq_sum = sum_eta_2 - (etas[i]) ** 2
            dfdeta += 2 * etas[i] * sq_sum

            lap_eta = self.lap(etas[i], self.dx, self.dy)

            term1 = dfdeta
            term1 = term1 - absKappa * lap_eta
            etas_new[i] = etas[i] - self.dtime * absL * (term1)

            fix_deviations(etas_new[i])
        return etas_new
