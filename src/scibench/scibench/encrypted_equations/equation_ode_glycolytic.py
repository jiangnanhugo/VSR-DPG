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
class Lorenz(KnownEquation):
    _eq_name = 'Lorenz'
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

        self.sympy_eq = [
            self.sigma * (x[1] - x[0]),
            x[0] * (x[0] - self.rho - x[2]),
            x[0] * x[1] - self.beta * x[2]
        ]


@register_eq_class
class Glycolytic_oscillator(KnownEquation):
    _eq_name = 'Glycolytic_oscillator'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'n3', 'n4', 'const']
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

        self.sympy_eq = [
            self.J0 - (self.k1 * x[0] * x[5]) / (1 + (x[5] / self.K1) ** self.q),
            2 * (self.k1 * x[0] * x[5]) / (1 + (x[5] / self.K1) ** self.q) - self.k2 * x[1] * (self.N - x[4]) - self.k6 * x[1] * x[4],
            self.k2 * x[1] * (self.N - x[4]) - self.k3 * x[2] * (self.A - x[5]),
            self.k3 * x[2] * (self.A - x[5]) - self.k4 * x[3] * x[4] - self.kappa * (x[3] - x[6]),
            self.k2 * x[1] * (self.N - x[4]) - self.k4 * x[3] * x[4] - self.k6 * x[1] * x[4],
            -2 * self.k1 * x[0] * x[5] / (1 + (x[5] / self.K1) ** self.q) + 2 * self.k3 * x[2] * (self.A - x[5]) - self.k5 * x[5],
            self.phi * self.kappa * (x[3] - x[6]) - self.K * x[6]
        ]

#
# # ---------------------
# # https://arxiv.org/pdf/2003.07140.pdf
# @register_eq_class
# class Glycolytic_a_simple_reaction_network(KnownEquation):
#     _eq_name = 'Glycolytic_a_simple_reaction_network'
#     _function_set = ['add', 'sub', 'mul', 'div', 'const']
#     expr_obj_thres = 1e-6
#
#     def __init__(self):
#         v1 = 2.5
#         k1 = 100
#         k_neg1 = 6
#
#         vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
#                                 LogUniformSampling(0.19, 2.16, only_positive=True),
#                                 LogUniformSampling(0.04, 0.20, only_positive=True),
#                                 LogUniformSampling(0.10, 0.35, only_positive=True), ]
#         super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
#         x = self.x
#         # v1 − k1s1x1 + k−1x2,
#         self.sympy_eqs = [
#             v1 - self.k1 * x[0] * x[2] + k_neg1 * x[3],
#             self.k2*x[1]-gamma*k3*s2**gamma*np.e+gmma *k_neg3*x[0] ]


# # %Lotka–Volterra equation
# # s
# @register_eq_class
# class Lotka_Volterra(KnownEquation):
#     _eq_name = 'Lotka–Volterra'
#     _function_set = ['add', 'sub', 'mul', 'div', 'const']
#     expr_obj_thres = 1e-6
#
#     def __init__(self):
#         # https://ulissigroup.cheme.cmu.edu/math-methods-chemical-engineering/notes/ordinary_differential_equations/19-linear-stability.html
#         alpha = 1
#         beta = 0.2
#         delta = 0.5
#         gamma = 0.2
#
#         vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
#                                 LogUniformSampling(0.19, 2.16, only_positive=True),
#                                 LogUniformSampling(0.04, 0.20, only_positive=True),
#                                 LogUniformSampling(0.10, 0.35, only_positive=True), ]
#         super().__init__(num_vars=7, vars_range_and_types=vars_range_and_types)
#         x = self.x
#
#         self.sympy_eqs = [alpha * x[0] - beta * x[0] * x[1],
#                           delta * x[0] * x[1] - gamma * x[1]]


# # %Competitive Lotka–Volterra equation
# # https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations
#         # https://arxiv.org/pdf/2303.04919.pdf
# @register_eq_class
# class Competitive_Lotka_Volterra(KnownEquation):
#     _eq_name = 'Competitive-Lotka–Volterra'
#     _function_set = ['add', 'sub', 'mul', 'div', 'const']
#     expr_obj_thres = 1e-6
#
#     def __init__(self, N):
#         #  N is the total number of interacting species.
#         #  For simplicity all self-interacting terms αii are often set to 1.
#         alpha = np.random.rand(N, N)
#         r = np.random.rand(N)
#         np.fill_diagonal(alpha, 1.0)
#         #
#         # vars_range_and_types = [LogUniformSampling(0.15, 1.6, only_positive=True),
#         #                         LogUniformSampling(0.19, 2.16, only_positive=True),
#         #                         LogUniformSampling(0.04, 0.20, only_positive=True),
#         #                         LogUniformSampling(0.10, 0.35, only_positive=True), ]
#         super().__init__(num_vars=N)
#         x = self.x
#
#         self.sympy_eqs = []
#         for i in range(N):
#             summand = [alpha[i, j] * x[j] for j in range(N)]
#             self.sympy_eqs.append(r[i] * x[i] * (1 - sum(summand)))
