import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from base import KnownEquation, LogUniformSampling, IntegerUniformSampling, UniformSampling, LogUniformSampling2d

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
import os
from sympy import Float
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
class spin_glass_N4_M2_1(KnownEquation):
    _eq_name = 'spin_glass_N4_M2_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6
    expr_consts_thres = None

    def __init__(self):
        self.dim = [1, 1, 1]

        vars_range_and_types = [IntegerUniformSampling(8, 9, only_positive=True),
                                IntegerUniformSampling(2, 3, only_positive=True),
                                IntegerUniformSampling(4, 5, only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=vars_range_and_types)
        self.sympy_eq = Float(7 / 40)


@register_eq_class
class spin_glass_N4_M2_2(KnownEquation):
    _eq_name = 'spin_glass_N4_M2_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6
    expr_consts_thres = None

    def __init__(self):
        self.dim = [1, 1, 1]

        vars_range_and_types = [IntegerUniformSampling(8, 9, only_positive=True),
                                IntegerUniformSampling(2, 3, only_positive=True),
                                IntegerUniformSampling(4, 5, only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=vars_range_and_types)
        self.sympy_eq = Float(-0.0125)


@register_eq_class
class spin_glass_N4_M2_3(KnownEquation):
    _eq_name = 'spin_glass_N4_M2_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6
    expr_consts_thres = None

    def __init__(self):
        self.dim = [1, 1, 1]

        vars_range_and_types = [IntegerUniformSampling(8, 9, only_positive=True),
                                IntegerUniformSampling(2, 3, only_positive=True),
                                IntegerUniformSampling(4, 5, only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=vars_range_and_types)
        self.sympy_eq = Float(-0.075)


@register_eq_class
class spin_glass_N5_M2_1(KnownEquation):
    _eq_name = 'spin_glass_N5_M2_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-8
    expr_consts_thres = None

    def __init__(self):
        self.dim = [1, 1, 1]

        vars_range_and_types = [IntegerUniformSampling(12, 13, only_positive=True),
                                IntegerUniformSampling(3, 4, only_positive=True),
                                IntegerUniformSampling(4, 5, only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=vars_range_and_types)
        self.sympy_eq = Float(0.10952381)


@register_eq_class
class spin_glass_N5_M2_2(KnownEquation):
    _eq_name = 'spin_glass_N5_M2_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6
    expr_consts_thres = None

    def __init__(self):
        self.dim = [1, 1, 1]

        vars_range_and_types = [IntegerUniformSampling(12, 13, only_positive=True),
                                IntegerUniformSampling(3, 4, only_positive=True),
                                IntegerUniformSampling(4, 5, only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=vars_range_and_types)
        self.sympy_eq = Float(-0.0047619)

@register_eq_class
class spin_glass_N5_M2_3(KnownEquation):
    _eq_name = 'spin_glass_N5_M2_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6
    expr_consts_thres = None

    def __init__(self):
        self.dim = [1, 1, 1]

        vars_range_and_types = [IntegerUniformSampling(12, 13, only_positive=True),
                                IntegerUniformSampling(3, 4, only_positive=True),
                                IntegerUniformSampling(4, 5, only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=vars_range_and_types)
        self.sympy_eq = Float(-0.01904762)


@register_eq_class
class spin_glass_N6_M2_1(KnownEquation):
    _eq_name = 'spin_glass_N6_M2_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6
    expr_consts_thres = None

    def __init__(self):
        self.dim = [1, 1]

        vars_range_and_types = [IntegerUniformSampling(16, 17, only_positive=True),
                                IntegerUniformSampling(4, 5, only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=vars_range_and_types)
        self.sympy_eq = Float(0.0787037)



@register_eq_class
class spin_glass_N6_M2_2(KnownEquation):
    _eq_name = 'spin_glass_N6_M2_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']
    expr_obj_thres = 1e-6
    expr_consts_thres = None

    def __init__(self):
        self.dim = [1, 1]

        vars_range_and_types = [IntegerUniformSampling(16, 17, only_positive=True),
                                IntegerUniformSampling(4, 5, only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=vars_range_and_types)
        self.sympy_eq = Float(-0.00462963)

