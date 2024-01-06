from collections import OrderedDict
import sympy
from base import KnownEquation, LogUniformSampling, IntegerUniformSampling, UniformSampling

EQUATION_CLASS_DICT = OrderedDict()


def register_eq_class(cls):
    EQUATION_CLASS_DICT[cls.__name__] = cls
    return cls


def get_eq_obj(key, **kwargs):
    if key in EQUATION_CLASS_DICT:
        return EQUATION_CLASS_DICT[key](**kwargs)
    raise KeyError(f'`{key}` is not expected as a equation object key')


@register_eq_class
class Livermore2_Vars2_1(KnownEquation):
    _eq_name = 'Livermore2_Vars2_1'
    _function_set = ['add', 'sub', 'mul', 'sqrt', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.sqrt(1.5 * x[0] + 0.8 * x[1]) * (2.97 * x[1] ** 2 + 2.8 * x[1])


@register_eq_class
class Livermore2_Vars2_2(KnownEquation):
    _eq_name = 'Livermore2_Vars2_2'
    _function_set = ['add', 'sub', 'mul', 'sqrt', 'n3', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = x[0] * x[1] * x[0] * sympy.sqrt(x[1]) * (6.28 * x[0] * x[1] + x[0] + 7.41 * x[1] ** 3 - 1.4 + x[0])


@register_eq_class
class Livermore2_Vars2_3(KnownEquation):
    _eq_name = 'Livermore2_Vars2_3'
    _function_set = ['add', 'sub', 'mul', 'sqrt', 'n2', 'n4', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = -2.72 * x[0] ** 2 * (x[0] ** 2 * x[1] + x[0] + sympy.sqrt(x[0] ** 4 + 0.3))


@register_eq_class
class Livermore2_Vars2_4(KnownEquation):
    _eq_name = 'Livermore2_Vars2_4'
    _function_set = ['add', 'sub', 'mul', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 0.29 * x[0] + 0.29 * x[1] + (x[0] - x[1]) ** 2


@register_eq_class
class Livermore2_Vars2_5(KnownEquation):
    _eq_name = 'Livermore2_Vars2_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'n3', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = x[0] + sympy.sin(x[1]) - 7.88 * x[0] * x[1] ** 2 + 10.59 * x[1] ** 3


@register_eq_class
class Livermore2_Vars2_6(KnownEquation):
    _eq_name = 'Livermore2_Vars2_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 2.01 * x[0] / (14.71 - 4.97 * x[1]) - x[1] * (-x[0] ** 2 + x[0] - x[1])


@register_eq_class
class Livermore2_Vars2_7(KnownEquation):
    _eq_name = 'Livermore2_Vars2_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 1.65 * sympy.sqrt(x[0]) + 2.65 * x[0] - 1.65 * sympy.sqrt(x[1]) + 1 - 2.11 * x[1] / x[0]


@register_eq_class
class Livermore2_Vars2_8(KnownEquation):
    _eq_name = 'Livermore2_Vars2_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'n3', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = -4.22 * x[0] ** 2 + 2.08 * x[1] ** 3 - 2.76 * x[1] ** 2 + (sympy.sqrt(x[0]) - x[0]) * sympy.log(x[1]) / sympy.log(
            x[0])


@register_eq_class
class Livermore2_Vars2_9(KnownEquation):
    _eq_name = 'Livermore2_Vars2_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = x[0] ** 2 + x[0] - x[1] - 2.18 * sympy.sqrt(0.42 * x[0]) - 0.21 * x[1] + 1 - 2.14


@register_eq_class
class Livermore2_Vars2_10(KnownEquation):
    _eq_name = 'Livermore2_Vars2_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = -1.92 * x[0] * x[1] * sympy.exp(2 * x[1])


@register_eq_class
class Livermore2_Vars2_11(KnownEquation):
    _eq_name = 'Livermore2_Vars2_11'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = x[0] * x[1] + sympy.sqrt(x[1]) + 3.52 * sympy.sin(x[0])


@register_eq_class
class Livermore2_Vars2_12(KnownEquation):
    _eq_name = 'Livermore2_Vars2_12'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'n3', 'log', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.sqrt(x[0] + x[1]) * sympy.exp(-x[1]) + \
                        sympy.log(6.27 * x[0] ** 3) + 4.32 * x[0] ** 2 * x[1] - 7.87 * x[1] ** 3


@register_eq_class
class Livermore2_Vars2_13(KnownEquation):
    _eq_name = 'Livermore2_Vars2_13'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = x[0] + x[1] * (-1.49 * x[0] * x[1] + 6.81 * x[0]) + x[1] * sympy.cos(x[0])


@register_eq_class
class Livermore2_Vars2_14(KnownEquation):
    _eq_name = 'Livermore2_Vars2_14'
    _function_set = ['add', 'sub', 'mul', 'div', 'n4', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = x[0] ** 4 / x[1] ** 4 - 0.91


@register_eq_class
class Livermore2_Vars2_15(KnownEquation):
    _eq_name = 'Livermore2_Vars2_15'
    _function_set = ['add', 'sub', 'mul', 'div', 'n3', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 11.57 * x[0] ** 3 + x[1]


@register_eq_class
class Livermore2_Vars2_16(KnownEquation):
    _eq_name = 'Livermore2_Vars2_16'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 2.0 * x[0] / (14.71 - 4.97 * x[1]) - x[1] * (-x[0] ** 2 + x[0] - x[1])


@register_eq_class
class Livermore2_Vars2_17(KnownEquation):
    _eq_name = 'Livermore2_Vars2_17'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = -sympy.sqrt(x[0]) * x[1] + x[0] - sympy.sin(-5.2 * x[0] + 9.32 * x[0] + 6.94 * x[1])


@register_eq_class
class Livermore2_Vars2_18(KnownEquation):
    _eq_name = 'Livermore2_Vars2_18'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (20.08 * x[0] * x[1] ** 2 + x[0]) / (x[0] * (sympy.sqrt(x[1]) + x[1]))


@register_eq_class
class Livermore2_Vars2_19(KnownEquation):
    _eq_name = 'Livermore2_Vars2_19'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (x[0] - 0.48 * sympy.cos(x[0]) / sympy.sqrt(0.23 * x[0]) * (x[1] ** 2 + sympy.log(x[1])) + \
                         (0.23 * x[0] + 1)) * x[0]


@register_eq_class
class Livermore2_Vars2_20(KnownEquation):
    _eq_name = 'Livermore2_Vars2_20'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'n4', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (x[0] ** 2 + x[1]) ** 2 * (-x[0] / (-8.62 * x[0] ** 2 - 5.38 * x[0] * x[1]) + x[1]) ** 4 * sympy.sin(
            sympy.sqrt(x[0])) ** 4 / x[0] ** 4


@register_eq_class
class Livermore2_Vars2_21(KnownEquation):
    _eq_name = 'Livermore2_Vars2_21'
    _function_set = ['add', 'sub', 'mul', 'div', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = x[0] + sympy.log(x[0] + 0.75 / (x[0] + x[1] + 0.41))


@register_eq_class
class Livermore2_Vars2_22(KnownEquation):
    _eq_name = 'Livermore2_Vars2_22'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.log(x[0]) * sympy.exp(x[1]) / (sympy.cos(1.88 * x[1]) - 4.09 * (0.49 * x[0] - 1) ** 2) / sympy.sqrt(
            x[0])


@register_eq_class
class Livermore2_Vars2_23(KnownEquation):
    _eq_name = 'Livermore2_Vars2_23'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = x[0] ** 2 + 1.37 * (1 - 0.53 * (-x[1] - 2.63) * (sympy.log(x[1]) - 1.16) / x[0])


@register_eq_class
class Livermore2_Vars2_24(KnownEquation):
    _eq_name = 'Livermore2_Vars2_24'
    _function_set = ['add', 'sub', 'mul', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 3.04 * x[0] + 3.04 * (x[0] - x[1] ** 2) ** 2 - 10.82


@register_eq_class
class Livermore2_Vars2_25(KnownEquation):
    _eq_name = 'Livermore2_Vars2_25'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = x[0] + x[0] / (9.25 * x[0] ** 2 * x[1] + x[0] + x[1] / x[0])


@register_eq_class
class Livermore2_Vars3_1(KnownEquation):
    _eq_name = 'Livermore2_Vars3_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[2] * x[1] + x[2] - x[0] ** 2 + 4.45 * x[1] ** 2 + x[2] - sympy.exp(-sympy.sin(x[0] - x[2])) * sympy.exp(x[0])


@register_eq_class
class Livermore2_Vars3_2(KnownEquation):
    _eq_name = 'Livermore2_Vars3_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = 9.28 * (0.91 - x[2]) ** 2 * (-0.5 * x[0] - x[1]) ** 2 / (0.34 * x[1] ** 2 + 1) ** 2


@register_eq_class
class Livermore2_Vars3_3(KnownEquation):
    _eq_name = 'Livermore2_Vars3_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'n3', 'log', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = -x[0] * (sympy.sqrt(x[0]) / (x[1] * (3.51 * x[1] ** 3 + 1.15 * x[1] * x[2])) - x[2] ** 2) - x[0] - x[2] + sympy.exp(
            x[0]) - sympy.log(x[0]) + sympy.log(sympy.sqrt(x[2]))


@register_eq_class
class Livermore2_Vars3_4(KnownEquation):
    _eq_name = 'Livermore2_Vars3_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[0] + 0.31 * x[1] * x[2] + sympy.sqrt(x[1] + x[2]) + (sympy.cos(x[1])) - 3.19


@register_eq_class
class Livermore2_Vars3_5(KnownEquation):
    _eq_name = 'Livermore2_Vars3_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'n3', 'n4', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = -2.24 * x[0] ** 3 * x[2] + x[2] ** 4 * (-x[1] + x[1] / x[0])


@register_eq_class
class Livermore2_Vars3_6(KnownEquation):
    _eq_name = 'Livermore2_Vars3_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[0] * (x[0] + (x[1] - 1.23) * (x[0] - x[1] - x[2])) * sympy.sin(sympy.exp(x[1])) / x[2]


@register_eq_class
class Livermore2_Vars3_7(KnownEquation):
    _eq_name = 'Livermore2_Vars3_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[0] * (x[1] + x[2] * (x[0] + x[2])) + x[2] * sympy.cos(0.89 * x[0] ** 2) + x[2] + 2.08


@register_eq_class
class Livermore2_Vars3_8(KnownEquation):
    _eq_name = 'Livermore2_Vars3_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'exp', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[0] / ((-x[1] + x[2] + (sympy.log(x[2]) / (x[0] ** 2 + sympy.sqrt(x[2])))) * (
                1.21 * x[0] ** 2 * x[2] + 0.44 * x[0] ** 2 + 4.95 * x[2] ** 2 + sympy.exp(2 * x[1])))


@register_eq_class
class Livermore2_Vars3_9(KnownEquation):
    _eq_name = 'Livermore2_Vars3_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[0] * sympy.sin(0.61 * x[0] / sympy.sqrt(
            0.36 * x[0] ** 2) - x[0] * sympy.sqrt(0.45 * x[0] * x[2] ** 2) + x[1]) * (x[0] + x[1] ** 2 - 1) / x[2]


@register_eq_class
class Livermore2_Vars3_10(KnownEquation):
    _eq_name = 'Livermore2_Vars3_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = -2.85 * (x[0] - 1.43) * (x[0] + x[1]) * (x[0] - x[2]) / x[0]


@register_eq_class
class Livermore2_Vars3_11(KnownEquation):
    _eq_name = 'Livermore2_Vars3_11'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = 4.83 * x[0] + x[0] / x[1] - 3.83 * x[1] - 3.83 * x[2]


@register_eq_class
class Livermore2_Vars3_12(KnownEquation):
    _eq_name = 'Livermore2_Vars3_12'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = -1.35 * x[0] * x[1] ** 2 + x[1] + x[2] + 3.18


@register_eq_class
class Livermore2_Vars3_13(KnownEquation):
    _eq_name = 'Livermore2_Vars3_13'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = 0.96 / (x[0] * sympy.sqrt(x[1])) * sympy.sqrt(x[0] * x[2] + 3.54) * sympy.exp(-x[1] + x[2]) * (
                x[0] + x[1]) / sympy.sin(x[0])


@register_eq_class
class Livermore2_Vars3_14(KnownEquation):
    _eq_name = 'Livermore2_Vars3_14'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[1] * (sympy.sqrt(x[0]) + x[0] * x[1]) * (x[1] - x[2] + sympy.exp((0.5 - x[1]) / x[0]))


@register_eq_class
class Livermore2_Vars3_15(KnownEquation):
    _eq_name = 'Livermore2_Vars3_15'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = sympy.log(x[0] * (x[1] + x[2])) / x[1] ** 2 - 4.68 * x[1] ** 2 * x[2]


@register_eq_class
class Livermore2_Vars3_16(KnownEquation):
    _eq_name = 'Livermore2_Vars3_16'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = sympy.sqrt(x[0]) + sympy.exp(x[0] * x[2]) + (x[0] + x[1]) + (
                11.13 * x[0] * x[2] ** 2 + 2 * x[0] + x[2]) * sympy.cos(x[0] ** 2)


@register_eq_class
class Livermore2_Vars3_17(KnownEquation):
    _eq_name = 'Livermore2_Vars3_17'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[0] * sympy.sqrt(2 * x[1]) * sympy.cos(4 * x[0]) - 3.406 * x[0] + 3.41 * x[1] + x[2] + 1


@register_eq_class
class Livermore2_Vars3_18(KnownEquation):
    _eq_name = 'Livermore2_Vars3_18'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[0] + x[1] * x[2] - 3.03 * sympy.sqrt(x[2])


@register_eq_class
class Livermore2_Vars3_19(KnownEquation):
    _eq_name = 'Livermore2_Vars3_19'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = 7.14 * x[0] * x[1] ** 2 + x[1] ** 2 * x[2] + x[2] * sympy.sqrt(x[0] * (x[0] + x[1]))


@register_eq_class
class Livermore2_Vars3_20(KnownEquation):
    _eq_name = 'Livermore2_Vars3_20'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[0] + 0.71 * (sympy.sqrt(x[1] / x[2])) - 4.21


@register_eq_class
class Livermore2_Vars3_21(KnownEquation):
    _eq_name = 'Livermore2_Vars3_21'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = -x[0] ** 2 * x[2] + 1.99 * x[0] * sympy.sqrt(0.25 * (x[1] + x[2] + 1.0))


@register_eq_class
class Livermore2_Vars3_22(KnownEquation):
    _eq_name = 'Livermore2_Vars3_22'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'sin', 'cos']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[2] + sympy.sin(x[0] * x[1] - x[1] - x[2] + 2.39)


@register_eq_class
class Livermore2_Vars3_23(KnownEquation):
    _eq_name = 'Livermore2_Vars3_23'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[0] ** 2 * x[1] / (x[0] + x[2])


@register_eq_class
class Livermore2_Vars3_24(KnownEquation):
    _eq_name = 'Livermore2_Vars3_24'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = -1.35 * x[0] * x[1] ** 2 + x[1] + x[2] + 3.18


@register_eq_class
class Livermore2_Vars3_25(KnownEquation):
    _eq_name = 'Livermore2_Vars3_25'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const', 'sin', 'cos']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[0] * sympy.sqrt(x[1]) * sympy.cos(x[0]) - 3.41 * (x[0] - x[1]) + x[2] + 1


@register_eq_class
class Livermore2_Vars4_1(KnownEquation):
    _eq_name = 'Livermore2_Vars4_1'
    _function_set = ['add', 'sub', 'mul', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] - x[1] * x[2] - x[1] - 3 * x[3]


@register_eq_class
class Livermore2_Vars4_2(KnownEquation):
    _eq_name = 'Livermore2_Vars4_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] * sympy.sqrt(2 * x[1]) * x[3] / x[2] + 1


@register_eq_class
class Livermore2_Vars4_3(KnownEquation):
    _eq_name = 'Livermore2_Vars4_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = 2 * x[0] + x[3] - 0.01 + x[2] / x[1]


@register_eq_class
class Livermore2_Vars4_4(KnownEquation):
    _eq_name = 'Livermore2_Vars4_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'n2', 'sin']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] - x[3] - (-x[0] + sympy.sin(x[0])) ** 2 / (x[0] ** 2 * x[1] ** 2 * x[2] ** 2)


@register_eq_class
class Livermore2_Vars4_5(KnownEquation):
    _eq_name = 'Livermore2_Vars4_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'n2', 'sin']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] + sympy.sin(
            x[1] / (x[0] * x[1] ** 2 * x[3] ** 2 * (-3.22 * x[1] * x[3] ** 2 + 13.91 * x[1] * x[3] + x[2]) / 2 + x[1]))


@register_eq_class
class Livermore2_Vars4_6(KnownEquation):
    _eq_name = 'Livermore2_Vars4_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'exp', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = -x[0] - 0.54 * sympy.exp(x[0]) * sympy.sqrt(x[3]) + sympy.cos(x[1]) * sympy.exp(-2 * x[0]) / x[2]


@register_eq_class
class Livermore2_Vars4_7(KnownEquation):
    _eq_name = 'Livermore2_Vars4_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'log', 'n2', 'cos']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] + sympy.cos(x[1]) / sympy.log(1 + x[1] ** 2) + x[2] + x[3]


@register_eq_class
class Livermore2_Vars4_8(KnownEquation):
    _eq_name = 'Livermore2_Vars4_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'n3', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] * (x[0] + x[3] + sympy.sin(
            (-x[0] * sympy.exp(x[2]) + x[1]) / (-4.47 * x[0] ** 2 * x[2] + 8.31 * x[2] ** 3 + 5.27 * x[2] ** 2))) - x[0]


@register_eq_class
class Livermore2_Vars4_9(KnownEquation):
    _eq_name = 'Livermore2_Vars4_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] - x[3] + sympy.cos(x[0] * (x[0] + x[1]) * (x[0] ** 2 * x[1] + x[2]) + x[2])


@register_eq_class
class Livermore2_Vars4_10(KnownEquation):
    _eq_name = 'Livermore2_Vars4_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'const', 'sin', 'cos', ]

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] + x[0] * (x[3] + (sympy.sqrt(x[1]) - sympy.sin(x[2])) / x[2])


@register_eq_class
class Livermore2_Vars4_11(KnownEquation):
    _eq_name = 'Livermore2_Vars4_11'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = 2 * x[0] + x[1] * (x[0] + sympy.sin(x[1] * x[2])) + sympy.sin(2 / x[3])


@register_eq_class
class Livermore2_Vars4_12(KnownEquation):
    _eq_name = 'Livermore2_Vars4_12'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] * x[1] + 16.97 * x[2] - x[3]


@register_eq_class
class Livermore2_Vars4_13(KnownEquation):
    _eq_name = 'Livermore2_Vars4_13'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[3] * (-x[2] - sympy.sin(x[0] ** 2 - x[0] + x[1]))


@register_eq_class
class Livermore2_Vars4_14(KnownEquation):
    _eq_name = 'Livermore2_Vars4_14'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] + sympy.cos(x[1] ** 2 * (-x[1] + x[2] + 3.23) + x[3])


@register_eq_class
class Livermore2_Vars4_15(KnownEquation):
    _eq_name = 'Livermore2_Vars4_15'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'log', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] * x[1] + sympy.log(x[2] + x[3]) + sympy.exp(x[1] - 0.28 / x[0]) - x[2] - x[3] / (2 * x[0] * x[2])


@register_eq_class
class Livermore2_Vars4_16(KnownEquation):
    _eq_name = 'Livermore2_Vars4_16'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = sympy.sqrt(x[1]) * (-x[0] ** 2) + sympy.exp(x[1]) \
                        + x[2] * (-x[3] + 1.81 / x[2]) \
                        - 2.34 * x[3] / x[0]


@register_eq_class
class Livermore2_Vars4_17(KnownEquation):
    _eq_name = 'Livermore2_Vars4_17'
    _function_set = ['add', 'sub', 'mul', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] ** 2 - x[1] - x[2] ** 2 - x[3]


@register_eq_class
class Livermore2_Vars4_18(KnownEquation):
    _eq_name = 'Livermore2_Vars4_18'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'exp', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] + sympy.sin(2 * x[1] + x[2]) - \
                        x[3] * sympy.exp(x[0]) + \
                        2.96 * sympy.sqrt(0.36 * x[1] ** 2 + x[1] * x[2] ** 2 + 0.94) + \
                        sympy.log(x[1] + 1)


@register_eq_class
class Livermore2_Vars4_19(KnownEquation):
    _eq_name = 'Livermore2_Vars4_19'
    _function_set = ['add', 'sub', 'mul', 'div', 'n3', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = (x[0] ** 3 * x[1] - 2.86 * x[0] + x[3]) / x[2]


@register_eq_class
class Livermore2_Vars4_20(KnownEquation):
    _eq_name = 'Livermore2_Vars4_20'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] + x[1] + 6.21 + 1 / (x[2] * x[3] + x[2] + 2.08)


@register_eq_class
class Livermore2_Vars4_21(KnownEquation):
    _eq_name = 'Livermore2_Vars4_21'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = x[0] * (x[1] - x[2] + x[3]) + 2 * x[3]


@register_eq_class
class Livermore2_Vars4_22(KnownEquation):
    _eq_name = 'Livermore2_Vars4_22'
    _function_set = ['add', 'sub', 'mul', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = 2 * x[0] - x[1] * x[2] + x[1] * sympy.exp(x[0]) - x[3]


@register_eq_class
class Livermore2_Vars4_23(KnownEquation):
    _eq_name = 'Livermore2_Vars4_23'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = -x[0] / x[1] - 2.23 * x[1] * x[2] + x[1] - \
                        2.23 * x[2] / sympy.sqrt(x[3]) - 2.23 * sympy.sqrt(x[3]) + sympy.log(x[0])


@register_eq_class
class Livermore2_Vars4_24(KnownEquation):
    _eq_name = 'Livermore2_Vars4_24'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = -4.81 * sympy.log(x[0]) * (x[0] * x[1]) + x[0] + sympy.sqrt(x[3]) + sympy.log(x[2])


@register_eq_class
class Livermore2_Vars4_25(KnownEquation):
    _eq_name = 'Livermore2_Vars4_25'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'sin', 'cos']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = 0.38 + (-x[0] / x[3] + sympy.cos(2 * x[0] * x[2] / (x[3] * (x[0] + x[1] * x[2]))) / x[3]) / x[1]


@register_eq_class
class Livermore2_Vars5_1(KnownEquation):
    _eq_name = 'Livermore2_Vars5_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = - 1.49610041*x[0] + 0.45513135* x[1] - 0.66025708* x[2] + -0.29496258*x[3] - 1.5078599*x[4] - 4.75


@register_eq_class
class Livermore2_Vars5_2(KnownEquation):
    _eq_name = 'Livermore2_Vars5_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[2] * (x[0] + x[4] + 0.27 / (x[2] ** 2 + (x[1] + x[3]) / (x[0] * x[1] + x[1])))


@register_eq_class
class Livermore2_Vars5_3(KnownEquation):
    _eq_name = 'Livermore2_Vars5_3'
    _function_set = ['add', 'sub', 'mul', 'log', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 2 * x[0] * x[1] * x[2] + x[4] - sympy.sin(x[0] * sympy.log(x[1] + 1) - x[0] + x[3])


@register_eq_class
class Livermore2_Vars5_4(KnownEquation):
    _eq_name = 'Livermore2_Vars5_4'
    _function_set = ['add', 'sub', 'mul', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 1.9278154269561125 *x[1] + 0.018273668817525703 *x[2] * x[3] + 1.5928760021692208 * x[4] ** 2 + -0.09851514736678954 *sympy.sin(x[0])


@register_eq_class
class Livermore2_Vars5_5(KnownEquation):
    _eq_name = 'Livermore2_Vars5_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[4] + 0.36 * sympy.log(1.0 + x[0] * x[1]) + sympy.sqrt(x[2]) + sympy.log(x[1] + x[3])


@register_eq_class
class Livermore2_Vars5_6(KnownEquation):
    _eq_name = 'Livermore2_Vars5_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[0] * x[3] + x[0] + x[1] + x[4] + sympy.sqrt(0.08 * x[0] / (x[2] * x[4]) + x[2])


@register_eq_class
class Livermore2_Vars5_7(KnownEquation):
    _eq_name = 'Livermore2_Vars5_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[0] * x[4] + sympy.sqrt(x[0] * x[1]) * sympy.cos(x[0]) - x[0] / (x[1] + x[2] + x[3] + 8.05)


@register_eq_class
class Livermore2_Vars5_8(KnownEquation):
    _eq_name = 'Livermore2_Vars5_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = sympy.sqrt(x[1]) * x[2] - \
                        x[3] - 0.07 * (x[0] + (x[0] - x[1]) * sympy.sqrt(x[1] + 0.99)) * sympy.cos(x[4])


@register_eq_class
class Livermore2_Vars5_9(KnownEquation):
    _eq_name = 'Livermore2_Vars5_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[0] * (x[2] + (x[0] + x[1]) / (x[1] * x[3] + x[4]))


@register_eq_class
class Livermore2_Vars5_10(KnownEquation):
    _eq_name = 'Livermore2_Vars5_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[0] / x[3] * (-0.25 * x[0] * x[2] * x[3] + x[1] - 8.43 * x[3] * x[4]) * sympy.sin(x[2] + 1) + x[3] * x[4]


@register_eq_class
class Livermore2_Vars5_11(KnownEquation):
    _eq_name = 'Livermore2_Vars5_11'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = -x[3] ** 2 + sympy.sqrt(x[0] * x[2] + x[4]) - x[1] + x[4] / x[2] + \
                        0.47 * sympy.sqrt(x[2] * x[0]) - sympy.sqrt(x[1]) / x[1]


@register_eq_class
class Livermore2_Vars5_12(KnownEquation):
    _eq_name = 'Livermore2_Vars5_12'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[0] * (x[1] - 1 / (x[2] * (x[3] + x[4])))


@register_eq_class
class Livermore2_Vars5_13(KnownEquation):
    _eq_name = 'Livermore2_Vars5_13'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = sympy.sqrt(x[0] * x[4]) * (x[1] - 1.52) - sympy.cos(4.03 * x[2] + x[3])


@register_eq_class
class Livermore2_Vars5_14(KnownEquation):
    _eq_name = 'Livermore2_Vars5_14'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = -x[0] / (x[1] * x[4]) + sympy.cos(x[0] * x[2] * x[3] * sympy.exp(-x[1]))


@register_eq_class
class Livermore2_Vars5_15(KnownEquation):
    _eq_name = 'Livermore2_Vars5_15'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = -x[3] + sympy.log(1 + x[0]) / sympy.log(1 + 11.06 * x[1] * x[4]) + x[2] - sympy.cos(x[1]) + x[4] + \
                        sympy.sqrt(x[1] * x[4])


@register_eq_class
class Livermore2_Vars5_16(KnownEquation):
    _eq_name = 'Livermore2_Vars5_16'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[1] + 0.33 * x[4] * (x[0] / (x[0] ** 2 + x[1]) + x[2] * x[3])


@register_eq_class
class Livermore2_Vars5_17(KnownEquation):
    _eq_name = 'Livermore2_Vars5_17'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'cos', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[0] - sympy.sin(x[1]) + sympy.sin(x[2]) - sympy.cos(-x[1] + sympy.sqrt(x[3]) + x[4]) + 0.78


@register_eq_class
class Livermore2_Vars5_18(KnownEquation):
    _eq_name = 'Livermore2_Vars5_18'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[0] * x[1] - x[3] - (x[2] * sympy.sqrt(1 / (x[0] * (x[2] + x[3]))) - 1.13 / x[2]) / x[4]


@register_eq_class
class Livermore2_Vars5_19(KnownEquation):
    _eq_name = 'Livermore2_Vars5_19'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 4.53 * x[0] * x[1] + x[0] - x[0] * sympy.cos(sympy.sqrt(x[1])) / x[1] - x[2] - x[3] - x[4]


@register_eq_class
class Livermore2_Vars5_20(KnownEquation):
    _eq_name = 'Livermore2_Vars5_20'
    _function_set = ['add', 'sub', 'mul', 'div', 'log', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = -sympy.exp(x[0] + x[4]) + \
                        sympy.sin(x[0] - 4.81) / (0.21 * (x[4] - sympy.log(1.0 + x[2] + x[3]) - sympy.exp(x[4])) / x[1])


@register_eq_class
class Livermore2_Vars5_21(KnownEquation):
    _eq_name = 'Livermore2_Vars5_21'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt',  'log', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = sympy.sqrt(x[3]) * (2 * x[0]) + \
                        sympy.cos(x[0] * (x[2] * x[3])) * sympy.exp(x[0] * x[1]) + x[2] - \
                        sympy.log(x[2] + 3.49) / x[4]


@register_eq_class
class Livermore2_Vars5_22(KnownEquation):
    _eq_name = 'Livermore2_Vars5_22'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'log', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = -x[0] - x[1] + x[2] + x[0] - x[1] * (
                sympy.sin(x[2]) - sympy.log(1 + x[0] * x[4] / (x[1] ** 2 + x[3])) / x[3]) - 0.73


@register_eq_class
class Livermore2_Vars5_23(KnownEquation):
    _eq_name = 'Livermore2_Vars5_23'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[0] * (x[1] / (x[2] + sympy.sqrt(x[1] * (x[3] + x[4])) * (1 - x[2] + x[3])) - x[4])


@register_eq_class
class Livermore2_Vars5_24(KnownEquation):
    _eq_name = 'Livermore2_Vars5_24'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = -x[1] * x[4] + sympy.sqrt(x[0]) + \
                        x[1] * (-x[0] + x[3] * sympy.cos(sympy.sqrt(x[2]) + x[2]) - (x[1] + 7.84 * x[2] ** 2 * x[4]) / x[4]) + \
                        x[1] / x[2]


@register_eq_class
class Livermore2_Vars5_25(KnownEquation):
    _eq_name = 'Livermore2_Vars5_25'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'log', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = x[0] + sympy.log(x[0]) + -3.57 * x[2] ** 2 * x[1] + x[1] + x[2] * sympy.log(x[3]) * sympy.sin(x[2]) / x[4] + x[2]


@register_eq_class
class Livermore2_Vars6_1(KnownEquation):
    _eq_name = 'Livermore2_Vars6_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] - x[5] + (x[0] + x[3] + x[4]) * sympy.sqrt(x[0] ** 2 + x[1]) - x[2]


@register_eq_class
class Livermore2_Vars6_2(KnownEquation):
    _eq_name = 'Livermore2_Vars6_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] * (2 * x[1] + x[1] / x[2] + x[3] + sympy.log(1.0 + x[0] * x[4] * x[5]))


@register_eq_class
class Livermore2_Vars6_3(KnownEquation):
    _eq_name = 'Livermore2_Vars6_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n4', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = sympy.sqrt(x[1] + x[4]) - sympy.sqrt(x[5] + x[2] ** 4 * x[3] ** 4 / (x[0] * x[1] ** 4))


@register_eq_class
class Livermore2_Vars6_4(KnownEquation):
    _eq_name = 'Livermore2_Vars6_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] * x[1] * (x[0] ** 2 + x[0]) - x[1] + x[2] ** 2 - x[2] - x[4] - x[5] - sympy.sin(x[3]) - sympy.cos(x[3])


@register_eq_class
class Livermore2_Vars6_5(KnownEquation):
    _eq_name = 'Livermore2_Vars6_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[1] * sympy.sqrt(x[0] * x[1]) * (x[0] * x[2] - x[2] - x[3]) + x[4] + x[5]


@register_eq_class
class Livermore2_Vars6_6(KnownEquation):
    _eq_name = 'Livermore2_Vars6_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = (x[0] / (x[1] * x[2] + 2 * sympy.log(2 + sympy.cos(x[0]))) - x[1] * x[3] + \
                         sympy.sin((x[1] * x[3] + x[4]) / x[5]) + \
                         sympy.cos(x[2])) * sympy.log(x[0])


@register_eq_class
class Livermore2_Vars6_7(KnownEquation):
    _eq_name = 'Livermore2_Vars6_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] * sympy.sqrt(x[0]) - x[5] ** 2 + \
                        sympy.sin(x[0]) * sympy.exp(-x[1]) - x[3] * (x[1] + x[2] ** 2) / (x[1] + x[4])


@register_eq_class
class Livermore2_Vars6_8(KnownEquation):
    _eq_name = 'Livermore2_Vars6_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] + x[1] ** 2 + 0.34 * x[2] * x[4] - x[3] + x[5]


@register_eq_class
class Livermore2_Vars6_9(KnownEquation):
    _eq_name = 'Livermore2_Vars6_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[3] * x[0] + sympy.exp(13.28 * x[2] * x[5]) - \
                        x[4] ** 2 * 4 * sympy.log(x[1]) / (x[0] * x[2] - x[1] ** 2) + \
                        x[1] - x[5] - sympy.log(0.5 + x[2])


@register_eq_class
class Livermore2_Vars6_10(KnownEquation):
    _eq_name = 'Livermore2_Vars6_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] + 61.36 * x[1] ** 2 + x[1] / (x[0] * x[2] * (x[3] - sympy.cos(x[3] * (2 * x[0] * x[1] * x[5] / x[4] + x[4]))))


@register_eq_class
class Livermore2_Vars6_11(KnownEquation):
    _eq_name = 'Livermore2_Vars6_11'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = (x[0] + x[0] / (x[1] + x[3] * (8.13 * x[0] ** 2 * x[5] + x[0] * x[1] * x[2] + 2 * x[1] + x[4] + x[5]))) ** 2


@register_eq_class
class Livermore2_Vars6_12(KnownEquation):
    _eq_name = 'Livermore2_Vars6_12'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = (1.41 * sympy.sqrt(x[0]) - x[1] - x[2] / sympy.sqrt(
            x[3] * (8.29 * x[0] * x[2] ** 2 + x[0] * x[4]) + x[3] + x[5])) / x[5]


@register_eq_class
class Livermore2_Vars6_13(KnownEquation):
    _eq_name = 'Livermore2_Vars6_13'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] + x[4] + 0.21 * sympy.sqrt(
            x[0] / (x[1] ** 2 * x[2] ** 2 * sympy.sqrt(x[5]) * (sympy.sqrt(x[2]) + x[2] + 2 * x[5] + (x[1] + x[3] + x[4]) / x[4])))


@register_eq_class
class Livermore2_Vars6_14(KnownEquation):
    _eq_name = 'Livermore2_Vars6_14'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = -2.07 * x[5] + sympy.log(x[1]) - x[5] - sympy.sqrt(x[2] * x[4]) + sympy.log(x[0]) + (x[4] + 1) / x[3]


@register_eq_class
class Livermore2_Vars6_15(KnownEquation):
    _eq_name = 'Livermore2_Vars6_15'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] * (x[0] + sympy.cos(x[1] ** 2 * x[2] * x[3] * (x[4] - 0.43 * x[5] ** 2))) / x[3]


@register_eq_class
class Livermore2_Vars6_16(KnownEquation):
    _eq_name = 'Livermore2_Vars6_16'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = -sympy.sqrt(x[0]) - x[0] + x[1] - x[3] - x[4] - sympy.sqrt(x[5] / x[2]) - 3.26


@register_eq_class
class Livermore2_Vars6_17(KnownEquation):
    _eq_name = 'Livermore2_Vars6_17'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] / (x[1] * x[3]) + (-x[4] + 2 * sympy.log(x[5]) * sympy.cos(2 * x[1] + x[2] ** 2 - x[3])) * (
                129.28 * x[0] ** 2 * x[1] ** 2 + x[2])


@register_eq_class
class Livermore2_Vars6_18(KnownEquation):
    _eq_name = 'Livermore2_Vars6_18'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'exp', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = sympy.sqrt(x[4]) * (
                2 * x[0] + sympy.cos(x[0] * (x[2] * x[3] * sympy.exp(x[0] * x[1]) + x[2] - sympy.log(0.5 + x[2]) - 3.49)) / x[5])


@register_eq_class
class Livermore2_Vars6_19(KnownEquation):
    _eq_name = 'Livermore2_Vars6_19'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] + x[1] + x[2] + 0.84 * sympy.sqrt(x[2] * x[5]) + x[3] - \
                        x[4] + x[1] + sympy.log(0.5 + x[2]) + \
                        sympy.exp(x[1]) / (x[1] - x[3])


@register_eq_class
class Livermore2_Vars6_20(KnownEquation):
    _eq_name = 'Livermore2_Vars6_20'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] - 0.97 * x[0] / (x[4] - x[5] * (x[0] * x[3] + x[5])) - x[1] + x[2] + \
                        sympy.sin(x[0] ** 2) / x[0]


@register_eq_class
class Livermore2_Vars6_21(KnownEquation):
    _eq_name = 'Livermore2_Vars6_21'
    _function_set = ['add', 'sub', 'mul', 'div', 'log', 'exp', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] + x[2] + sympy.sin(-3.47 * x[1]) * sympy.log(0.5 + x[5]) / x[4] + x[3] + 25.56 * sympy.exp(x[4]) / x[
            1] * sympy.sin(x[1])


@register_eq_class
class Livermore2_Vars6_22(KnownEquation):
    _eq_name = 'Livermore2_Vars6_22'
    _function_set = ['add', 'sub', 'mul', 'cos', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] + (x[3] + sympy.sin(-0.22 * (x[2] - x[3] + 1.0)) * sympy.cos(x[5])) * sympy.cos(x[1] + 2.27 * x[4])


@register_eq_class
class Livermore2_Vars6_23(KnownEquation):
    _eq_name = 'Livermore2_Vars6_23'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] + x[3] + 2 * sympy.log(x[0]) + \
                        x[0] * (-x[5] + 1.88 * sympy.sqrt(0.71 * x[0] + x[1])) + \
                        0.28 * (x[2] - x[3] / x[4])


@register_eq_class
class Livermore2_Vars6_24(KnownEquation):
    _eq_name = 'Livermore2_Vars6_24'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = 0.24 * x[1] + \
                        1.42 * sympy.sqrt(x[2]) / (x[5] * sympy.sqrt(x[3] + x[4])) \
                        + sympy.sin(x[0]) / x[5]


@register_eq_class
class Livermore2_Vars6_25(KnownEquation):
    _eq_name = 'Livermore2_Vars6_25'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=6)
        x = self.x
        self.sympy_eq = x[0] - x[1] ** 2 - x[2] + \
                        x[4] * sympy.cos(x[2]) + x[4] + x[5] - \
                        2.19 * sympy.sqrt(x[2] + 0.44 / x[3])


####
@register_eq_class
class Livermore2_Vars7_1(KnownEquation):
    _eq_name = 'Livermore2_Vars7_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[5] * x[6] * (x[2] + sympy.cos(-4 * x[0] ** 2 + x[1] * x[2] * x[3] + x[4]))


@register_eq_class
class Livermore2_Vars7_2(KnownEquation):
    _eq_name = 'Livermore2_Vars7_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = -x[3] - x[4] * x[6] + sympy.sqrt(x[0]) - x[1] - x[2] - x[3] - 2 * x[4] * x[5]


@register_eq_class
class Livermore2_Vars7_3(KnownEquation):
    _eq_name = 'Livermore2_Vars7_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[0] + x[1] + x[2] + sympy.cos((x[1] + x[3] * (x[3] + x[4]) / x[5] - x[6]) ** 4 / x[6]) + 1


@register_eq_class
class Livermore2_Vars7_4(KnownEquation):
    _eq_name = 'Livermore2_Vars7_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[1] + x[4] + sympy.log(x[3] * sympy.sqrt(x[5]) * x[6] * (x[0] + x[2] ** 2 + x[4]))


@register_eq_class
class Livermore2_Vars7_5(KnownEquation):
    _eq_name = 'Livermore2_Vars7_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'const', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = (-0.12 * x[0] / (x[2] * x[4] ** (5 / 2) * x[5] * (-x[1] + x[2] * x[3] + sympy.cos(2 * x[0]))) + x[1]) * sympy.exp(
            -x[5] + x[6])


@register_eq_class
class Livermore2_Vars7_6(KnownEquation):
    _eq_name = 'Livermore2_Vars7_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[0] ** 2 - sympy.cos(4.69 * sympy.exp(x[1] * (x[2] / x[6] - x[5]) * (x[3] ** 3 * x[4] + x[3])))


@register_eq_class
class Livermore2_Vars7_7(KnownEquation):
    _eq_name = 'Livermore2_Vars7_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[0] / (97.02 * x[1] ** 2 * x[5] ** 4 + x[2] + x[5] * sympy.sin(x[6] / x[4])) + x[2] ** (1 / 4) - x[3] * x[
            6] - sympy.log(x[5]) ** 2


@register_eq_class
class Livermore2_Vars7_8(KnownEquation):
    _eq_name = 'Livermore2_Vars7_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = 4.73 * x[0] + sympy.cos(x[5] * sympy.sqrt(x[1] ** 2 * x[2] * (x[3] + x[5]) ** 2 / (x[4] * (x[1] + x[6]))))


@register_eq_class
class Livermore2_Vars7_9(KnownEquation):
    _eq_name = 'Livermore2_Vars7_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'n3', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[1] - x[5] ** 2 + 0.56 * (-x[0] + x[2] + x[3]) / (x[0] * x[1] ** 3 * x[4] * x[6])


@register_eq_class
class Livermore2_Vars7_10(KnownEquation):
    _eq_name = 'Livermore2_Vars7_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = -2.07 * x[6] + sympy.log(x[1]) - x[5] - sympy.sqrt(x[2] * x[4]) + sympy.log(x[0]) + (x[4] + 1) / x[3]


@register_eq_class
class Livermore2_Vars7_11(KnownEquation):
    _eq_name = 'Livermore2_Vars7_11'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'const', 'sin', 'cos']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[1] * (x[0] * sympy.cos(x[1] - x[3] + 4.52 + x[6] / (x[2] * x[5])) ** 4 / (x[1] * x[2] ** 2) + 2 * x[3] + x[4])


@register_eq_class
class Livermore2_Vars7_12(KnownEquation):
    _eq_name = 'Livermore2_Vars7_12'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[0] * (
                x[1] + x[3] + sympy.cos(sympy.exp(x[5] * x[6] * (x[3] + 0.43 * x[2] * (x[0] * x[1] + x[0]) / (x[0] * x[4])))))


@register_eq_class
class Livermore2_Vars7_13(KnownEquation):
    _eq_name = 'Livermore2_Vars7_13'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[3] + sympy.sin((x[0] * (x[0] * x[6] * (x[2] + x[5]) + 1.21) * sympy.exp(-x[4])))


@register_eq_class
class Livermore2_Vars7_14(KnownEquation):
    _eq_name = 'Livermore2_Vars7_14'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = 4.63 * x[0] ** 2 * x[5] + 1.31 * x[5] - sympy.cos(
            x[6] * (x[6] + (x[1] ** 2 * x[4] ** 2 * x[5] ** 2 * (x[0] + x[2] * x[3] * x[5]) ** 2 + x[5]) / x[0]))


@register_eq_class
class Livermore2_Vars7_15(KnownEquation):
    _eq_name = 'Livermore2_Vars7_15'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[2] * x[3] + sympy.log(x[0] * x[1]) - 3.69 / (
                -x[0] * sympy.exp(-4 * x[6]) + x[1] * x[4] - 1.99 + x[5] / sympy.sqrt(x[0]))


@register_eq_class
class Livermore2_Vars7_16(KnownEquation):
    _eq_name = 'Livermore2_Vars7_16'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[3] + sympy.cos(0.78 * x[0] * (x[2] + x[5] ** 2) * (x[1] * x[6] + x[2]) / (x[1] + 2.19) + x[3] ** 2 * x[4])


@register_eq_class
class Livermore2_Vars7_17(KnownEquation):
    _eq_name = 'Livermore2_Vars7_17'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = (-x[6] + sympy.cos(
            (x[0] * x[4] * (x[0] + 1.42) * (-sympy.sqrt(x[1] + x[5]) + (-x[3] + x[4] * x[6]) / x[2])))) / sympy.sqrt(x[4])


@register_eq_class
class Livermore2_Vars7_18(KnownEquation):
    _eq_name = 'Livermore2_Vars7_18'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[0] + sympy.sqrt(2) * sympy.sqrt(x[0] / x[4]) / 2 + sympy.cos(-x[1] * (-x[2] + 3.67 / x[0]) + x[3] + x[5] * x[6])


@register_eq_class
class Livermore2_Vars7_19(KnownEquation):
    _eq_name = 'Livermore2_Vars7_19'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = 2 * x[0] * x[1] + x[3] * sympy.exp(
            sympy.sqrt(x[1]) * (x[4] * (1.22 * x[1] * x[3] * x[5] + 2.65 * x[5]) * sympy.sin(x[6]) - x[5] - x[6]) + x[2])


@register_eq_class
class Livermore2_Vars7_20(KnownEquation):
    _eq_name = 'Livermore2_Vars7_20'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'exp', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[0] - sympy.exp(
            x[3] / (x[2] + x[3] * (x[4] + x[6] + sympy.exp(x[5]) / sympy.sqrt(x[2] + x[5])) + 3.42 * sympy.sqrt(x[4])))


@register_eq_class
class Livermore2_Vars7_21(KnownEquation):
    _eq_name = 'Livermore2_Vars7_21'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[0] * x[0] + x[1] + x[1] / 8.07 * x[0] ** 2 * x[1] * x[2] * x[3] * x[4] * (x[2] + x[3]) - x[4] + x[5] + x[0] + x[1]


@register_eq_class
class Livermore2_Vars7_22(KnownEquation):
    _eq_name = 'Livermore2_Vars7_22'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[0] + x[1] * x[4] + x[2] + 2.21 * sympy.sqrt(0.97 * x[3]) + sympy.exp(x[3] + x[5] + x[6])


@register_eq_class
class Livermore2_Vars7_23(KnownEquation):
    _eq_name = 'Livermore2_Vars7_23'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[0] * sympy.cos(x[0]) - sympy.sqrt(x[2]) * x[3] / (
                -14.13 * x[1] * x[2] * x[4] + 13.78 * x[1] * x[5] + x[2] + 13.04 * x[3] * x[5] * x[6] + x[4] +
                -x[5] + x[6]) + sympy.cos(x[1])


@register_eq_class
class Livermore2_Vars7_24(KnownEquation):
    _eq_name = 'Livermore2_Vars7_24'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = x[5] + 2 * x[6] + sympy.sin(x[0] * (x[0] + x[1] - 3.03)) + \
                        x[6] / (x[0] * x[2] ** 2) * sympy.sqrt(x[4]) * sympy.sin(x[3] ** 2)


@register_eq_class
class Livermore2_Vars7_25(KnownEquation):
    _eq_name = 'Livermore2_Vars7_25'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'n2', 'log', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=7)
        x = self.x
        self.sympy_eq = -1.16 * x[0] * sympy.log(sympy.sqrt(x[3]) + x[6]) - x[2] * sympy.cos(x[4]) / (
                x[0] * (x[5] + 0.95) + x[2] ** 2 * (x[1] + x[5]) ** 2)
