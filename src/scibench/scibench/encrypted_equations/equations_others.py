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
class Keijzer_1(KnownEquation):
    _eq_name = 'Keijzer_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = 0.3 * x[0] * sympy.sin(2 * sympy.pi * x[0])


@register_eq_class
class Keijzer_2(KnownEquation):
    _eq_name = 'Keijzer_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = 0.3 * x[0] * sympy.sin(2 * sympy.pi * x[0])


@register_eq_class
class Keijzer_3(KnownEquation):
    _eq_name = 'Keijzer_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = 0.3 * x[0] * sympy.sin(2 * sympy.pi * x[0])


@register_eq_class
class Keijzer_4(KnownEquation):
    _eq_name = 'Keijzer_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log',  'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = x[0] * x[0] * x[0] * sympy.exp(-x[0]) * sympy.cos(x[0]) * sympy.sin(x[0]) * (
                sympy.sin(x[0]) * sympy.sin(x[0]) * sympy.cos(x[0]) - 1)


@register_eq_class
class Keijzer_5(KnownEquation):
    _eq_name = 'Keijzer_5'
    _function_set = ['add', 'sub', 'mul', 'div',  'inv', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = (30 * x[0] * x[2]) / ((x[0] - 10) * x[1] * x[1])


@register_eq_class
class Keijzer_6(KnownEquation):
    _eq_name = 'Keijzer_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log',  'inv', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = (x[0] * (x[0] + 1)) / 2


@register_eq_class
class Keijzer_7(KnownEquation):
    _eq_name = 'Keijzer_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log',  'inv', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.log(x[0])


@register_eq_class
class Keijzer_8(KnownEquation):
    _eq_name = 'Keijzer_8'
    _function_set = ['add', 'sub', 'mul', 'div',  'inv', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sqrt(x[0])


@register_eq_class
class Keijzer_9(KnownEquation):
    _eq_name = 'Keijzer_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'log',  'inv', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.log(x[0] + sympy.sqrt(x[0] * x[0] + 1))


@register_eq_class
class Keijzer_10(KnownEquation):
    _eq_name = 'Keijzer_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp',  'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], x[1])


@register_eq_class
class Keijzer_11(KnownEquation):
    _eq_name = 'Keijzer_11'
    _function_set = ['add', 'sub', 'mul', 'div',  'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = x[0] * x[1] + sympy.sin((x[0] - 1) * (x[1] - 1))


@register_eq_class
class Keijzer_12(KnownEquation):
    _eq_name = 'Keijzer_12'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp',  'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 4) - sympy.Pow(x[0], 3) + sympy.Pow(x[1], 2) / 2 - x[1]


@register_eq_class
class Keijzer_13(KnownEquation):
    _eq_name = 'Keijzer_13'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 6 * sympy.sin(x[0]) * sympy.cos(x[1])


@register_eq_class
class Keijzer_14(KnownEquation):
    _eq_name = 'Keijzer_14'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp',  'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 8 / (2 + sympy.Pow(x[0], 2) + sympy.Pow(x[1], 2))


@register_eq_class
class Keijzer_15(KnownEquation):
    _eq_name = 'Keijzer_15'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp',  'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (sympy.Pow(x[0], 3)) / (5) + (sympy.Pow(x[1], 3)) / (2) - x[1] - x[0]


@register_eq_class
class Korns_1(KnownEquation):
    _eq_name = 'Korns_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 1.57 + 24.3 * x[3]


@register_eq_class
class Korns_2(KnownEquation):
    _eq_name = 'Korns_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 0.23 + 14.2 * (x[3] + x[1]) / (3 * x[4])


@register_eq_class
class Korns_3(KnownEquation):
    _eq_name = 'Korns_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 4.9 * (x[3] - x[0] + (x[1]) / (x[4])) / (3 * x[4]) - 5.41


@register_eq_class
class Korns_4(KnownEquation):
    _eq_name = 'Korns_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 0.13 * sympy.sin(x[2]) - 2.3


@register_eq_class
class Korns_5(KnownEquation):
    _eq_name = 'Korns_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'abs', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 3 + 2.13 * sympy.log(abs(x[4]))


@register_eq_class
class Korns_6(KnownEquation):
    _eq_name = 'Korns_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'abs', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 1.3 + 0.13 * sympy.sqrt(abs(x[0]))


@register_eq_class
class Korns_7(KnownEquation):
    _eq_name = 'Korns_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 2.1380940889 * (1 - sympy.exp(-0.54723748542 * x[0]))


@register_eq_class
class Korns_8(KnownEquation):
    _eq_name = 'Korns_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'abs', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 6.87 + 11 * sympy.sqrt(abs(7.23 * x[0] * x[3] * x[4]))


@register_eq_class
class Korns_9(KnownEquation):
    _eq_name = 'Korns_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log', 'n2', 'sqrt', 'abs']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = (sympy.sqrt(abs(x[0]))) / (sympy.log(abs(x[1]))) * (sympy.exp(x[2])) / (sympy.Pow(x[3], 2))


@register_eq_class
class Korns_10(KnownEquation):
    _eq_name = 'Korns_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'n3', 'n4', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 0.81 + 24.3 * (2 * x[1] + 3 * sympy.Pow(x[2], 2)) / (4 * sympy.Pow(x[3], 3) + 5 * sympy.Pow(x[4], 4))


@register_eq_class
class Korns_11(KnownEquation):
    _eq_name = 'Korns_11'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'n2', 'n3', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 6.87 + 11 * sympy.cos(7.23 * sympy.Pow(x[0], 3))


@register_eq_class
class Korns_12(KnownEquation):
    _eq_name = 'Korns_12'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 2 - 2.1 * sympy.cos(9.8 * x[0]) * sympy.sin(1.3 * x[4])


@register_eq_class
class Koza_2(KnownEquation):
    _eq_name = 'Koza_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv',  'exp', 'log']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 5) - 2 * sympy.Pow(x[0], 3) + x[0]


@register_eq_class
class Koza_3(KnownEquation):
    _eq_name = 'Koza_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 6) - 2 * sympy.Pow(x[0], 4) + sympy.Pow(x[0], 2)


@register_eq_class
class Meier_3(KnownEquation):
    _eq_name = 'Meier_3'
    _function_set = ['add', 'sub', 'mul', 'div',  'log']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (sympy.Pow(x[0], 2) * sympy.Pow(x[1], 2)) / ((x[0] + x[1]))


@register_eq_class
class Meier_4(KnownEquation):
    _eq_name = 'Meier_4'
    _function_set = ['add', 'sub', 'mul', 'div',  'exp', 'log']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (sympy.Pow(x[0], 5)) / (sympy.Pow(x[1], 3))


@register_eq_class
class Nguyen_1(KnownEquation):
    _eq_name = 'Nguyen_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'n3', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Nguyen_2(KnownEquation):
    _eq_name = 'Nguyen_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'n3', 'n4', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 4) + sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Nguyen_3(KnownEquation):
    _eq_name = 'Nguyen_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'n3', 'n4', 'n5', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 5) + sympy.Pow(x[0], 4) + sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Nguyen_4(KnownEquation):
    _eq_name = 'Nguyen_4'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 6) + sympy.Pow(x[0], 5) + sympy.Pow(x[0], 4) + sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Nguyen_5(KnownEquation):
    _eq_name = 'Nguyen_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sin(sympy.Pow(x[0], 2)) * sympy.cos(x[0]) - 1


@register_eq_class
class Nguyen_6(KnownEquation):
    _eq_name = 'Nguyen_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sin(x[0]) + sympy.sin(x[0] + sympy.Pow(x[0], 2))


@register_eq_class
class Nguyen_7(KnownEquation):
    _eq_name = 'Nguyen_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.log(x[0] + 1) + sympy.log(sympy.Pow(x[0], 2) + 1)


@register_eq_class
class Nguyen_8(KnownEquation):
    _eq_name = 'Nguyen_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sqrt(x[0])


@register_eq_class
class Nguyen_9(KnownEquation):
    _eq_name = 'Nguyen_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.sin(x[0]) + sympy.sin(sympy.Pow(x[1], 2))


@register_eq_class
class Nguyen_10(KnownEquation):
    _eq_name = 'Nguyen_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 2 * sympy.sin(x[0]) * sympy.cos(x[1])


@register_eq_class
class Nguyen_11(KnownEquation):
    _eq_name = 'Nguyen_11'
    _function_set = ['add', 'sub', 'mul', 'div',  'exp']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], x[1])


@register_eq_class
class Nguyen_12(KnownEquation):
    _eq_name = 'Nguyen_12'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'n2', 'n3', 'n4', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 4) - sympy.Pow(x[0], 3) + (sympy.Pow(x[1], 2)) / (2) - x[1]


@register_eq_class
class Nguyen_12a(KnownEquation):
    _eq_name = 'Nguyen_12a'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'n2', 'n3', 'n4', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 4) - sympy.Pow(x[0], 3) + sympy.Pow(x[1], 2) / 2 - x[1]


@register_eq_class
class Constant_1(KnownEquation):
    _eq_name = 'Constant_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'n3', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = 3.39 * sympy.Pow(x[0], 3) + 2.12 * sympy.Pow(x[0], 2) + 1.78 * x[0]


@register_eq_class
class Constant_2(KnownEquation):
    _eq_name = 'Constant_2'
    _function_set = ['add', 'sub', 'mul', 'sin', 'cos', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sin(sympy.Pow(x[0], 2)) * sympy.cos(x[0]) - 0.75


@register_eq_class
class Constant_3(KnownEquation):
    _eq_name = 'Constant_3'
    _function_set = ['add', 'sub', 'mul', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.sin(1.5 * x[0]) * sympy.cos(0.5 * x[1])


@register_eq_class
class Constant_4(KnownEquation):
    _eq_name = 'Constant_4'
    _function_set = ['add', 'mul',  'const']
    expr_obj_thres = 1
    expr_consts_thres = None

    def __init__(self, vars_range_and_types=None):
        if vars_range_and_types is None:
            vars_range_and_types = [LogUniformSampling(0.1, 10, only_positive=True), LogUniformSampling(0.1, 5, only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=vars_range_and_types)
        x = self.x
        self.sympy_eq = 2.7 * sympy.Pow(x[0], x[1])


@register_eq_class
class Constant_5(KnownEquation):
    _eq_name = 'Constant_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sqrt(1.23 * x[0])


@register_eq_class
class Constant_6(KnownEquation):
    _eq_name = 'Constant_6'
    _function_set = ['add', 'sub', 'mul', 'div',  'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 0.426)


@register_eq_class
class Constant_7(KnownEquation):
    _eq_name = 'Constant_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 2 * sympy.sin(1.3 * x[0]) * sympy.cos(x[1])


@register_eq_class
class Constant_8(KnownEquation):
    _eq_name = 'Constant_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.log(x[0] + 1.4) + sympy.log(sympy.Pow(x[0], 2) + 1.3)


@register_eq_class
class Livermore_1(KnownEquation):
    _eq_name = 'Livermore_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = 1. / 3 + x[0] + sympy.sin(x[0] * x[0])


@register_eq_class
class Livermore_2(KnownEquation):
    _eq_name = 'Livermore_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sin(sympy.Pow(x[0], 2)) * sympy.cos(x[0]) - 2


@register_eq_class
class Livermore_3(KnownEquation):
    _eq_name = 'Livermore_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sin(sympy.Pow(x[0], 3)) * sympy.cos(sympy.Pow(x[0], 2)) - 1


@register_eq_class
class Livermore_4(KnownEquation):
    _eq_name = 'Livermore_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.log(x[0] + 1) + sympy.log(sympy.Pow(x[0], 2) + 1) + sympy.log(x[0])


@register_eq_class
class Livermore_5(KnownEquation):
    _eq_name = 'Livermore_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 4) - sympy.Pow(x[0], 3) + sympy.Pow(x[1], 2) - x[1]


@register_eq_class
class Livermore_6(KnownEquation):
    _eq_name = 'Livermore_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = 4 * sympy.Pow(x[0], 4) + 3 * sympy.Pow(x[0], 3) + 2 * sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Livermore_7(KnownEquation):
    _eq_name = 'Livermore_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = (sympy.exp(x[0]) - sympy.exp(-1 * x[0])) / (2)


@register_eq_class
class Livermore_7a(KnownEquation):
    _eq_name = 'Livermore_7a'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = (sympy.exp(x[0]) - sympy.exp(-1 * x[0])) / (2)


@register_eq_class
class Livermore_8(KnownEquation):
    _eq_name = 'Livermore_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = (sympy.exp(x[0]) + sympy.exp(-1 * x[0])) / (2)


@register_eq_class
class Livermore_8a(KnownEquation):
    _eq_name = 'Livermore_8a'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = (sympy.exp(x[0]) + sympy.exp(-1 * x[0])) / (2)


@register_eq_class
class Livermore_9(KnownEquation):
    _eq_name = 'Livermore_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 9) + sympy.Pow(x[0], 8) + sympy.Pow(x[0], 7) + sympy.Pow(x[0], 6) + sympy.Pow(x[0], 5) + sympy.Pow(
            x[0], 4) + sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Livermore_10(KnownEquation):
    _eq_name = 'Livermore_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 6 * sympy.sin(x[0]) * sympy.cos(x[1])


@register_eq_class
class Livermore_11(KnownEquation):
    _eq_name = 'Livermore_11'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'exp', 'pow']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (sympy.Pow(x[0], 2) * sympy.Pow(x[1], 2)) / ((x[0] + x[1]))


@register_eq_class
class Livermore_12(KnownEquation):
    _eq_name = 'Livermore_12'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (sympy.Pow(x[0], 5)) / (sympy.Pow(x[1], 3))


@register_eq_class
class Livermore_13(KnownEquation):
    _eq_name = 'Livermore_13'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 1 / 3)


@register_eq_class
class Livermore_14(KnownEquation):
    _eq_name = 'Livermore_14'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'n3', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0] + sympy.sin(x[0]) + sympy.sin(sympy.Pow(x[1], 2))


@register_eq_class
class Livermore_15(KnownEquation):
    _eq_name = 'Livermore_15'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 1 / 5)


@register_eq_class
class Livermore_16(KnownEquation):
    _eq_name = 'Livermore_16'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 2 / 3)


@register_eq_class
class Livermore_17(KnownEquation):
    _eq_name = 'Livermore_17'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 4 * sympy.sin(x[0]) * sympy.cos(x[1])


@register_eq_class
class Livermore_18(KnownEquation):
    _eq_name = 'Livermore_18'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sin(sympy.Pow(x[0], 2)) * sympy.cos(x[0]) - 5


@register_eq_class
class Livermore_19(KnownEquation):
    _eq_name = 'Livermore_19'
    _function_set = ['add', 'sub', 'mul', 'div', 'const', 'n2', 'n3', 'log']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.log(sympy.Pow(x[0], 2) + x[0]) + sympy.log(sympy.Pow(x[0], 3) + x[0])


@register_eq_class
class Livermore_20(KnownEquation):
    _eq_name = 'Livermore_20'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'n4',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 5) + sympy.Pow(x[0], 4) + sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Livermore_21(KnownEquation):
    _eq_name = 'Livermore_21'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.exp(-1 * sympy.Pow(x[0], 2))


@register_eq_class
class Livermore_22(KnownEquation):
    _eq_name = 'Livermore_22'
    _function_set = ['add', 'sub', 'mul', 'div', 'pow']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 8) + sympy.Pow(x[0], 7) + sympy.Pow(x[0], 6) + sympy.Pow(x[0], 5) + sympy.Pow(x[0], 4) + sympy.Pow(
            x[0], 3) + sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Livermore_23(KnownEquation):
    _eq_name = 'Livermore_23'
    _function_set = ['add', 'sub', 'mul', 'div', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.exp(-0.5 * sympy.Pow(x[0], 2))


@register_eq_class
class Pagie_1(KnownEquation):
    _eq_name = 'Pagie_1'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 1 / (1 + sympy.Pow(x[0], -4)) + 1 / ((1 + sympy.Pow(x[1], -4)))


@register_eq_class
class Nonic(KnownEquation):
    _eq_name = 'Nonic'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 9) + sympy.Pow(x[0], 8) + sympy.Pow(x[0], 7) + sympy.Pow(x[0], 6) + sympy.Pow(x[0], 5) + sympy.Pow(
            x[0], 4) + sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Poly_10(KnownEquation):
    _eq_name = 'Poly_10'
    _function_set = ['add', 'sub', 'mul', 'div']

    def __init__(self):
        super().__init__(num_vars=10)
        x = self.x
        self.sympy_eq = x[0] * x[1] + x[2] * x[3] + x[4] * x[5] + x[0] * x[6] * x[8] + x[2] * x[5] * x[9]


@register_eq_class
class R1(KnownEquation):
    _eq_name = 'R1'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = (sympy.Pow(x[0] + 1, 3)) / (sympy.Pow(x[0], 2) - x[0] + 1)


@register_eq_class
class R2(KnownEquation):
    _eq_name = 'R2'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = ((sympy.Pow(x[0], 5) - 3 * sympy.Pow(x[0], 3) + 1)) / ((sympy.Pow(x[0], 2) + 1))


@register_eq_class
class R3(KnownEquation):
    _eq_name = 'R3'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = ((sympy.Pow(x[0], 6) + sympy.Pow(x[0], 5))) / (
            (sympy.Pow(x[0], 4) + sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0] + 1))


@register_eq_class
class R1a(KnownEquation):
    _eq_name = 'R1a'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = (sympy.Pow(x[0] + 1, 3)) / (sympy.Pow(x[0], 2) - x[0] + 1)


@register_eq_class
class R2a(KnownEquation):
    _eq_name = 'R2a'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = ((sympy.Pow(x[0], 5) - 3 * sympy.Pow(x[0], 3) + 1)) / ((sympy.Pow(x[0], 2) + 1))


@register_eq_class
class R3a(KnownEquation):
    _eq_name = 'R3a'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = ((sympy.Pow(x[0], 6) + sympy.Pow(x[0], 5))) / (
            (sympy.Pow(x[0], 4) + sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0] + 1))


@register_eq_class
class Sine(KnownEquation):
    _eq_name = 'Sine'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos',  'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sin(x[0]) + sympy.sin(x[0] + sympy.Pow(x[0], 2))


@register_eq_class
class Vladislavleva_1(KnownEquation):
    _eq_name = 'Vladislavleva_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'exp', 'expneg', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (sympy.exp(-sympy.Pow(x[0] - 1, 2))) / (1.2 + sympy.Pow((x[1] - 2.5), 2))


@register_eq_class
class Vladislavleva_2(KnownEquation):
    _eq_name = 'Vladislavleva_2'
    _function_set = ['add', 'sub', 'mul', 'n2', 'n3', 'exp', 'expneg', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.exp(-x[0]) * sympy.Pow(x[0], 3) * sympy.cos(x[0]) * sympy.sin(x[0]) * (
                sympy.cos(x[0]) * sympy.Pow(sympy.sin(x[0]), 2) - 1)


@register_eq_class
class Vladislavleva_3(KnownEquation):
    _eq_name = 'Vladislavleva_3'
    _function_set = ['add', 'sub', 'mul', 'n2', 'n3', 'exp', 'expneg', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.exp(-x[0]) * sympy.Pow(x[0], 3) * sympy.cos(x[0]) * sympy.sin(x[0]) * (
                sympy.cos(x[0]) * sympy.Pow(sympy.sin(x[0]), 2) - 1) * (x[1] - 5)


@register_eq_class
class Vladislavleva_4(KnownEquation):
    _eq_name = 'Vladislavleva_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=5)
        x = self.x
        self.sympy_eq = 10 / (5 + (sympy.Pow((x[0] - 3), 2) + sympy.Pow((x[1] - 3), 2) + sympy.Pow((x[2] - 3), 2) +
                                   sympy.Pow((x[3] - 3), 2) + sympy.Pow((x[4] - 3), 2)))


@register_eq_class
class Vladislavleva_5(KnownEquation):
    _eq_name = 'Vladislavleva_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = 30 * (x[0] - 1) * (x[2] - 1) / ((x[0] - 10) * sympy.Pow(x[1], 2))


@register_eq_class
class Vladislavleva_6(KnownEquation):
    _eq_name = 'Vladislavleva_6'
    _function_set = ['add', 'sub', 'mul', 'cos', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 6 * sympy.sin(x[0]) * sympy.cos(x[1])


@register_eq_class
class Vladislavleva_7(KnownEquation):
    _eq_name = 'Vladislavleva_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (x[0] - 3) * (x[1] - 3) + 2 * sympy.sin(x[0] - 4) * (x[1] - 4)


@register_eq_class
class Vladislavleva_8(KnownEquation):
    _eq_name = 'Vladislavleva_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'n4', 'n3', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = (sympy.Pow((x[0] - 3), 4) + sympy.Pow((x[1] - 3), 3) - (x[1] - 3)) / (sympy.Pow((x[1] - 2), 4) + 10)


@register_eq_class
class Jin_1(KnownEquation):
    _eq_name = 'Jin_1'
    _function_set = ['add', 'sub', 'mul', 'n2', 'n3', 'n4', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 2.5 * sympy.Pow(x[0], 4) - 1.3 * sympy.Pow(x[0], 3) + 0.5 * sympy.Pow(x[1], 2) - 1.7 * x[1]


@register_eq_class
class Jin_2(KnownEquation):
    _eq_name = 'Jin_2'
    _function_set = ['add', 'sub', 'mul', 'n2', 'n3', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 8.0 * sympy.Pow(x[0], 2) + 8.0 * sympy.Pow(x[1], 3) - 15.0


@register_eq_class
class Jin_3(KnownEquation):
    _eq_name = 'Jin_3'
    _function_set = ['add', 'sub', 'mul', 'n2', 'n3', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 0.2 * sympy.Pow(x[0], 3) + 0.5 * sympy.Pow(x[1], 3) - 1.2 * x[1] - 0.5 * x[0]


@register_eq_class
class Jin_4(KnownEquation):
    _eq_name = 'Jin_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 1.5 * sympy.exp(x[0]) + 5.0 * sympy.cos(x[1])


@register_eq_class
class Jin_5(KnownEquation):
    _eq_name = 'Jin_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 6.0 * sympy.sin(x[0]) * sympy.cos(x[1])


@register_eq_class
class Jin_6(KnownEquation):
    _eq_name = 'Jin_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 1.35 * x[0] * x[1] + 5.5 * sympy.sin((x[0] - 1.0) * (x[1] - 1.0))


@register_eq_class
class Neat_1(KnownEquation):
    _eq_name = 'Neat_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'n3', 'n4']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 4) + sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Neat_2(KnownEquation):
    _eq_name = 'Neat_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'n3', 'n4', 'n5']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 5) + sympy.Pow(x[0], 4) + sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0]


@register_eq_class
class Neat_3(KnownEquation):
    _eq_name = 'Neat_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'n2']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sin(sympy.Pow(x[0], 2)) * sympy.cos(x[0]) - 1


@register_eq_class
class Neat_4(KnownEquation):
    _eq_name = 'Neat_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'log']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.log(x[0] + 1) + sympy.log(sympy.Pow(x[0], 2) + 1)


@register_eq_class
class Neat_5(KnownEquation):
    _eq_name = 'Neat_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 2 * sympy.sin(x[0]) * sympy.cos(x[1])


@register_eq_class
class Neat_6(KnownEquation):
    _eq_name = 'Neat_6'
    _function_set = ['add', 'mul', 'inv', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.harmonic(x[0])


@register_eq_class
class Neat_7(KnownEquation):
    _eq_name = 'Neat_7'
    _function_set = ['add', 'sub', 'mul', 'sin', 'cos', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 2 - 2.1 * sympy.cos(9.8 * x[0]) * sympy.sin(1.3 * x[1])


@register_eq_class
class Neat_8(KnownEquation):
    _eq_name = 'Neat_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'exp', 'expneg', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.exp(-(x[0] - 1) ** 2) / (1.2 + (x[1] - 2.5) ** 2)


@register_eq_class
class Neat_9(KnownEquation):
    _eq_name = 'Neat_9'
    _function_set = ['add', 'sub', 'mul', 'div',  'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 1 / (1 + sympy.Pow(x[0], -4)) + 1 / (1 + sympy.Pow(x[1], -4))


@register_eq_class
class GrammarVAE_1(KnownEquation):
    _eq_name = 'GrammarVAE_1'
    _function_set = ['add', 'mul', 'div', 'sin', 'exp', 'pow']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = 1. / 3 + x[0] + sympy.sin(sympy.Pow(x[0], 2))


@register_eq_class
class Const_Test_1(KnownEquation):
    _eq_name = 'Const_Test_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = 3.14159265358979323846 * x[0] * x[0]


@register_eq_class
class Const_Test_2(KnownEquation):
    _eq_name = 'Const_Test_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = 2.178 * x[0] * x[0]


@register_eq_class
class Poly_1(KnownEquation):
    _eq_name = 'Poly_1'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'exp', 'log', 'sqrt']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x
        self.sympy_eq = x[1] / sympy.sqrt(sympy.Pow(x[0], 2) + sympy.Pow(x[1], 2) + sympy.Pow(x[2], 2))


@register_eq_class
class Poly_2(KnownEquation):
    _eq_name = 'Poly_2'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp',  'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.Pow(x[0], 3) + sympy.Pow(x[0], 2) + x[0] + sympy.sin(x[0]) + sympy.sin(sympy.Pow(x[1], 2))


@register_eq_class
class Poly_3(KnownEquation):
    _eq_name = 'Poly_3'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'sin', 'cos', 'exp', 'log', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = sympy.cos(x[1]) / (sympy.sqrt(12 * x[0] * x[1] + 1.3 + x[0] - 0.05 * sympy.Pow(x[1], 2)) + x[0])


@register_eq_class
class Poly_4(KnownEquation):
    _eq_name = 'Poly_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'sin', 'cos', 'exp', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=10)
        x = self.x
        self.sympy_eq = sympy.sin(x[3]) / (sympy.sqrt(12 * x[0] * x[1] + 1.3 - 0.05 * x[2] * x[5] * x[9]) * sympy.exp(x[6]))


@register_eq_class
class Poly_5(KnownEquation):
    _eq_name = 'Poly_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.sin(x[0]**3 - x[0] - sympy.pi / 6)
