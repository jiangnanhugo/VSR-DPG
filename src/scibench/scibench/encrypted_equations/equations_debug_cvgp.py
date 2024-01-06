from collections import OrderedDict
import sympy
from base import KnownEquation

EQUATION_CLASS_DICT = OrderedDict()


def register_eq_class(cls):
    EQUATION_CLASS_DICT[cls.__name__] = cls
    return cls


def get_eq_obj(key, **kwargs):
    if key in EQUATION_CLASS_DICT:
        return EQUATION_CLASS_DICT[key](**kwargs)
    raise KeyError(f'`{key}` is not expected as a equation object key')


@register_eq_class
class Debug_1(KnownEquation):
    _eq_name = 'Debug_1'
    _function_set = ['add', 'mul', 'div', 'pow', 'inv', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=3)
        x = self.x

        self.sympy_eq = 30 * x[0] * x[2] / ((x[0] - 10) * x[1] * x[1])


@register_eq_class
class Debug_2(KnownEquation):
    _eq_name = 'Debug_2'
    _function_set = ['add', 'mul', 'div', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 30 * x[0] * x[1] / (x[0] - 10)


@register_eq_class
class Debug_3(KnownEquation):
    _eq_name = 'Debug_3'
    _function_set = ['add', 'mul', 'div', 'pow', 'inv', 'sqrt', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.sympy_eq = 30 * x[0] / ((x[0] - 10) * x[1])


@register_eq_class
class Debug_4(KnownEquation):
    _eq_name = 'Debug_4'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = 30 * x[0] / (x[0] - 10)


@register_eq_class
class Debug_5(KnownEquation):
    _eq_name = 'Debug_5'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = x[0] / (x[0] - 1)


@register_eq_class
class Debug_6(KnownEquation):
    _eq_name = 'Debug_6'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'cos', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = -2.1 * sympy.cos(9.8 * x[0]) + 2


@register_eq_class
class Debug_7(KnownEquation):
    _eq_name = 'Debug_7'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'cos', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = -2.1 * sympy.cos(9.8 * x[0])


#
@register_eq_class
class Debug_8(KnownEquation):
    _eq_name = 'Debug_8'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'cos', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = -sympy.cos(9.8 * x[0])


#
@register_eq_class
class Debug_9(KnownEquation):
    _eq_name = 'Debug_9'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'cos', 'sin', 'const']

    def __init__(self):
        super().__init__(num_vars=1)
        x = self.x
        self.sympy_eq = sympy.cos(9.8 * x[0])


@register_eq_class
class Debug_10(KnownEquation):
    _eq_name = 'Debug_10'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'const']

    def __init__(self):
        super().__init__(num_vars=2)
        x = self.x
        self.preorder_traversal = [
            ('mul', 'binary'), (9.8, 'const'), ('n2', 'unary'), ('sub', 'binary'), (str(x[0]), 'var'), (str(x[1]), 'var')
        ]
        self.sympy_eq = 9.8 * (x[0] - x[1]) ** 2


@register_eq_class
class Debug_11(KnownEquation):
    _eq_name = 'Debug_11'
    _function_set = ['add', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = 3 * x[0] * (4 * x[2] - 2) - 2 * (x[1] - 2) * x[3]


@register_eq_class
class Debug_12(KnownEquation):
    _eq_name = 'Debug_12'
    _function_set = ['add', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        super().__init__(num_vars=4)
        x = self.x
        self.sympy_eq = 3 * x[0] * x[2] - 2 * x[1] * x[3]

#
#
# @register_eq_class
# class Keijzer_11(KnownEquation):
#     _eq_name = 'Keijzer_11'
#     _function_set = ['add', 'mul', 'div', 'exp', 'log', 'pow', 'inv', 'sqrt', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=2)
#         x = self.x
#         self.sympy_eq = x[0] * x[1] + sympy.sin((x[0] - 1) * (x[1] - 1))
#
#
# @register_eq_class
# class Keijzer_12(KnownEquation):
#     _eq_name = 'Keijzer_12'
#     _function_set = ['add', 'mul', 'div', 'exp', 'log', 'pow', 'inv', 'sqrt', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=2)
#         x = self.x
#         self.sympy_eq = sympy.Pow(x[0], 4) - sympy.Pow(x[0], 3) + sympy.Pow(x[1], 2) / 2 - x[1]
#
#
# @register_eq_class
# class Keijzer_13(KnownEquation):
#     _eq_name = 'Keijzer_13'
#     _function_set = ['add', 'mul', 'div', 'exp', 'log', 'pow', 'inv', 'sqrt', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=2)
#         x = self.x
#         self.sympy_eq = 6 * sympy.sin(x[0]) * sympy.cos(x[1])
#
#
# @register_eq_class
# class Keijzer_14(KnownEquation):
#     _eq_name = 'Keijzer_14'
#     _function_set = ['add', 'mul', 'div', 'exp', 'log', 'pow', 'inv', 'sqrt', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=2)
#         x = self.x
#         self.sympy_eq = 8 / (2 + sympy.Pow(x[0], 2) + sympy.Pow(x[1], 2))
#
#
# @register_eq_class
# class Keijzer_15(KnownEquation):
#     _eq_name = 'Keijzer_15'
#     _function_set = ['add', 'mul', 'div', 'exp', 'log', 'pow', 'inv', 'sqrt', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=2)
#         x = self.x
#         self.sympy_eq = (sympy.Pow(x[0], 3)) / (5) + (sympy.Pow(x[1], 3)) / (2) - x[1] - x[0]
#
#
# @register_eq_class
# class Korns_1(KnownEquation):
#     _eq_name = 'Korns_1'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 1.57 + 24.3 * x[3]
#
#
# @register_eq_class
# class Korns_2(KnownEquation):
#     _eq_name = 'Korns_2'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 0.23 + 14.2 * ((x[3] + x[1])) / ((3 * x[4]))
#
#
# @register_eq_class
# class Korns_3(KnownEquation):
#     _eq_name = 'Korns_3'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 4.9 * ((x[3] - x[0] + (x[1]) / (x[4]))) / ((3 * x[4])) - 5.41
#
#
# @register_eq_class
# class Korns_4(KnownEquation):
#     _eq_name = 'Korns_4'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 0.13 * sympy.sin(x[2]) - 2.3
#
#
# @register_eq_class
# class Korns_5(KnownEquation):
#     _eq_name = 'Korns_5'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 3 + 2.13 * sympy.log(abs(x[4]))
#
#
# @register_eq_class
# class Korns_6(KnownEquation):
#     _eq_name = 'Korns_6'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 1.3 + 0.13 * sympy.sqrt(abs(x[0]))
#
#
# @register_eq_class
# class Korns_7(KnownEquation):
#     _eq_name = 'Korns_7'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 2.1380940889 * (1 - sympy.exp(-0.54723748542 * x[0]))
#
#
# @register_eq_class
# class Korns_8(KnownEquation):
#     _eq_name = 'Korns_8'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 6.87 + 11 * sympy.sqrt(abs(7.23 * x[0] * x[3] * x[4]))
#
#
# @register_eq_class
# class Korns_9(KnownEquation):
#     _eq_name = 'Korns_9'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = (sympy.sqrt(abs(x[0]))) / (sympy.log(abs(x[1]))) * (sympy.exp(x[2])) / (sympy.Pow(x[3], 2))
#
#
# @register_eq_class
# class Korns_10(KnownEquation):
#     _eq_name = 'Korns_10'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 0.81 + 24.3 * (2 * x[1] + 3 * sympy.Pow(x[2], 2)) / (((4 * sympy.Pow(x[3], 3) + 5 * sympy.Pow(x[4], 4))))
#
#
# @register_eq_class
# class Korns_11(KnownEquation):
#     _eq_name = 'Korns_11'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 6.87 + 11 * sympy.cos(7.23 * sympy.Pow(x[0], 3))
#
#
# @register_eq_class
# class Korns_12(KnownEquation):
#     _eq_name = 'Korns_12'
#     _function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'sqrt', 'tan', 'tanh', 'const']
#
#     def __init__(self):
#         super().__init__(num_vars=5)
#         x = self.x
#         self.sympy_eq = 2 - 2.1 * sympy.cos(9.8 * x[0]) * sympy.sin(1.3 * x[4])
#
#
