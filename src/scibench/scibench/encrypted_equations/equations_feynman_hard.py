from collections import OrderedDict
import numpy as np
from base import KnownEquation, LogUniformSampling, IntegerUniformSampling, UniformSampling
import sympy

GRAVITATIONAL_CONSTANT = 6.67430e-11
GRAVITATIONAL_ACCELERATION = 9.80665
SPEED_OF_LIGHT = 2.99792458e8
ELECTRIC_CONSTANT = 8.854e-12
PLANCK_CONSTANT = 6.626e-34
BOLTZMANN_CONSTANT = 1.380649e-23
BOHR_MAGNETON = 9.2740100783e-24
DIRAC_CONSTANT = 1.054571817e-34
ELECTRON_MASS = 9.10938356e-31
FINE_STRUCTURE_CONSTANT = 7.2973525693e-3

EQUATION_CLASS_DICT = OrderedDict()


def register_eq_class(cls):
    EQUATION_CLASS_DICT[cls.__name__] = cls
    return cls


def get_eq_obj(key, **kwargs):
    if key in EQUATION_CLASS_DICT:
        return EQUATION_CLASS_DICT[key](**kwargs)
    raise KeyError(f'`{key}` is not expected as a equation object key')


@register_eq_class
class FeynmanICh50Eq261(KnownEquation):
    """
    - Equation: I.50.26
    - Raw: x1 * (cos(omega * t) + alpha * cos(omega * t) ** 2)
    - Num. Vars: 4
    - Vars:
        - x[0]: x1 (float, positive)
        - x[1]: omega (float)
        - x[2]: t (float, positive)
        - x[3]: alpha (float)
    - Constraints:
    """
    _eq_name = 'feynman-i.50.26.1'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        vars_range_and_types = [
            LogUniformSampling(1.0e-1, 1.0e1, only_positive=True), LogUniformSampling(1.0e1, 1.0e3, only_positive=True),
            LogUniformSampling(1.0e-3, 1.0e-1, only_positive=True), LogUniformSampling(1.0e-3, 1.0e-1, only_positive=True)
        ]

        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)
        x = self.x
        c = np.random.randn(2)
        self.sympy_eq = x[0] * (sympy.cos(c[0] * x[2]) + x[3] * sympy.cos(c[1] * x[2]) ** 2)


@register_eq_class
class FeynmanICh50Eq2621(KnownEquation):
    """
    - Equation: I.50.26
    - Raw: x1 * (cos(omega * t) + alpha * cos(omega * t) ** 2)
    - Num. Vars: 4
    - Vars:
        - x[0]: x1 (float, positive)
        - x[1]: omega (float)
        - x[2]: t (float, positive)
        - x[3]: alpha (float)
    - Constraints:
    """
    _eq_name = 'feynman-i.50.26.2.1'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        vars_range_and_types = [
            LogUniformSampling(1.0e-1, 1.0e1, only_positive=True), LogUniformSampling(1.0e1, 1.0e3, only_positive=True),
            LogUniformSampling(1.0e-3, 1.0e-1, only_positive=True), LogUniformSampling(1.0e-3, 1.0e-1, only_positive=True)
        ]

        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)
        x = self.x
        c = np.random.randn(3)
        self.sympy_eq = x[0] * (sympy.cos(c[0] * x[2]) + c[2] * sympy.cos(c[1] * x[2]) ** 2)


@register_eq_class
class FeynmanICh50Eq2632(KnownEquation):
    """
    - Equation: I.50.26
    - Raw: x1 * (cos(omega * t) + alpha * cos(omega * t) ** 2)
    - Num. Vars: 4
    - Vars:
        - x[0]: x1 (float, positive)
        - x[1]: omega (float)
        - x[2]: t (float, positive)
        - x[3]: alpha (float)
    - Constraints:
    """
    _eq_name = 'feynman-i.50.26.3.2'
    _function_set = ['add', 'sub', 'mul', 'div', 'n2', 'sin', 'cos', 'const']

    def __init__(self):
        vars_range_and_types = [
            LogUniformSampling(1.0e-1, 1.0e1, only_positive=True), LogUniformSampling(1.0e1, 1.0e3, only_positive=True),
            LogUniformSampling(1.0e-3, 1.0e-1, only_positive=True), LogUniformSampling(1.0e-3, 1.0e-1, only_positive=True)
        ]

        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)
        x = self.x
        c = np.random.randn(3)
        self.sympy_eq = c[2] * (sympy.cos(c[0] * x[2]) + x[3] * sympy.cos(c[1] * x[2]) ** 2)


###
@register_eq_class
class FeynmanIICh35Eq211(KnownEquation):
    """
    - Equation: II.35.21
    - Raw: n_rho * mom * tanh(mom * B / (1.380649e-23 * T))
    - Num. Vars: 4
    - Vars:
        - x[0]: n_rho (integer -> real due to its order, positive)
        - x[1]: mom (float, positive)
        - x[2]: B (float, positive)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0
    """
    _eq_name = 'feynman-ii.35.21.1'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        vars_range_and_types = [
            LogUniformSampling(1.0e-2, 1.0e1, only_positive=True),
            LogUniformSampling(1.0e-2, 1.0e1, only_positive=True),
            LogUniformSampling(1.0e-2, 1.0e-1, only_positive=True),
            LogUniformSampling(1.0e-2, 1.0e1, only_positive=True)
        ]

        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)
        x = self.x
        c = np.random.randn(1)
        self.sympy_eq = x[0] * x[1] * sympy.tanh(c[0] * x[2] / x[3])

@register_eq_class
class FeynmanIICh35Eq212(KnownEquation):
    """
    - Equation: II.35.21
    - Raw: n_rho * mom * tanh(mom * B / (1.380649e-23 * T))
    - Num. Vars: 4
    - Vars:
        - x[0]: n_rho (integer -> real due to its order, positive)
        - x[1]: mom (float, positive)
        - x[2]: B (float, positive)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0
    """
    _eq_name = 'feynman-ii.35.21.2'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        vars_range_and_types = [
            LogUniformSampling(1.0e-2, 1.0e1, only_positive=True),
            LogUniformSampling(1.0e-2, 1.0e1, only_positive=True),
            LogUniformSampling(1.0e-2, 1.0e-1, only_positive=True), LogUniformSampling(1.0e-2, 1.0e1, only_positive=True)
        ]

        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)
        x = self.x
        c = np.random.randn(2)
        self.sympy_eq = c[0] * x[1] * sympy.tanh(c[1] * x[2] / x[3])


@register_eq_class
class FeynmanIICh35Eq213(KnownEquation):
    """
    - Equation: II.35.21
    - Raw: n_rho * mom * tanh(mom * B / (1.380649e-23 * T))
    - Num. Vars: 4
    - Vars:
        - x[0]: n_rho (integer -> real due to its order, positive)
        - x[1]: mom (float, positive)
        - x[2]: B (float, positive)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0
    """
    _eq_name = 'feynman-ii.35.21.3'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        vars_range_and_types = [
            LogUniformSampling(1.0e-2, 1.0e1, only_positive=True),
            LogUniformSampling(1.0e-2, 1.0e1, only_positive=True),
            LogUniformSampling(1.0e-2, 1.0e1, only_positive=True), LogUniformSampling(1.0e-2, 1.0e1, only_positive=True)
        ]

        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)
        x = self.x
        c = np.random.randn(2)
        self.sympy_eq = c[0] * x[1] * sympy.tanh(x[1] * x[2] / c[1])


##
@register_eq_class
class FeynmanIIICh14Eq141(KnownEquation):
    """
    - Equation: III.14.14
    - Raw: I_0 * (exp(q * Volt / (1.380649e-23 * T)) - 1)
    - Num. Vars: 4
    - Vars:
        - x[0]: I_0 (float)
        - x[1]: q (float, positive)
        - x[2]: Volt (float)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0
    """
    _eq_name = 'feynman-iii.14.14.1'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        vars_range_and_types = [
            LogUniformSampling(1.0e-3, 1.0e1, only_positive=True), LogUniformSampling(1.0e-1, 1.0e1, only_positive=True),
            LogUniformSampling(1.0e-1, 1.0e1, only_positive=True), LogUniformSampling(1.0e-1, 1.0e1, only_positive=True)
        ]

        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)
        x = self.x
        c = np.random.randn(2)
        self.sympy_eq = x[0] * (sympy.exp(x[1] * c[0] / x[3]) - c[1])


@register_eq_class
class FeynmanIIICh14Eq142(KnownEquation):
    """
    - Equation: III.14.14
    - Raw: I_0 * (exp(q * Volt / (1.380649e-23 * T)) - 1)
    - Num. Vars: 4
    - Vars:
        - x[0]: I_0 (float)
        - x[1]: q (float, positive)
        - x[2]: Volt (float)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0
    """
    _eq_name = 'feynman-iii.14.14.2'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        vars_range_and_types = [
            LogUniformSampling(1.0e-3, 1.0e1, only_positive=True), LogUniformSampling(1.0e-1, 1.0e1, only_positive=True),
            LogUniformSampling(1.0e-1, 1.0e1, only_positive=True), LogUniformSampling(1.0e-1, 1.0e1, only_positive=True)
        ]

        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)
        x = self.x
        c = np.random.randn(2)
        self.sympy_eq = x[0] * (sympy.exp(x[1] * x[2] / c[0]) - c[1])


@register_eq_class
class FeynmanIIICh14Eq143(KnownEquation):
    """
    - Equation: III.14.14
    - Raw: I_0 * (exp(q * Volt / (1.380649e-23 * T)) - 1)
    - Num. Vars: 4
    - Vars:
        - x[0]: I_0 (float)
        - x[1]: q (float, positive)
        - x[2]: Volt (float)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0
    """
    _eq_name = 'feynman-iii.14.14.3'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        vars_range_and_types = [
            LogUniformSampling(1.0e-3, 1.0e1, only_positive=True), LogUniformSampling(1.0e-1, 1.0e1, only_positive=True),
            LogUniformSampling(1.0e-1, 1.0e1, only_positive=True), LogUniformSampling(1.0e-1, 1.0e1, only_positive=True)
        ]

        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)
        x = self.x
        c = np.random.randn(3)
        self.sympy_eq = c[0] * sympy.exp(c[1] * x[2] / (x[3])) - c[2]

@register_eq_class
class FeynmanIIICh14Eq144(KnownEquation):
    """
    - Equation: III.14.14
    - Raw: I_0 * (exp(q * Volt / (1.380649e-23 * T)) - 1)
    - Num. Vars: 4
    - Vars:
        - x[0]: I_0 (float)
        - x[1]: q (float, positive)
        - x[2]: Volt (float)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0
    """
    _eq_name = 'feynman-iii.14.14.4'
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'const']

    def __init__(self):
        vars_range_and_types = [
            LogUniformSampling(1.0e-3, 1.0e1, only_positive=True), LogUniformSampling(1.0e-1, 1.0e1, only_positive=True),
            LogUniformSampling(1.0e-1, 1.0e1, only_positive=True), LogUniformSampling(1.0e-1, 1.0e1, only_positive=True)
        ]

        super().__init__(num_vars=4, vars_range_and_types=vars_range_and_types)
        x = self.x
        c = np.random.randn(3)
        self.sympy_eq = c[0] * sympy.exp(x[1] * x[2] / c[1]) - c[2]