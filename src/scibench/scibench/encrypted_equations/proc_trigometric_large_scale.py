import copy
import pickle
import os

import pylab as p
from sympy import *
import click
from sympy.parsing.sympy_parser import parse_expr
import numpy as np


def read_true_program(filename):
    prog = pickle.load(open(filename, 'rb'))
    preorder_traversal_expr = prog['preorder']
    preorder_traversal_tuple = []
    for loc, val in zip(prog['const_loc'], prog['consts']):
        preorder_traversal_expr[loc] = val
    for idx, it in enumerate(preorder_traversal_expr):
        if idx in prog['const_loc']:
            preorder_traversal_tuple.append((it, 'const'))
        elif it.startswith('x') or it.startswith('X'):
            preorder_traversal_tuple.append((it, 'var'))
        elif it in ['add', 'mul', 'sub', 'div']:
            preorder_traversal_tuple.append((it, 'binary'))
        elif it in ['inv', 'sqrt', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'n4']:
            preorder_traversal_tuple.append((it, 'unary'))
    return preorder_traversal_expr, preorder_traversal_tuple


#
def sympy_expr(traversal):
    """
    Returns the attribute self.sympy_expr.

    This is actually a bit complicated because we have to go:
    traversal --> tree --> serialized tree --> SymPy expression
    """
    tree = build_tree(traversal)
    tree = convert_to_sympy(tree)
    tree_str = tree.__repr__()
    expr = parse_expr(tree_str)
    return expr


# Possible library elements that sympy capitalizes
capital = ["add", "mul", "pow"]


class Node(object):
    """Basic tree class supporting printing"""

    def __init__(self, val):

        self.val = val
        self.children = []

    def __repr__(self):
        if len(self.children) == 0:
            return "{}".format(self.val)

        if len(self.children) == 0:
            return self.val
        if self.val == "add":
            return "{} + {}".format(repr(self.children[0]), repr(self.children[1]))
        elif self.val == "mul":
            return "{} * {}".format(repr(self.children[0]), repr(self.children[1]))
        elif self.val == "inv":
            children_repr = ",".join(repr(child) for child in self.children)
            return "1 / {}".format(children_repr)
        else:
            children_repr = ",".join(repr(child) for child in self.children)
            return "{}({})".format(self.val, children_repr)


op_arity_dict = {
    'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'inv': 1, 'sqrt': 1,
    'sin': 1, 'cos': 1, 'exp': 1, 'log': 1, 'n2': 1, 'n3': 1, 'n4': 1}


def build_tree(traversal):
    """Recursively builds tree from pre-order traversal"""

    op = traversal.pop(0)
    if op in op_arity_dict:
        n_children = op_arity_dict[op]
    else:
        node = Node(op)
        return node
    val = op
    if val in capital:
        val = val.capitalize()

    node = Node(val)

    for _ in range(n_children):
        node.children.append(build_tree(traversal))

    return node


# this function is used for pretty print the expression
def convert_to_sympy(node):
    """Adjusts trees to only use node values supported by sympy"""

    if node.val == "div":
        node.val = "Mul"
        new_right = Node("Pow")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "sub":
        node.val = "Add"
        new_right = Node("Mul")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "inv":
        node.val = Node("Pow")
        node.children.append(Node("-1"))

    elif node.val == "neg":
        node.val = Node("Mul")
        node.children.append(Node("-1"))

    elif node.val == "n2":
        node.val = "Pow"
        node.children.append(Node("2"))

    elif node.val == "n3":
        node.val = "Pow"
        node.children.append(Node("3"))

    elif node.val == "n4":
        node.val = "Pow"
        node.children.append(Node("4"))

    for child in node.children:
        convert_to_sympy(child)

    return node


template = """

@register_eq_class
class {}(KnownEquation):
    _eq_name = '{}'
    _function_set = {}
    
    def __init__(self):
        super().__init__(num_vars={})
        x = self.x
        self.sympy_eq = {}
        self.sympy_eq_preorder_traversal = {}
"""


def write_to_files(equations, template, output_folder, used_vars,total_variables):
    fw = open(os.path.join(output_folder, f"equations_trigonometric_nv{used_vars}_large_scale_{total_variables}.py"), 'w')
    fw.write("""from collections import OrderedDict
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

""")
    for spl in equations:
        function_set = None
        for key in function_set_dict:
            if key in spl[0]:
                function_set = function_set_dict[key]
        spl2 = spl[2]
        for i in range(total_variables, -1, -1):
            spl2 = spl2.replace(f'X{i}', f'x[{i}]')
        spl2 = spl2.replace('log', 'sympy.log').replace(
            'exp', 'sympy.exp').replace('sin', 'sympy.sin').replace('cos', 'sympy.cos').replace('div', 'sympy.div').replace(
            'sqrt', 'sympy.sqrt').replace('pow', 'sympy.pow')
        fw.write(template.format(
            spl[0].replace("-", "_"),
            spl[0],
            function_set,
            spl[1],
            spl2,
            spl[3])
        )


function_set_dict = {
    'inv_': ["add", "sub", "mul", "inv", "const"],
    'sincos_': ["add", "sub", "mul", "sin", "cos", "const"],
    'sincosinv_': ["add", "sub", "mul", "inv", "sin", "cos", "const"],
}


def replace_with_rand_variables(rand_mapping, preorder_traversal_expr, preorder_traversal_tuple):
    new_traversal_expr = []
    new_traversal_tuple = []
    for ti in preorder_traversal_expr:
        if ti in rand_mapping:
            new_traversal_expr.append(rand_mapping[ti])
        else:
            new_traversal_expr.append(ti)
    for ti in preorder_traversal_tuple:
        if ti[0] in rand_mapping:
            new_traversal_tuple.append((rand_mapping[ti[0]], ti[1]))
        else:
            new_traversal_tuple.append(ti)
    return new_traversal_expr, new_traversal_tuple


@click.command()
@click.option("--total_variables", default=50, help="Number of total variables.")
@click.option('--basepath', default='/home/jiangnan/PycharmProjects/xyx_dso/data/')
@click.option('--output_folder', default='./')
def main(total_variables, basepath, output_folder):
    program_files = []
    for root, dirs, files in os.walk(basepath, topdown=False):
        if 'nv5' in root:
            for name in files:
                if name.endswith(".data"):
                    program_files.append(os.path.join(root, name))
    equations = []

    for filename in program_files:
        print(filename)
        preorder_traversal_expr, preorder_traversal_tuple = read_true_program(filename)
        expr = sympy_expr(copy.deepcopy(preorder_traversal_expr))
        num_vars = len(expr.free_symbols)
        used = set()

        for t in range(10):
            select_variables = np.random.choice([f"X{i}" for i in range(total_variables)], num_vars, replace=False)
            while "_".join(select_variables) in used:
                select_variables = np.random.choice([f"X{i}" for i in range(total_variables)], num_vars, replace=False)
            used.add("_".join(select_variables))
            rand_mapping = {}
            for i in range(num_vars):
                rand_mapping[f"X_{i}"] = select_variables[i]
            preorder_traversal_expr_local, preorder_traversal_tuple_local = replace_with_rand_variables(
                rand_mapping,
                preorder_traversal_expr,
                preorder_traversal_tuple)
            # print(preorder_traversal_tuple)
            new_expr = sympy_expr(preorder_traversal_expr_local)
            eq_class_name = "_".join(filename.split("/")[-2:])[:-5] + f"_totalvars_{total_variables}_rand_{t}"

            equations.append([eq_class_name, total_variables, str(new_expr), preorder_traversal_tuple_local])

    write_to_files(equations, template, output_folder,num_vars, total_variables)


if __name__ == '__main__':
    main()
