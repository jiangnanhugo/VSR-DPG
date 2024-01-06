#### Requirement:
import json
import os

import numpy as np
import xxhash
from typing import List, Dict, Set

from cryptography.fernet import Fernet

import sympy
from sympy import *


def generate_new_key(saveto_filename):
    key = Fernet.generate_key()

    # string the key in a file
    with open(saveto_filename, 'wb') as filekey:
        filekey.write(key)


def encrypt_equation(equation, output_eq_file, key_filename=None, is_encrypted=0):
    """
    {eq_name: "", n_vars: , eq_expression: {}}
    """
    if is_encrypted == 1:
        # opening the key
        with open(key_filename, 'rb') as filekey:
            key = filekey.read()

        # using the generated key
        fernet = Fernet(key)
        # encrypting the Sympy Equation
        encrypted = fernet.encrypt(equation)
        with open(output_eq_file, 'wb') as encrypted_file:
            encrypted_file.write(b'1\n')
            encrypted_file.write(encrypted)
    else:
        with open(output_eq_file, 'wb') as encrypted_file:
            encrypted_file.write(b'0\n')
            encrypted_file.write(equation)


def decrypt_equation(eq_file, key_filename=None):
    with open(eq_file, 'rb') as enc_file:
        encrypted = enc_file.readline()
        if encrypted == b'1\n':
            encrypted = enc_file.readline()
            fernet = Fernet(open(key_filename, 'rb').read())
            decrypted = fernet.decrypt(encrypted)
        elif encrypted == b'0\n':
            decrypted = enc_file.readline()
    one_equation = json.loads(decrypted)
    preorder_traversal = eval(one_equation['eq_expression'])
    print("-" * 20)
    for key in one_equation:
        print(key, "\t", one_equation[key])
    print("preorder:", preorder_traversal)
    print("-" * 20)
    return one_equation


def to_binary_expr_tree(expr):
    if isinstance(expr, Symbol):
        return str(expr)
    elif isinstance(expr, Float) or isinstance(expr, Integer) or isinstance(expr, Rational):
        return [expr]
    elif expr == sympy.pi:
        return np.pi
    elif expr == sympy.EulerGamma:
        return np.euler_gamma
    else:
        op = expr.func
        args = expr.args

        if len(args) <= 2:
            return [op.__name__] + [to_binary_expr_tree(arg) for arg in args]
        else:
            left = to_binary_expr_tree(args[0])
            right = to_binary_expr_tree(op(*args[1:]))
            return [op.__name__, left, right]


def is_float(s):
    """Determine whether the input variable can be cast to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def symbolic_equation_to_preorder_traversal(expr) -> List:
    def flatten(S):
        if S == []:
            return S
        if isinstance(S[0], list):
            return flatten(S[0]) + flatten(S[1:])
        return S[:1] + flatten(S[1:])

    preorder_traversal_expr = flatten(to_binary_expr_tree(expr))
    preorder_traversal_tuple = []
    for idx, it in enumerate(preorder_traversal_expr):
        if is_float(it):
            preorder_traversal_tuple.append((it, 'const'))
        elif it.startswith('x') or it.startswith('X'):
            preorder_traversal_tuple.append((it, 'var'))
        elif it in ['add', 'Add', 'mul', 'Mul', 'sub', 'Sub', 'div', 'Div', 'pow', 'Pow']:
            preorder_traversal_tuple.append((it.lower(), 'binary'))
        elif it in ['inv', 'Inv', 'sqrt', 'Sqrt', 'sin', 'Sin', 'cos', 'Cos', 'exp', 'Exp', 'log', 'Log', 'n2', 'n3', 'n4']:
            preorder_traversal_tuple.append((it.lower(), 'unary'))
    return preorder_traversal_tuple


def main(private_key_folder='./', key_filename="public.key", output_folder="./", folder_prefix='equation_family'):
    if not os.path.isfile(os.path.join(private_key_folder, key_filename)):
        print('A new key is generated!')
        generate_new_key(key_filename)
    name_map = {}
    for eqname in EQUATION_CLASS_DICT:

        one_equation = get_eq_obj(eqname)
        if one_equation.num_vars not in name_map:
            name_map[one_equation.num_vars]=[]
        name_map[one_equation.num_vars].append(eqname)

    for key in name_map:
        print(f'var{key}=', end='"')
        for na in name_map[key]:
            print(f"{na}",end=" ")
        print('"')



if __name__ == '__main__':
    X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7, X_8, X_9, X_10, X_11, X_12 =symbols('X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9,X_10,X_11,X_12')
    from equations_feynman import *

    main(output_folder='/home/jiangnan/PycharmProjects/scibench/data/', folder_prefix='equations_feynman')
