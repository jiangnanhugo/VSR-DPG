# from equation_tree import EquationTree
from equation_tree import sample
#
# equations = sample()
# print(equations)

# equations = sample(n=100, max_num_variables=200, depth=8)
# for x in equations:
#     print(x)

p = {
    'structures': {'[0, 1, 2, 3, 3, 2, 3, 4, 4, 1, 2, 3, 3]': .5, '[0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 3, 3]': .5},
    'features': {'constants': .5, 'variables': .5},
    'functions': {'sin': .5, 'cos': .5},
    'operators': {'+': .8, '-': .2}
}
equations_with_prior = sample(n=100, prior=p, max_num_variables=100)
for x in equations_with_prior:
    print(x)