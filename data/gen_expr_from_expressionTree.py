# from equation_tree import EquationTree
from equation_tree import sample
#
# equations = sample()
# print(equations)

# equations = sample(n=100, max_num_variables=20, depth=5)
# for x in equations:
#     print(x)

p = {
    'structures': {'[0, 1, 1]': .3, '[0, 1, 2]': .3, '[0, 1, 2, 3, 2, 3, 1]': .4},
    'features': {'constants': .5, 'variables': .5},
    'functions': {'sin': .5, 'cos': .5},
    'operators': {'+': .8, '-': .2}
}
equations_with_prior = sample(n=100, prior=p, max_num_variables=10)
for x in equations_with_prior:
    print(x)