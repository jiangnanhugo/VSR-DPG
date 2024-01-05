from distutils.core import setup

import numpy

required = [
    "pytest",
    "cython",
    "numpy",
    "tensorflow",
    "numba",
    "sympy",
    "pandas",
    "scikit-learn",
    "click",
    "deap",
    "pathos",
    "seaborn",
    "progress",
    "tqdm",
    "commentjson",
    "PyYAML"
]



setup(name='grammar',
      version='1.0',
      description='Deep symbolic optimization with control variable experiment.',
      author='NA',
      packages=['grammar'],
      install_requires=required,
      )
