from distutils.core import setup

import numpy
from Cython.Build import cythonize

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



setup(name='cvdso',
      version='1.0',
      description='Deep symbolic optimization with control variable experiment.',
      author='NA',
      packages=['cvdso'],
      setup_requires=["numpy"],
      include_dirs=[numpy.get_include()],
      install_requires=required,
      )
