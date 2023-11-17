from distutils.core import setup
import os
from setuptools import dist

dist.Distribution().fetch_build_eggs(['Cython', 'numpy'])

import numpy
from Cython.Build import cythonize

required = [
    "pytest",
    "cython",
    "numpy",
    "tensorflow==1.14",
    "numba==0.53.1",
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
      description='Deep symbolic optimization with control variable expeirment.',
      author='NA',
      packages=['cvdso'],
      setup_requires=["numpy", "Cython"],
      ext_modules=cythonize([os.path.join('cvdso', 'cyfunc.pyx')]),
      include_dirs=[numpy.get_include()],
      install_requires=required,
      extras_require=extras
      )
