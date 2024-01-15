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
    "tensorflow==1.15.5",
    "numba",
    "sympy",
    "pandas",
    "scikit-learn",
    "click",
    "pathos",
    "seaborn",
    "progress",
    "tqdm",
    "commentjson",
    "PyYAML",
    "protobuf==3.19"
]


setup(name='dso',
      version='1.1',
      description='Deep symbolic optimization.',
      author='LLNL',
      packages=['dso'],
      setup_requires=["numpy", "Cython"],
      ext_modules=cythonize([os.path.join('dso', 'cyfunc.pyx')]),
      include_dirs=[numpy.get_include()],
      install_requires=required
      )
