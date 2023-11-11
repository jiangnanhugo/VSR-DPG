from setuptools import setup
import os
from setuptools import dist

dist.Distribution().fetch_build_eggs(['Cython', 'numpy'])

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
    "click",
    "tqdm",
    "commentjson",
    "PyYAML",
]


setup(name='cvDSO',
      version='1.0',
      description='Deep symbolic optimization.',
      author='LLNL',
      packages=['cvdso'],
      setup_requires=["numpy", "Cython"],
      ext_modules=cythonize([os.path.join('cvdso', 'cyfunc.pyx')]),
      include_dirs=[numpy.get_include()],
      install_requires=required
      )
