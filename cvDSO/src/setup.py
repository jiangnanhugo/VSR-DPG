from distutils.core import setup


required = [
    "cython",
    "numpy",
    "tensorflow==2.15.0",
    "numba",
    "sympy",
    "click",
    "tqdm",
    "commentjson",
    "PyYAML"
]



setup(name='grammar',
      version='1.0',
      description='Deep symbolic optimization with control variable experiment.',
      packages=['grammar'],
      install_requires=required,
      )
