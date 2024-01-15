

import numpy as np
from scipy.integrate import solve_ivp
from pysindy.utils import pendulum_on_cart

import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

t_train = np.linspace(0.0001, 10, 25600)
x0 = np.random.rand(4)
x_train = solve_ivp(pendulum_on_cart, (t_train[0], t_train[-1]), x0, t_eval=t_train).y.T

print(t_train)
print(x_train)
optimizer = ps.STLSQ(threshold=1e-6)
import numpy as np
from pysindy.feature_library import FourierLibrary, CustomLibrary
from pysindy.feature_library import GeneralizedLibrary
# x = np.array([[0., -1], [1., 0.], [2., -1.]])
# functions = [lambda x,y:x+y, lambda x,y: x-y, lambda x,y: x*y, lambda x,y: x/y, lambda x: np.sin(x), lambda x: np.cos(x), lambda x: x**2 ]
# lib_custom = CustomLibrary(library_functions=functions)
lib_fourier = FourierLibrary()
library = ps.PolynomialLibrary(degree=2)
lib_generalized = GeneralizedLibrary([library, lib_fourier])
# lib_generalized.fit(x)
# lib_generalized.transform(x)
# ps.GeneralizedLibrary()

model = ps.SINDy(
    optimizer=optimizer,
    feature_library=lib_generalized
)
model.fit(x_train, t_train)
model.print()