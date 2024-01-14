

import numpy as np
from scipy.integrate import solve_ivp
from pysindy.utils import yeast

import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

t_train = np.linspace(0.0001, 10, 25600)
x0 = np.random.rand(7)
x_train = solve_ivp(yeast, (t_train[0], t_train[-1]), x0, t_eval=t_train).y.T

print(t_train)
print(x_train)
optimizer = ps.STLSQ(threshold=1e-6)
library = ps.PolynomialLibrary(degree=4)
model = ps.SINDy(
    optimizer=optimizer,
    feature_library=library
)
model.fit(x_train, t_train)
model.print()