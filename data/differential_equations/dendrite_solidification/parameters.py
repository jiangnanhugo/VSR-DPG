import numpy as np


Nx = 200
Ny = 200
NxNy = Nx*Ny

dx = 0.03
dy = 0.03

#- Time integration parameters:

nstep = 2000
nprint = 50
dtime = 1.e-4

#--- Material specific parameters:

tau = 0.0003
epsilonb = 0.01
mu = 1.0
kappa = 1.8
delta = 0.02
aniso = 6.0
alpha = 0.9
gamma = 10.0
teq = 1.0
theta0 = 0.2



seed = 2.0

pix=4.0*np.arctan(1.0)
