import numpy as np
import datetime
import torch

def nucleaus(Nx,Ny,seed):
    phi = torch.zeros((Nx,Ny),dtype=torch.double)
    tempr = torch.zeros((Nx,Ny),dtype=torch.double)
    for i in range(Nx):
        for j in range(Ny):
            if (i-Nx/2)**2+(j-Ny/2)**2 < seed:
                phi[i,j] = 1.0
    
    print("initializing ...")
    # phi = torch.rand(Nx,Ny)
    # tempr = torch.rand(Nx,Ny)
    return phi,tempr

def random_phi(Nx,Ny):
    phi = torch.rand(Nx,Ny,dtype=torch.double)
    tempr = torch.zeros((Nx,Ny),dtype=torch.double)
    return phi,tempr

if __name__=='__main__':
    nucleaus(100,100,100)