from datetime import datetime
import imageio
from scipy import ndimage, misc
import torch
import os
import math

import parameters as param
from init_dendrite import nucleaus

from dendrite_model import DendriteSingleTimestep

output_dir = "output/"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if __name__=='__main__':
    time0 = datetime.now()

    Nx = param.Nx
    Ny = param.Ny
    seed = param.seed
    nstep = param.nstep
    
    param.dtime = 1e-4
    tau = 3*param.dtime

    phi, tempr = nucleaus(Nx,Ny,seed)
    den_model = DendriteSingleTimestep(dx=param.dx,dy=param.dy,dt=param.dtime)
    mparams = {'tau':tau,'epsilonb':0.01,'kappa':1.8,'delta':0.02,'aniso':4,'alpha':0.9,'gamma':10.0,'teq':1.0,'theta0':0.2}
    den_model.init_params(mparams)


    all_phi = None
    all_tempr = None
    all_data = []

    ttime = 0
    dtime = param.dtime

    with torch.no_grad():
        for istep in  range(nstep):
            ttime = ttime +dtime
            phi, tempr = den_model(phi,tempr)
            
            data_item = {'step':istep,'phi':phi.clone(),'Tmpr':tempr.clone()}
            print("step=%d, phi=%r, T=%r"%(istep,phi[0,0].item(),tempr[0,0].item()))
            if math.isnan(phi[0,0].item()):
                break
            all_data.append(data_item)
    print("Total time",ttime)
    torch.save(all_data,output_dir+'dendrite_data.pkl')


