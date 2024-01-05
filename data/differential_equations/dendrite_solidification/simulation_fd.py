from datetime import datetime
import parameters
from init_dendrite import nucleaus
import imageio
from scipy import ndimage, misc
import torch

from dendrite_model import DendriteSingleTimestep

output_dir = "output/"


if __name__=='__main__':
    time0 = datetime.now()

    Nx = parameters.Nx
    Ny = parameters.Ny
    seed = parameters.seed
    nstep = parameters.nstep

    phi, tempr = nucleaus(Nx,Ny,seed)
    den_model = DendriteSingleTimestep()

    all_phi = None
    all_tempr = None

    ttime = 0
    dtime = parameters.dtime


    with torch.no_grad():
        for istep in  range(nstep):
            ttime = ttime +dtime
            phi, tempr = den_model(phi,tempr)
            phi1 = phi.unsqueeze(0)
            tempr1 = tempr.unsqueeze(0)
            if istep%parameters.nprint==0:
                print("Step",istep)
                if all_phi is None:
                    all_phi = phi1
                    all_tempr = tempr1
                else:
                    all_phi = torch.cat((all_phi,phi1),0)
                    all_tempr = torch.cat((all_tempr,tempr1),0)
    
    print("all_phi=",all_phi.size())
    print("all_tempr=",all_tempr.size())

    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    print('Compute Time: %10f\n'%compute_time)
    torch.save(all_phi,output_dir+"phi.pt")
    torch.save(all_tempr,output_dir+"tempr.pt")


