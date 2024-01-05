import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diff_ops import LaplacianOp, GradientOp

class DendriteSingleTimestep(nn.Module):
    def __init__(self, dx, dy, dt):
        super(DendriteSingleTimestep, self).__init__()
        self.tau = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.epsilonb = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.kappa = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.delta = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.aniso = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.teq = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.theta0 = nn.Parameter(torch.tensor([0.0]), requires_grad=False)

        self.pix = 4.0*np.arctan(1.0)
        self.lap = LaplacianOp()
        self.grad = GradientOp()
        self.dx = dx
        self.dy = dy
        self.dtime = dt

    def init_params(self,params):
        self.tau.data = torch.tensor([params['tau']]) # 0.0003
        self.epsilonb.data = torch.tensor([params['epsilonb']]) # 0.01
        self.kappa.data = torch.tensor([params['kappa']]) # 1.8
        self.delta.data = torch.tensor([params['delta']]) # 0.02
        self.aniso.data = torch.tensor([params['aniso']]) # param.aniso
        self.alpha.data = torch.tensor([params['alpha']]) # 0.9
        self.gamma.data = torch.tensor([params['gamma']]) # 10.0
        self.teq.data = torch.tensor([params['teq']]) # 1.0
        self.theta0.data = torch.tensor([params['theta0']]) # 0.2
    
    def forward(self,phi,tempr):
        lap_phi = self.lap(phi,self.dx,self.dy)
        lap_tempr = self.lap(tempr,self.dx,self.dy)
        
        phidx = self.grad(phi,'x',self.dx)
        phidy = self.grad(phi,'y',self.dy)
        
        theta = torch.atan2(phidy,phidx)


        epsilon = self.epsilonb*(1.0+self.delta*torch.cos(self.aniso*(theta-self.theta0)))
        epsilon_deriv = -self.epsilonb*self.aniso*self.delta*torch.sin(self.aniso*(theta-self.theta0))

        prod_mat1 = epsilon*epsilon_deriv*phidx
        prod_mat2 =-epsilon*epsilon_deriv*phidy
        
        term1 = self.grad(prod_mat1,'y',self.dy)
        term2 = self.grad(prod_mat2,'x',self.dx)

        m = (self.alpha/self.pix) * torch.atan(self.gamma*(self.teq-tempr))

        phiold = torch.clone(phi)
        dphi_dt = (self.dtime/self.tau)*(term1+term2+(epsilon**2)*lap_phi +\
                    phiold*(1.0-phiold)*(phiold-0.5+m))
        phi_new = phi + dphi_dt
        tempr_new = tempr + self.dtime*lap_tempr + self.kappa*(dphi_dt)
        

        # assert ((phi_new-phi)==dphi_dt).all(),np.linalg.norm(phi_new-phi-dphi_dt)
        # print("dphi_dt",dphi_dt[0,0]/self.dtime)
        # print("phi=%r, Temp=%r, m=%r"%(phi_new[0,0].item(),tempr_new[0,0].item(),m[0,0].item()))
        return phi_new,tempr_new