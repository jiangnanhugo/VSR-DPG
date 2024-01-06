import os

import numpy as np
from scibench.encrypted_equations.diff_ops import LaplacianOp, DifferentialOp

from scibench.symbolic_equation_evaluator_public import decrypt_equation, Equation_evaluator
import torch

import torchvision
import torchvision.io
from scibench.symbolic_data_generator import *


class SpinodalDecomp64x64(object):
    _eq_name = 'Spinodal_Decomposition_64x64'
    _function_set = ['add', 'sub', 'mul', 'div', 'clamp', 'laplacian', 'const']
    expr_obj_thres = 0.01
    expr_consts_thres = None
    simulated_exec = True

    def __init__(self):
        self.A = torch.tensor(1)  # np.random.randn(1)[0]  # .to(device)
        self.M = torch.tensor(1)  # np.random.randn(1)[0]  # .to(device)
        self.kappa = torch.tensor(0.5)  # np.random.randn(1)[0]  # .to(device)
        self.dt = 0.01

        self.lap = LaplacianOp()
        self.diff = DifferentialOp()

        self.dx = 1
        self.dy = 1
        self.Nx = 64
        self.Ny = 64
        self.dim = [(self.Nx, self.Ny), ]

        # vars_range_and_types = [LogUniformSampling2d(1e-3, 1.0, only_positive=True, dim=(self.Nx, self.Ny))]
        # self.x = [MatrixSymbol('X_0', self.Nx, self.Ny)]
        # super().__init__(num_vars=1, vars_range_and_types=vars_range_and_types)
        c = ['X_0']

        self.torch_func = self.forward
        self.sympy_eq = "EMPTY"
        ### consts = [0:1.0, 1:2.0, 2:A(1), 3:kappa(0.5), 4:dt(1e-2), 5:M(1)]
        consts = [1.0, 2.0, self.A, self.kappa, self.dt, self.M]

        ### tree1 = 2 * self.A * c * (1 - c) * (1 - 2*c)
        tree1 = [(2, "mul"), (0, 1), (2, "mul"), (0, 2), (2, "mul"), (1, 0), (2, "mul"), (2, "sub"), (0, 0), (1, 0), (2, "sub"),
                 (0, 0), (2, "mul"), (0, 1), (1, 0)]
        ### tree2 = self.kappa * self.lap(c, self.dx, self.dy)
        tree2 = [(2, "mul"), (0, 3), (2, "laplacian"), (1, 0)]
        # deltaF = 2 * self.A * c * (1-c) * (1-2*c) - self.kappa * self.lap(c, self.dx, self.dy)
        deltaF = [(2, "sub")]
        deltaF.extend(tree1)
        deltaF.extend(tree2)
        # dc = self.dt * self.lap(self.M*deltaF, self.dx, self.dy)
        dc = [(2, "mul"), (0, 4), (2, "laplacian"), (2, "mul"), (0, 5)]
        dc.extend(deltaF)
        # c_new = torch.clamp(c + dc, min=0.0001, max=0.9999)
        preorder_traversal = [(2, "clamp"), (2, "add"), (1, 0)]
        preorder_traversal.extend(dc)
        self.preorder_traversal = []
        for x in preorder_traversal:
            if x[0] == 0:
                self.preorder_traversal.append((consts[x[1]], "const"))
            elif x[0] == 1:
                self.preorder_traversal.append((str(c[x[1]]), "var"))
            elif x[0] == 2:
                if x[1] in ['mul', 'sub', 'div', 'add']:
                    self.preorder_traversal.append((x[1], "binary"))
                elif x[1] == "laplacian":
                    self.preorder_traversal.append(("laplacian", "unary"))
                elif x[1] == "clamp":
                    self.preorder_traversal.append((x[1], "unary"))

    def forward(self, c):
        # equation (4:18) + (4:17)
        deltaF = 2 * self.A * c * (1 - c) * (1 - 2 * c) - self.kappa * self.lap(c, self.dx, self.dy)
        # equation (4.16)
        dc = self.dt * self.lap(self.M * deltaF, self.dx, self.dy)
        # c_new = c+dc
        c_new = torch.clamp(c + dc, min=0.0001, max=0.9999)
        return c_new


def to_csv(X, y, filename):
    d = np.concatenate((X, y), axis=1)
    np.random.shuffle(d)
    np.savetxt(filename + ".csv", d, delimiter=",")


def to_npz(all_phi, filename):
    print("saving to:", filename)
    np.save(filename, all_phi)


def to_mp4(all_phi, filename):
    print("saving to", filename)
    # all_phi=np.
    # all_phi = np.stack(all_phi, axis=0)
    # print('the shape is', all_phi.shape)
    # all_phi = torch.from_numpy(all_phi)
    all_phi = [torch.from_numpy(phi) for phi in all_phi]
    all_phi_T = torch.stack(all_phi, dim=0) * 255.0
    all_phi_T = torch.unsqueeze(all_phi_T, dim=-1)
    print("all_phi_T.size=", all_phi_T.size())

    all_phi_T_C = all_phi_T.repeat([1, 1, 1, 3])
    print("all_phi_T_C.size=", all_phi_T_C.size())

    all_phi_T_C = all_phi_T_C.type(torch.ByteTensor)

    torchvision.io.write_video(filename + '.mp4', all_phi_T_C, fps=24)


if __name__ == '__main__':
    basepath = "/home/jiangnan/PycharmProjects/scibench/data/unencrypted/equations_pde/{}.in"
    to_folder = "/home/jiangnan/PycharmProjects/scibench/data/unencrypted/equations_pde/{}"
    for prog in ['SpinodalDecomp64x64', ]:
        filename = basepath.format(prog)
        data_query_oracle = Equation_evaluator(filename, noise_type='normal', noise_scale=0.0, metric_name='neg_mse')
        dataX = DataX(data_query_oracle.get_vars_range_and_types())
        batchsize = 1
        simulated_steps = 2400
        Nx, Ny = data_query_oracle.dim[0]
        c0 = dataX.randn(batchsize).squeeze()
        print(c0.shape)
        all_y = data_query_oracle.execute_simulate(c0, simulated_steps, return_last_step=False)
        # print(all_y[0].shape, len(all_y))
        # print(all_y[-1])
        output_filename = "_".join([prog, "bs" + str(batchsize), 'steps' + str(simulated_steps)])
        to_npz(all_y, to_folder.format(output_filename))
        to_mp4(all_y, to_folder.format(output_filename))

        # spin = SpinodalDecomp64x64()
        # c0 = torch.from_numpy(c0)
        # c0 = c0.type(torch.float)
        # all_phis = [c0.numpy()]
        # c = c0
        # for t in range(simulated_steps):
        #     cnew = spin.forward(c)
        #     all_phis.append(cnew.numpy())
        #     c = cnew
        # print(len(all_y), len(all_phis))
        # print(all_phis[-1].numpy()[0])
        # to_mp4(all_phis, to_folder.format(output_filename))
        # for x, y in zip(all_y, all_phis):
        #     if type(x) != np.ndarray:
        #         print(x.numpy().shape, y.shape, np.sum(x.numpy() - y))
        #     else:
        #         print(x.shape, y.shape, np.sum(x - y))
        #     # print(y)
        #     # print('-' * 40)
