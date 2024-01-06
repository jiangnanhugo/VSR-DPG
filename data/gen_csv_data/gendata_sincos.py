import os

from scibench.symbolic_equation_evaluator_public import decrypt_equation, Equation_evaluator
import numpy as np
from scibench.symbolic_data_generator import *


def to_csv(X, y, filename):
    d = np.concatenate((X, y), axis=1)
    np.random.shuffle(d)
    np.savetxt(filename + ".csv", d, delimiter=",")


if __name__ == '__main__':
    basepath = "/home/jiangnan/PycharmProjects/scibench/data/unencrypted/equations_trigometric/sincos_nv3_nt22_prog_{}.in"
    to_folder = "/home/jiangnan/PycharmProjects/scibench/eureqa/data/equations_trigometric/sincos_nv3_nt22_prog_{}"
    for prog in range(10):
        filename = basepath.format(prog)
        data_query_oracle = Equation_evaluator(filename, noise_type='normal', noise_scale=0.0, metric_name='neg_mse')
        dataX = DataX(data_query_oracle.get_vars_range_and_types())
        batchsize = 1000


        X = dataX.randn(sample_size=batchsize).T
        y = data_query_oracle.evaluate(X).reshape(-1,1)
        print(X.shape, y.shape)
        filename_csv = to_folder.format(prog)
        # to_csv(X, y, filename_csv)
        print(f"{prog} done......")
        # print(one_eq['eq_expression'].execute(X))
