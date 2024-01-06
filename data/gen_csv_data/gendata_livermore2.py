from scibench.symbolic_equation_evaluator_public import Equation_evaluator

from scibench.symbolic_data_generator import *


def to_csv(X, y, filename):
    d = np.concatenate((X, y), axis=1)
    np.random.shuffle(d)
    np.savetxt(filename + ".csv", d, delimiter=",")


if __name__ == '__main__':
    basepath = "/home/jiangnan/PycharmProjects/scibench/data/unencrypted/equations_livermore2/Livermore2_Vars{}_{}.in"
    to_folder = "/home/jiangnan/PycharmProjects/scibench/eureqa/data/equations_livermore2/Livermore2_Vars{}_{}"
    for vari in [2, 3,4,5,6, 7]:
        for prog in range(1, 26):
            filename = basepath.format(vari, prog)
            data_query_oracle = Equation_evaluator(filename, noise_type='normal', noise_scale=0.0, metric_name='neg_mse')
            dataX = DataX(data_query_oracle.get_vars_range_and_types())
            batchsize = 100000

            X = dataX.randn(sample_size=batchsize).T
            y = data_query_oracle.evaluate(X).reshape(-1, 1)
            print(X.shape, y.shape)
            filename_csv = to_folder.format(vari, prog)
            to_csv(X, y, filename_csv)
            print(filename_csv)
            print(f"{vari}-{prog} done......")
