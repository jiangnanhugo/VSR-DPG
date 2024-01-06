import os
import numpy as np
from scibench.symbolic_equation_evaluator_public import Equation_evaluator

from scibench.symbolic_data_generator import DataX


def to_csv(X, y, filename):
    d = np.concatenate((X, y), axis=1)
    np.random.shuffle(d)
    np.savetxt(filename + ".csv", d, delimiter=",")



var2 = ['FeynmanICh12Eq1', 'FeynmanICh6Eq20', 'FeynmanICh10Eq7', 'FeynmanICh12Eq4', 'FeynmanICh14Eq3',
        'FeynmanICh12Eq5', 'FeynmanICh14Eq4', 'FeynmanICh15Eq10', 'FeynmanICh16Eq6', 'FeynmanICh25Eq13',
        'FeynmanICh26Eq2', 'FeynmanICh32Eq5', 'FeynmanICh34Eq10', 'FeynmanICh34Eq14', 'FeynmanICh38Eq12',
        'FeynmanICh39Eq10', 'FeynmanICh41Eq16', 'FeynmanICh43Eq31', 'FeynmanICh48Eq2', 'FeynmanIICh3Eq24',
        'FeynmanIICh4Eq23', 'FeynmanIICh8Eq7', 'FeynmanIICh10Eq9', 'FeynmanIICh11Eq28', 'FeynmanIICh13Eq17',
        'FeynmanIICh13Eq23', 'FeynmanIICh13Eq34', 'FeynmanIICh24Eq17', 'FeynmanIICh34Eq29a', 'FeynmanIICh38Eq14',
        'FeynmanIIICh4Eq32', 'FeynmanIIICh4Eq33', 'FeynmanIIICh7Eq38', 'FeynmanIIICh8Eq54', 'FeynmanIIICh15Eq14',
        'FeynmanBonus8', 'FeynmanBonus10']
var3 = ['FeynmanICh6Eq20b', 'FeynmanICh12Eq2', 'FeynmanICh15Eq3t', 'FeynmanICh15Eq3x', 'FeynmanBonus20',
        'FeynmanICh18Eq12', 'FeynmanICh27Eq6', 'FeynmanICh30Eq3', 'FeynmanICh30Eq5', 'FeynmanICh37Eq4',
        'FeynmanICh39Eq11', 'FeynmanICh39Eq22', 'FeynmanICh43Eq43', 'FeynmanICh47Eq23', 'FeynmanIICh6Eq11',
        'FeynmanIICh6Eq15b', 'FeynmanIICh11Eq27', 'FeynmanIICh15Eq4', 'FeynmanIICh15Eq5', 'FeynmanIICh21Eq32',
        'FeynmanIICh34Eq2a', 'FeynmanIICh34Eq2', 'FeynmanIICh34Eq29b', 'FeynmanIICh37Eq1', 'FeynmanIIICh13Eq18',
        'FeynmanIIICh15Eq12', 'FeynmanIIICh15Eq27', 'FeynmanIIICh17Eq37', 'FeynmanIIICh19Eq51', 'FeynmanBonus5',
        'FeynmanBonus7', 'FeynmanBonus9', 'FeynmanBonus15', 'FeynmanBonus18']
var4 = ['FeynmanICh8Eq14', 'FeynmanICh13Eq4', 'FeynmanICh13Eq12', 'FeynmanICh18Eq4', 'FeynmanICh18Eq16',
        'FeynmanICh24Eq6', 'FeynmanICh29Eq16', 'FeynmanICh32Eq17', 'FeynmanICh34Eq8', 'FeynmanICh40Eq1',
        'FeynmanICh43Eq16', 'FeynmanICh44Eq4', 'FeynmanICh50Eq26', 'FeynmanIICh11Eq20', 'FeynmanIICh34Eq11',
        'FeynmanIICh35Eq18', 'FeynmanIICh35Eq21', 'FeynmanIICh38Eq3', 'FeynmanIIICh10Eq19', 'FeynmanIIICh14Eq14',
        'FeynmanIIICh21Eq20', 'FeynmanBonus1', 'FeynmanBonus3', 'FeynmanBonus11', 'FeynmanBonus19']
var5 = ['FeynmanICh12Eq11', 'FeynmanIICh2Eq42', 'FeynmanIICh6Eq15a', 'FeynmanIICh11Eq3', 'FeynmanIICh11Eq17',
        'FeynmanIICh36Eq38', 'FeynmanIIICh9Eq52', 'FeynmanBonus4', 'FeynmanBonus12', 'FeynmanBonus13', 'FeynmanBonus14',
        'FeynmanBonus16']

var678 = ['FeynmanICh11Eq19', 'FeynmanBonus2', 'FeynmanBonus17', 'FeynmanBonus6', 'FeynmanICh9Eq18']

if __name__ == '__main__':
    basepath = "/home/jiangnan/PycharmProjects/scibench/data/unencrypted/equations_feynman/{}.in"
    to_folder = "/home/jiangnan/PycharmProjects/scibench/eureqa/data/equations_feynman/{}"

    for prog in var2+var3+var4+var5+var678:
        filename = basepath.format(prog)
        print(prog)
        data_query_oracle = Equation_evaluator(filename, noise_type='normal', noise_scale=0.0, metric_name='neg_mse')
        dataX = DataX(data_query_oracle.get_vars_range_and_types())
        batchsize = 100000

        # X = np.random.rand(batchsize, n_input) * 9.5 + 0.5
        X = dataX.randn(sample_size=batchsize).T
        y = data_query_oracle.evaluate(X).reshape(-1, 1)
        print(X.shape, y.shape)
        filename_csv = to_folder.format(prog)
        to_csv(X, y, filename_csv)
        print(f"{prog} done......")
        # print(one_eq['eq_expression'].execute(X))
