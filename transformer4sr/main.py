import torch
import sympy
from collections import OrderedDict

from model.transformer_model import TransformerModel
from model._utils import count_nb_params
from model._utils import is_tree_complete
from model._utils import translate_integers_into_tokens
from datasets._utils import from_sequence_to_sympy

import time
import argparse
import random
import numpy as np
from scibench.symbolic_data_generator import DataX
from scibench.symbolic_equation_evaluator_public import Equation_evaluator


def construct_transformer_model(nb_samples, max_nb_var, model_basepath):
    # First reload big model
    transformer = TransformerModel(
        enc_type='mix',
        nb_samples=nb_samples,  # Number of samples par dataset
        max_nb_var=7,  # Max number of variables
        d_model=256,
        vocab_size=18 + 2,  # len(vocab) + padding token + <SOS> token
        seq_length=30,  # vocab_size + 1 - 1 (add <SOS> but shifted right)
        h=4,
        N_enc=4,
        N_dec=8,
        dropout=0.25,
    )
    total_nb_params = count_nb_params(transformer, print_all=False)
    print(f'Total number params = {total_nb_params}')

    # In[3]:

    PATH_WEIGHTS = model_basepath + '/best_model_weights/mix_label_smoothing/model_weights.pt'
    hixon_state_dict = torch.load(PATH_WEIGHTS, map_location=torch.device('cpu'))

    my_state_dict = OrderedDict()
    for key in hixon_state_dict.keys():
        assert key[:7] == "module."
        my_state_dict[key[7:]] = hixon_state_dict[key]

    out = transformer.load_state_dict(my_state_dict, strict=True)
    transformer.eval()  # deactivate training mode (important)
    print(out)  # This should print <All keys matched susccessfully>
    return transformer


def decode_with_transformer(transformer, dataset):
    """
    Greedy decode with the Transformer model.
    Decode until the equation tree is completed.
    Parameters:
      - transformer: torch Module object
      - dataset: tabular dataset
      shape = (batch_size=1, nb_samples=50, nb_max_var=7, 1)
    """
    encoder_output = transformer.encoder(dataset)  # Encoder output is fixed for the batch
    seq_length = transformer.decoder.positional_encoding.seq_length
    decoder_output = torch.zeros((dataset.shape[0], seq_length + 1), dtype=torch.int64)  # initialize Decoder output
    decoder_output[:, 0] = 1
    is_complete = torch.zeros(dataset.shape[0], dtype=torch.bool)  # check when decoding is finished

    for n1 in range(seq_length):
        padding_mask = torch.eq(decoder_output[:, :-1], 0).unsqueeze(1).unsqueeze(1)
        future_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        mask_dec = torch.logical_or(padding_mask, future_mask)
        temp = transformer.decoder(
            target_seq=decoder_output[:, :-1],
            mask_dec=mask_dec,
            output_enc=encoder_output,
        )
        temp = transformer.last_layer(temp)
        decoder_output[:, n1 + 1] = torch.where(is_complete, 0, torch.argmax(temp[:, n1], axis=-1))
        for n2 in range(dataset.shape[0]):
            if is_tree_complete(decoder_output[n2, 1:]):
                is_complete[n2] = True
    return decoder_output


def run_transformer(dataset, nb_samples, max_nb_var, model_basepath, total_predictions=1000):
    # Generate input for the Encoder using torch.Tensor object
    encoder_input = torch.Tensor(dataset).unsqueeze(0).unsqueeze(-1)


    set_of_preds=set()
    for i in range(total_predictions):
        transformer = construct_transformer_model(nb_samples, max_nb_var, model_basepath)
        decoder_output = decode_with_transformer(transformer, encoder_input)
        decoder_tokens = translate_integers_into_tokens(decoder_output[0])
        sympy_pred = from_sequence_to_sympy(decoder_tokens)
        print(i, sympy_pred)
        set_of_preds.add(sympy_pred)
    for si in set_of_preds:
        print(si)


def main(equation_name, metric_name, noise_type, noise_scale, model_basepath):
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name)
    dataX = DataX(data_query_oracle.get_vars_range_and_types())
    nvars = data_query_oracle.get_nvars()
    regress_batchsize = 500
    operators_set = data_query_oracle.get_operators_set()
    X_train = dataX.randn(sample_size=regress_batchsize).T
    print(X_train.shape)
    y_train = data_query_oracle.evaluate(X_train)

    dataset = np.zeros((regress_batchsize, 7))
    dataset[:, 0] = y_train

    for i in range(nvars):
        dataset[:, i + 1] = X_train[:, i]

    start = time.time()
    run_transformer(dataset, regress_batchsize, nvars+1, model_basepath)
    end_time = time.time() - start

    print("Tansformer4SR {} mins".format(np.round(end_time / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation_name", help="the filename of the true program.")
    parser.add_argument('--optimizer',
                        nargs='?',
                        choices=['BFGS', 'L-BFGS-B', 'Nelder-Mead', 'CG', 'basinhopping', 'dual_annealing', 'shgo', 'direct'],
                        help='list servers, storage, or both (default: %(default)s)')
    parser.add_argument("--model_basepath", type=str, default='', help="The name of the metric for loss.")
    parser.add_argument("--metric_name", type=str, default='neg_mse', help="The name of the metric for loss.")
    parser.add_argument("--noise_type", type=str, default='normal', help="The name of the noises.")
    parser.add_argument("--noise_scale", type=float, default=0.0, help="This parameter adds the standard deviation of the noise")

    args = parser.parse_args()

    seed = int(time.perf_counter() * 10000) % 1000007
    random.seed(seed)
    print('random seed=', seed)

    seed = int(time.perf_counter() * 10000) % 1000007
    np.random.seed(seed)
    print('np.random seed=', seed)
    print(args)

    main(args.equation_name, args.metric_name, args.noise_type, args.noise_scale, args.model_basepath)
