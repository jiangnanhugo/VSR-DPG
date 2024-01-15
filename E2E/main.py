import torch
import click
import sympy as sp
import symbolicregression
from scibench.symbolic_data_generator import *
from scibench.symbolic_equation_evaluator_public import Equation_evaluator
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, Symbol


def execute(expr_str: str, data_X: np.ndarray, input_var_Xs):
    """
    evaluate the output of expression with the given input.
    consts: list of constants.
    """
    expr = parse_expr(expr_str)
    used_vars, used_idx = [], []
    for idx, xi in enumerate(input_var_Xs):
        if str(xi) in expr_str:
            used_idx.append(idx)
            used_vars.append(xi)
    try:
        f = lambdify(used_vars, expr, 'numpy')
        if len(used_idx) != 0:
            y_hat = f(*[data_X[:, i] for i in used_idx])
        else:
            y_hat = float(expr)
        if y_hat is complex:
            return np.ones(data_X.shape[-1]) * np.infty
    except TypeError as e:
        # print(e, expr, input_var_Xs, data_X.shape)
        y_hat = np.ones(data_X.shape[-1]) * np.infty
    except KeyError as e:
        # print(e, expr)
        y_hat = np.ones(data_X.shape[-1]) * np.infty

    return y_hat


@click.command()
@click.option('--equation_name', default=None, type=str, help="Name of equation")
@click.option('--metric_name', default='inv_nrmse', type=str, help="evaluation metrics")
@click.option('--noise_type', default='normal', type=str, help="")
@click.option('--noise_scale', default=0.0, type=float, help="")
@click.option('--pretrained_model_filepath', type=str, help="pertrained pytorch model filepath")
def main(equation_name, metric_name, noise_type, noise_scale, pretrained_model_filepath):
    """Runs DSO in parallel across multiple seeds using multiprocessing."""

    model = torch.load(pretrained_model_filepath, map_location=torch.device('cpu'))

    est = symbolicregression.model.SymbolicTransformerRegressor(
        model=model,
        max_input_points=200,
        n_trees_to_refine=100,
        rescale=True
    )

    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name)
    dataX = DataX(data_query_oracle.get_vars_range_and_types())
    nvars = data_query_oracle.get_nvars()
    input_var_Xs = [Symbol(f'X{i}') for i in range(nvars)]
    regress_batchsize = 2560
    X_train = dataX.randn(sample_size=regress_batchsize).T

    print(X_train.shape)
    y_train = data_query_oracle.evaluate(X_train)
    hof = []
    regress_batchsize = 256
    X_test = dataX.randn(sample_size=regress_batchsize).T
    for i in range(20):
        est.fit(X_train, y_train)
        replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/", 'nan':'1', 'inf': '1'}
        model_str = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()

        for op, replace_op in replace_ops.items():
            model_str = model_str.replace(op, replace_op)
        for i in range(nvars):
            model_str = model_str.replace(f'x_{i}', f"X{i}")
        print(sp.parse_expr(model_str))

        ypred_test = execute(model_str, X_test, input_var_Xs)

        dict_of_result = data_query_oracle._evaluate_all_losses(X_test, ypred_test)
        hof.append((sp.parse_expr(model_str), dict_of_result))
        # dict_of_result['tree_edit_distance'] = self.task.data_query_oracle.compute_normalized_tree_edit_distance(expr_str)
    hof = sorted(hof, key=lambda x: x[1]['neg_nmse'], reverse=True)
    for expr, dict_of_result in hof:
        print('\t', expr)
        print('-' * 30)
        for mertic_name in dict_of_result:
            print(f"{mertic_name} {dict_of_result[mertic_name]}")
        print('-' * 30)


if __name__ == "__main__":
    main()
