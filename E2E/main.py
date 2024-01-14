
import torch
import numpy as np
import sympy as sp
import symbolicregression
model_path = "model.pt"
model = torch.load(model_path, map_location=torch.device('cpu'))

est = symbolicregression.model.SymbolicTransformerRegressor(
    model=model,
    max_input_points=200,
    n_trees_to_refine=100,
    rescale=True
)

x = np.random.randn(100, 2)
y = np.cos(2 * np.pi * x[:, 0]) + x[:, 1] ** 2


est.fit(x, y)
replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
model_str = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()

for op, replace_op in replace_ops.items():
    model_str = model_str.replace(op, replace_op)
print(sp.parse_expr(model_str))
