# README: Vertical Symbolic Regression via Deep Policy Gradient#

## Directory

### Dataset

- `data`: the generated dataset. Every file represent a ground-truth expression.
- `src/scibench`: the dataoracle API to draw data.

#### Prerequisite of using these methods
- install the dependency package
```bash
pip install -r requirements.txt
```
- install our data oracle
```bash
cd src/scibench
pip install -e .
```

### Methods
- `cvDSO`: the proposed method.
- `ProGED`: from https://github.com/brencej/ProGED.
- `SPL`: symbolic physics learner, from https://github.com/isds-neu/SymbolicPhysicsLearner.
- `E2E`: End to end transformer for symbolic regression, from https://github.com/facebookresearch/symbolicregression.
- `gp_and_cvgp`: genetic programming  (GP) and VSR-GP algorithm, from https://github.com/jiangnanhugo/cvgp
- `dso_classic`: the codebase for DSR, VPG, PQT and GPMeld, from https://github.com/dso-org/deep-symbolic-optimization
- `odeformer`:

#### Extra
- plots: the jupyter notebook to generate our figure.
- result: contains all the output of all the programs, the training logs.


### 3. Look at the summarized result
Just open the `result` and `plots` folders.




## Citation

If you want to reuse this material, please considering citing the following:
```
@article{kamienny2022end,
  title={End-to-end symbolic regression with transformers},
  author={Kamienny, Pierre-Alexandre and d'Ascoli, St{\'e}phane and Lample, Guillaume and Charton, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:2204.10532},
  year={2022}
}
```

