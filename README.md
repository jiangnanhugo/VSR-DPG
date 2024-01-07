# README: Vertical Symbolic Regression #

### Directory

### Data Oracle

- `data`: the generated dataset. Every file represent a ground-truth expression.
- `src/scibench`: the dataoracle API.

#### Baselines
- `ProGED`: from https://github.com/brencej/ProGED.


#### Extra
- plots: the jupyter notebook to generate our figure.
- result: contains all the output of all the programs, the training logs.


### 3. Look at the summarized result
Just open the `result` and `plots` folders.


# Prerequisite of using these methods
- install the dependency package
```bash
pip install -r requirements.txt
```
- install our data oracle
```bash
cd src/scibench
pip install -e .
```

- If you want to run the genetic programming (GP) or our proposed VSR-GP, please goto the `go_vsr_gp` folder.
