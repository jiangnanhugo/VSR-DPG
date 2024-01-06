# README: Vertical Symbolic Regression #

### Directory

### Data Oracle

- `data`: the generated dataset. Every file represent a ground-truth expression.
- `src/scibench`: the dataoracle API.

#### Baselines
- `dso_classic`: public code implementation from https://github.com/brendenpetersen/deep-symbolic-optimization. It contains the imeplementation of methods `DSR, PQT, VPG, GPMeld `.
- `gp_and_vsr_gp`: the re-implementation of the our proposed control variable genetic programming algorithm (https://github.com/jiangnanhugo/cvgp) and the classic genetic programming algorithm. We change the code that is relevant to the dataloader.
- `mcts_and_vsr_mcts`: the implementation of the our proposed vertical discovery path for Monte Carlo tree search and the classic Monte Carlo tree search algorithm.
- `Eureqa`: the commercial genetic search algorithm.

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
- If you want to run Monte Carlo tree search (MCTS) or our proposed VSR-MCTS, please goto the `mcts_and_vsr_mcts` folder.
- If you want to run deep reinforcement learning based baselines (DSR, PQT, VPG), please goto the `dso_classic` folder. 
You need to install the `DSO` library and a specific Python interpreter 3.7 (due to the dependecy on the tensorflow with version 1.15.4) before running the method.
- If you want to run the commercial evolutionary search algorithm (Eureqa), please goto the `eureqa` folder.

In each folder, we provide a detailed steps and scripts for you to run the program.

