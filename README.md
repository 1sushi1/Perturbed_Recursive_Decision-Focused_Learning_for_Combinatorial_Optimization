# Recursive Decision-Focused Learning Reproduction

This repository contains a compact PyTorch reproduction of the two R-DFL variants proposed in:

`From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback`

The two reproduced models are:

- `RDFLUnrolled`: R-DFL-U, explicit recursive unrolling with standard automatic differentiation.
- `RDFLImplicit`: R-DFL-I, fixed-point forward pass with implicit differentiation at equilibrium.

The implementation also includes continuous optimization layers for the two benchmark families in the paper:

- Recursive multi-product newsvendor problem.
- Recursive bipartite matching problem.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run a Synthetic Smoke Test

```bash
python examples/train_synthetic.py --problem newsvendor --model unrolled
python examples/train_synthetic.py --problem newsvendor --model implicit
python examples/train_synthetic.py --problem matching --model unrolled
python examples/train_synthetic.py --problem matching --model implicit
```

The examples use synthetic data because the paper's real-world matching dataset is not included in the PDF.

## Discrete Optimizers

This repo also includes differentiable perturbed optimizers following Berthet et al., `Learning with Differentiable Perturbed Optimizers`. The layer wraps a black-box discrete argmax oracle and estimates

`E_Z[argmax_y <theta + sigma Z, y>]`

with Monte Carlo Gaussian perturbations. Its backward pass uses the Gaussian integration-by-parts Jacobian estimator.

```bash
python examples/train_discrete.py --problem topk --model unrolled
python examples/train_discrete.py --problem matching --model unrolled
```

For a fuller comparison on discrete problems:

```bash
python examples/run_discrete_benchmark.py --problem topk --models all --epochs 50
python examples/run_discrete_benchmark.py --problem matching --models all --epochs 50
```

This compares PTO, SPO+, S-DFL with perturbed optimizers, perturbed R-DFL-U/I, and relaxed continuous R-DFL-U/I. By default, discrete experiments use `--data-mode recursive`, where hidden prices are generated as `c = f(v, x)` and the target decision is `x* = argmin_x c*^T x`.

For recursive data generation, use `--fixed-point-mode fixed --fixed-point-steps T` to choose a fixed number of feedback iterations, or `--fixed-point-mode converged --convergence-tol 1e-4 --max-fixed-point-steps 100` to iterate until convergence.
The initial decision state is sampled randomly before recursive feedback iterations begin.

To plot discrete convergence curves:

```bash
python examples/plot_discrete_convergence.py --problem topk --models all --epochs 50
python examples/plot_discrete_convergence.py --problem matching --models all --epochs 50
```

Available layers:

- `PerturbedOptimizerLayer`: generic black-box argmax wrapper.
- `PerturbedTopKLayer`: perturbed discrete top-k.
- `PerturbedBipartiteMatchingLayer`: perturbed Hungarian matching.

## Reproduce The Benchmark Comparison

The benchmark runner implements the paper's four methods: PTO, S-DFL, R-DFL-U, and R-DFL-I. It reports decision RMSE, cost RMSE, regret, and training time.

```bash
python examples/run_benchmark.py --problem newsvendor --scale small --models all --repeats 5 --epochs 50
python examples/run_benchmark.py --problem matching --scale small --models all --repeats 5 --epochs 50
```

The paper's matching benchmark uses a real-world dataset that is not distributed in the PDF, so this script defaults to `--data-mode recursive`: a synthetic recursive system where target costs are generated as `c = f(v, x*)` and `x* = G(c*)`. To mimic Table 1 scales, use `--scale small`, `--scale mid`, or `--scale large`.

## Plot Convergence Curves

```bash
python examples/plot_convergence.py --problem newsvendor --scale small --models all --epochs 50
python examples/plot_convergence.py --problem matching --scale small --models rdfl-u rdfl-i --epochs 50
```

The script writes both a CSV and a PNG under `outputs/`, for example:

- `outputs/convergence_newsvendor_small.csv`
- `outputs/convergence_newsvendor_small_test_regret.png`

## Code Map

- `rdfl/predictors.py`: MLP predictor `F_theta(v, x)`.
- `rdfl/optim_layers.py`: differentiable continuous optimization layers `G(c)`.
- `rdfl/perturbed.py`: differentiable perturbed discrete optimizer layers.
- `rdfl/models.py`: the two paper models, `RDFLUnrolled` and `RDFLImplicit`.
- `rdfl/baselines.py`: PTO and S-DFL baselines.
- `examples/train_synthetic.py`: end-to-end training script.
- `examples/run_benchmark.py`: comparison runner for PTO, S-DFL, R-DFL-U, and R-DFL-I.
- `examples/plot_convergence.py`: convergence curve runner that writes CSV and PNG outputs.

## Notes

The paper's newsvendor objective is linear, which can be non-unique and therefore not differentiable without additional assumptions. This implementation uses a small quadratic regularizer in the optimization layer so that the solution map is single-valued and differentiable, matching the differentiability assumption stated in the paper. The newsvendor projection layer uses an active-set KKT backward pass for the box and total-order constraints. The bipartite matching layer already follows the paper's quadratic regularization, but currently uses differentiable projected-gradient iterations rather than a hand-written KKT matrix.
