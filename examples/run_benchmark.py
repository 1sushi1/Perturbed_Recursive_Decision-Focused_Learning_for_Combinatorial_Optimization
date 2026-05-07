from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from rdfl import BipartiteMatchingLayer, FeatureMLP, FeedbackMLP, NewsvendorLayer, RDFLImplicit, RDFLUnrolled
from rdfl.baselines import PTOPipeline, SDFLModel


@dataclass(frozen=True)
class ProblemSpec:
    feature_dim: int
    decision_dim: int
    optimizer_layer: torch.nn.Module


def make_synthetic_data(
    n_samples: int,
    feature_dim: int,
    decision_dim: int,
    seed: int,
    nonlinear: bool = True,
) -> TensorDataset:
    """Generate a non-recursive synthetic `(features, costs)` dataset.

    This is the simple one-way data mode:

        v ~ Normal(0, I)
        c = v W + 0.35 * sin(v H) + noise

    where `v` is the observed context/features and `c` is the target cost
    vector used by the downstream optimization layer. Importantly, this
    generator does *not* use a decision variable `x` when producing costs.
    It is therefore useful as a PTO/S-DFL-style control setting or a quick
    smoke test, but it does not contain the recursive feedback structure that
    R-DFL is designed to model. The default benchmark path uses
    `make_recursive_synthetic_data`; this function is selected only with
    `--data-mode one_way`.

    Returns:
        TensorDataset containing pairs `(features, costs)` with shapes
        `[n_samples, feature_dim]` and `[n_samples, decision_dim]`.
    """

    # Use an explicit generator so the same seed always recreates the same
    # synthetic dataset, independent of other PyTorch randomness.
    generator = torch.Generator().manual_seed(seed)

    # Context/features v for each sample.
    features = torch.randn(n_samples, feature_dim, generator=generator)

    # Linear map W from context space to cost space. The sqrt(feature_dim)
    # scaling keeps the cost magnitude roughly stable as feature_dim changes.
    weights = torch.randn(feature_dim, decision_dim, generator=generator) / feature_dim**0.5
    costs = features @ weights

    if nonlinear:
        # Add a small nonlinear component so the prediction task is not purely
        # linear while still remaining easy enough for the MLP baseline.
        hidden = torch.randn(feature_dim, decision_dim, generator=generator) / feature_dim**0.5
        costs = costs + 0.35 * torch.sin(features @ hidden)

    # Observation noise in the target costs.
    costs = costs + 0.15 * torch.randn(n_samples, decision_dim, generator=generator)
    return TensorDataset(features, costs)


def make_recursive_synthetic_data(
    n_samples: int,
    feature_dim: int,
    decision_dim: int,
    optimizer_layer: torch.nn.Module,
    seed: int,
    feedback_strength: float = 0.7,
    fixed_point_steps: int = 30,
    fixed_point_mode: str = "fixed",
    convergence_tol: float = 1e-4,
    max_fixed_point_steps: int = 100,
) -> TensorDataset:
    """Closed-loop synthetic data for the paper's recursive setting.

    Hidden pipeline:

        c_t = f(v, x_t)
        x_{t+1} = G(c_t)

    The returned target costs are generated at the final decision state, and
    evaluation uses x* = G(c*) as the ground-truth decision.
    """

    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, feature_dim, generator=generator)
    feature_weights = torch.randn(feature_dim, decision_dim, generator=generator) / feature_dim**0.5
    feedback = torch.randn(decision_dim, decision_dim, generator=generator) / decision_dim**0.5
    feedback = feedback_strength * feedback

    def oracle_cost(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        base = v @ feature_weights
        rec = torch.tanh(x @ feedback)
        return base + rec

    with torch.no_grad():
        # Use the same fixed initial decision as the R-DFL models. This keeps
        # the data-generation recursion and the model recursion aligned:
        #
        #   data:  x0 = 0 -> c1 = f(v, x0) -> x1 = G(c1) -> ...
        #   model: x0 = 0 -> c1 = F_theta(v, x0) -> x1 = G(c1) -> ...
        #
        # If this starts from random decisions while the model starts from
        # zeros, the benchmark compares two different closed-loop systems.
        x = torch.zeros(n_samples, decision_dim)
        total_steps = fixed_point_steps if fixed_point_mode == "fixed" else max_fixed_point_steps
        for _ in range(total_steps):
            costs = oracle_cost(features, x)
            next_x = optimizer_layer(costs)
            if fixed_point_mode == "converged":
                delta = (next_x - x).norm() / (1.0 + x.norm())
                x = next_x
                if delta <= convergence_tol:
                    break
            else:
                x = next_x
        true_costs = oracle_cost(features, x)
        x_star = optimizer_layer(true_costs)
        true_costs = oracle_cost(features, x_star)
        true_costs = true_costs + 0.05 * torch.randn(n_samples, decision_dim, generator=generator)
    return TensorDataset(features, true_costs)


def build_problem(problem: str, scale: str) -> ProblemSpec:
    """Create the continuous benchmark optimization layer and dimensions.

    This function centralizes the experiment configuration for the two
    continuous benchmark families used in the R-DFL paper:

    - `newsvendor`: multi-product newsvendor / R-MPNP.
    - `matching`: continuous relaxation of bipartite matching / R-BMP.

    The returned `ProblemSpec` tells the rest of the training code how many
    feature dimensions to generate, how many cost/decision variables the
    predictor should output, and which downstream optimizer `G(c)` should be
    used to turn predicted costs into decisions.
    """

    # Synthetic contexts v are always 8-dimensional in these experiments.
    feature_dim = 8

    if problem == "newsvendor":
        # Match the paper's small/mid/large newsvendor decision dimensions:
        # 10, 50, and 100 products.
        decision_dims = {"small": 10, "mid": 50, "large": 100}
        decision_dim = decision_dims[scale]

        # Continuous strongly-convex newsvendor layer:
        #   min c^T x + 0.5 * lambda * ||x||^2
        #   s.t. 0 <= x_i <= 1 and 0.3n <= sum_i x_i <= 0.6n.
        layer = NewsvendorLayer(
            decision_dim=decision_dim,
            sum_lower=0.3 * decision_dim,
            sum_upper=0.6 * decision_dim,
            lower=0.0,
            upper=1.0,
            l2_strength=1.0,
        )
        return ProblemSpec(feature_dim, decision_dim, layer)

    if problem == "matching":
        # Match the paper's small/mid/large matching dimensions:
        # 4x4=16, 15x15=225, and 30x30=900 decision variables.
        sides = {"small": 4, "mid": 15, "large": 30}
        side = sides[scale]

        # Continuous bipartite matching relaxation. Large instances use fewer
        # projection/solver iterations to keep experiments reasonably fast.
        layer = BipartiteMatchingLayer(
            num_left=side,
            num_right=side,
            min_matches=max(1.0, side / 2.0),
            l2_strength=0.1,
            step_size=0.2,
            projection_iters=15 if scale == "large" else 25,
            solver_iters=30 if scale == "large" else 60,
        )
        return ProblemSpec(feature_dim, side * side, layer)

    raise ValueError(f"unknown problem: {problem}")


def split_dataset(dataset: TensorDataset, seed: int):
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def build_model(name: str, spec: ProblemSpec, steps: int, backward_steps: int):
    if name == "pto":
        predictor = FeatureMLP(spec.feature_dim, spec.decision_dim, hidden_dim=32, dropout=0.1)
        return PTOPipeline(predictor, spec.optimizer_layer)
    if name == "sdfl":
        predictor = FeatureMLP(spec.feature_dim, spec.decision_dim, hidden_dim=32, dropout=0.1)
        return SDFLModel(predictor, spec.optimizer_layer)
    if name == "rdfl-u":
        predictor = FeedbackMLP(spec.feature_dim, spec.decision_dim, hidden_dim=32, dropout=0.1)
        return RDFLUnrolled(predictor, spec.optimizer_layer, unroll_steps=steps)
    if name == "rdfl-i":
        predictor = FeedbackMLP(spec.feature_dim, spec.decision_dim, hidden_dim=32, dropout=0.1)
        return RDFLImplicit(predictor, spec.optimizer_layer, rootfind_steps=steps, backward_steps=backward_steps)
    raise ValueError(f"unknown model: {name}")


def train_one(model_name: str, model, opt_layer, train_loader, epochs: int, lr: float) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for features, true_costs in train_loader:
            if model_name == "pto":
                _, pred_costs = model(features, return_costs=True)
                loss = F.mse_loss(pred_costs, true_costs)
            else:
                decisions = model(features)
                loss = opt_layer.regret(decisions, true_costs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return time.perf_counter() - start


@torch.no_grad()
def evaluate(model, opt_layer, loader) -> dict[str, float]:
    model.eval()
    regrets = []
    decision_sqerr = []
    cost_sqerr = []
    count = 0
    for features, true_costs in loader:
        decisions, pred_costs = model(features, return_costs=True)
        true_decisions = opt_layer(true_costs)
        regrets.append(opt_layer.regret(decisions, true_costs).sum().item())
        decision_sqerr.append((decisions - true_decisions).square().sum().item())
        cost_sqerr.append((pred_costs - true_costs).square().sum().item())
        count += features.shape[0]
    dim = next(iter(loader))[0].shape[0]
    del dim
    decision_dim = opt_layer.decision_dim
    total_items = count * decision_dim
    return {
        "regret": sum(regrets) / count,
        "decision_rmse": (sum(decision_sqerr) / total_items) ** 0.5,
        "cost_rmse": (sum(cost_sqerr) / total_items) ** 0.5,
    }


def summarize(rows: list[dict[str, float | str]]) -> None:
    names = []
    for row in rows:
        if row["model"] not in names:
            names.append(row["model"])

    print("model,decision_rmse,cost_rmse,regret,time_sec")
    for name in names:
        subset = [row for row in rows if row["model"] == name]
        metrics = {}
        for key in ["decision_rmse", "cost_rmse", "regret", "time_sec"]:
            values = [float(row[key]) for row in subset]
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            metrics[key] = f"{mean:.6f}±{std:.6f}"
        print(f"{name},{metrics['decision_rmse']},{metrics['cost_rmse']},{metrics['regret']},{metrics['time_sec']}")


def run(args: argparse.Namespace) -> None:
    model_names = ["pto", "sdfl", "rdfl-u", "rdfl-i"] if args.models == ["all"] else args.models
    rows = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        torch.manual_seed(seed)
        spec = build_problem(args.problem, args.scale)
        if args.data_mode != "recursive":
            dataset = make_synthetic_data(args.samples, spec.feature_dim, spec.decision_dim, seed)
        else:
            dataset = make_recursive_synthetic_data(
                args.samples,
                spec.feature_dim,
                spec.decision_dim,
                spec.optimizer_layer,
                seed,
                feedback_strength=args.feedback_strength,
                fixed_point_steps=args.fixed_point_steps,
                fixed_point_mode=args.fixed_point_mode,
                convergence_tol=args.convergence_tol,
                max_fixed_point_steps=args.max_fixed_point_steps,
            )
            
        train_set, _, test_set = split_dataset(dataset, seed)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        for model_name in model_names:
            torch.manual_seed(seed)
            spec = build_problem(args.problem, args.scale)
            model = build_model(model_name, spec, args.steps, args.backward_steps)
            print(f"starting repeat={repeat + 1} model={model_name}", flush=True)
            elapsed = train_one(model_name, model, spec.optimizer_layer, train_loader, args.epochs, args.lr)
            metrics = evaluate(model, spec.optimizer_layer, test_loader)
            rows.append({"model": model_name, "time_sec": elapsed, **metrics})
            print(
                f"repeat={repeat + 1} model={model_name} "
                f"decision_rmse={metrics['decision_rmse']:.6f} "
                f"regret={metrics['regret']:.6f} time={elapsed:.2f}s"
            )

    summarize(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["newsvendor", "matching"], default="newsvendor")
    parser.add_argument("--scale", choices=["small", "mid", "large"], default="small")
    parser.add_argument("--models", nargs="+", choices=["all", "pto", "sdfl", "rdfl-u", "rdfl-i"], default=["all"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--data-mode", choices=["recursive", "one_way"], default="recursive")
    parser.add_argument("--feedback-strength", type=float, default=0.7)
    parser.add_argument("--fixed-point-steps", type=int, default=30)
    parser.add_argument("--fixed-point-mode", choices=["fixed", "converged"], default="fixed")
    parser.add_argument("--convergence-tol", type=float, default=1e-2)
    parser.add_argument("--max-fixed-point-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backward-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
