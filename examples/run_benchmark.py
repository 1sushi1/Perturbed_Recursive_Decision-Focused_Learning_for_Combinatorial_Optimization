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
    """One-way synthetic cost data used for smoke tests.

    This is intentionally simple, but it is not the right data model for
    demonstrating R-DFL's feedback advantage.
    """

    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, feature_dim, generator=generator)
    weights = torch.randn(feature_dim, decision_dim, generator=generator) / feature_dim**0.5
    costs = features @ weights
    if nonlinear:
        hidden = torch.randn(feature_dim, decision_dim, generator=generator) / feature_dim**0.5
        costs = costs + 0.35 * torch.sin(features @ hidden)
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
) -> TensorDataset:
    """Closed-loop synthetic data for the paper's recursive setting.

    The hidden data-generating system is c = f(v, x). We first find the
    equilibrium x* = G(f(v, x*)) and then expose only (v, c*) to the learner,
    matching the decision-focused training interface.
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
        x = torch.zeros(n_samples, decision_dim)
        for _ in range(fixed_point_steps):
            costs = oracle_cost(features, x)
            x = optimizer_layer(costs)
        true_costs = oracle_cost(features, x)
        true_costs = true_costs + 0.05 * torch.randn(n_samples, decision_dim, generator=generator)
    return TensorDataset(features, true_costs)


def build_problem(problem: str, scale: str) -> ProblemSpec:
    feature_dim = 8
    if problem == "newsvendor":
        decision_dims = {"small": 10, "mid": 50, "large": 100}
        decision_dim = decision_dims[scale]
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
        sides = {"small": 4, "mid": 15, "large": 30}
        side = sides[scale]
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
        if args.data_mode == "recursive":
            dataset = make_recursive_synthetic_data(
                args.samples,
                spec.feature_dim,
                spec.decision_dim,
                spec.optimizer_layer,
                seed,
                feedback_strength=args.feedback_strength,
            )
        else:
            dataset = make_synthetic_data(args.samples, spec.feature_dim, spec.decision_dim, seed)
        train_set, _, test_set = split_dataset(dataset, seed)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        for model_name in model_names:
            torch.manual_seed(seed)
            spec = build_problem(args.problem, args.scale)
            model = build_model(model_name, spec, args.steps, args.backward_steps)
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
    parser.add_argument("--data-mode", choices=["one_way", "recursive"], default="one_way")
    parser.add_argument("--feedback-strength", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backward-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
