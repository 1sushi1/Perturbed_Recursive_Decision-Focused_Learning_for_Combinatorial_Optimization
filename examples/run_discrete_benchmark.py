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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from rdfl import (
    BipartiteMatchingLayer,
    FeatureMLP,
    FeedbackMLP,
    PerturbedBipartiteMatchingLayer,
    PerturbedTopKLayer,
    RDFLImplicit,
    RDFLUnrolled,
    RelaxedTopKLayer,
)
from rdfl.baselines import SDFLModel
from rdfl.perturbed import BipartiteMatchingOracle, TopKOracle


@dataclass(frozen=True)
class DiscreteSpec:
    feature_dim: int
    decision_dim: int
    oracle: object
    perturbed_layer: nn.Module
    relaxed_layer: nn.Module


class CostPredictorPipeline(nn.Module):
    def __init__(self, predictor: nn.Module, oracle) -> None:
        super().__init__()
        self.predictor = predictor
        self.oracle = oracle

    def forward(self, features: torch.Tensor, return_costs: bool = False):
        pred_costs = self.predictor(features)
        with torch.no_grad():
            decisions = self.oracle(-pred_costs)
        if return_costs:
            return decisions, pred_costs
        return decisions


def make_synthetic_costs(n_samples: int, feature_dim: int, decision_dim: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, feature_dim, generator=generator)
    weights = torch.randn(feature_dim, decision_dim, generator=generator) / feature_dim**0.5
    costs = features @ weights
    costs = costs + 0.35 * torch.sin(features @ torch.randn(feature_dim, decision_dim, generator=generator))
    costs = costs + 0.15 * torch.randn(n_samples, decision_dim, generator=generator)
    return TensorDataset(features, costs)


def make_recursive_synthetic_costs(
    n_samples: int,
    feature_dim: int,
    decision_dim: int,
    oracle,
    seed: int,
    feedback_strength: float = 0.7,
    fixed_point_steps: int = 20,
    fixed_point_mode: str = "fixed",
    convergence_tol: float = 1e-4,
    max_fixed_point_steps: int = 100,
) -> TensorDataset:
    """Generate recursive discrete benchmark data.

    Hidden pipeline:

        c_t = f(v, x_t)
        x_{t+1} = argmin_x c_t^T x

    The returned target cost c* is the price at the final decision state, and
    evaluation uses x* = argmin_x c*^T x as the ground-truth decision.
    """

    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, feature_dim, generator=generator)
    feature_weights = torch.randn(feature_dim, decision_dim, generator=generator) / feature_dim**0.5
    feedback = torch.randn(decision_dim, decision_dim, generator=generator) / decision_dim**0.5
    feedback = feedback_strength * feedback

    def price_fn(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        base = v @ feature_weights
        decision_effect = torch.tanh(x @ feedback)
        return base + decision_effect

    with torch.no_grad():
        random_scores = torch.randn(n_samples, decision_dim, generator=generator)
        x = oracle(random_scores)
        total_steps = fixed_point_steps if fixed_point_mode == "fixed" else max_fixed_point_steps
        for _ in range(total_steps):
            costs = price_fn(features, x)
            next_x = oracle(-costs)
            if fixed_point_mode == "converged":
                delta = (next_x - x).norm() / (1.0 + x.norm())
                x = next_x
                if delta <= convergence_tol:
                    break
            else:
                x = next_x
        costs = price_fn(features, x)
        x_star = oracle(-costs)
        costs = price_fn(features, x_star)
        costs = costs + 0.05 * torch.randn(n_samples, decision_dim, generator=generator)
    return TensorDataset(features, costs)


def split_dataset(dataset: TensorDataset, seed: int):
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    return random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))


def build_spec(args: argparse.Namespace) -> DiscreteSpec:
    feature_dim = 8
    if args.problem == "topk":
        oracle = TopKOracle(args.k)
        perturbed = PerturbedTopKLayer(
            decision_dim=args.decision_dim,
            k=args.k,
            num_samples=args.perturbations,
            sigma=args.sigma,
        )
        relaxed = RelaxedTopKLayer(args.decision_dim, args.k, l2_strength=args.relaxed_l2)
        return DiscreteSpec(feature_dim, args.decision_dim, oracle, perturbed, relaxed)

    if args.problem == "matching":
        decision_dim = args.side * args.side
        oracle = BipartiteMatchingOracle(args.side)
        perturbed = PerturbedBipartiteMatchingLayer(
            side=args.side,
            num_samples=args.perturbations,
            sigma=args.sigma,
        )
        relaxed = BipartiteMatchingLayer(
            num_left=args.side,
            num_right=args.side,
            min_matches=float(args.side),
            l2_strength=args.relaxed_l2,
            solver_iters=args.relaxed_iters,
        )
        return DiscreteSpec(feature_dim, decision_dim, oracle, perturbed, relaxed)

    raise ValueError(f"unknown problem: {args.problem}")


def build_model(name: str, spec: DiscreteSpec, args: argparse.Namespace):
    if name in {"pto", "spo-plus"}:
        predictor = FeatureMLP(spec.feature_dim, spec.decision_dim, hidden_dim=32, dropout=0.1)
        return CostPredictorPipeline(predictor, spec.oracle)
    if name == "sdfl-perturbed":
        predictor = FeatureMLP(spec.feature_dim, spec.decision_dim, hidden_dim=32, dropout=0.1)
        return SDFLModel(predictor, spec.perturbed_layer)
    if name == "rdfl-perturbed-u":
        predictor = FeedbackMLP(spec.feature_dim, spec.decision_dim, hidden_dim=32, dropout=0.1)
        return RDFLUnrolled(predictor, spec.perturbed_layer, unroll_steps=args.steps)
    if name == "rdfl-perturbed-i":
        predictor = FeedbackMLP(spec.feature_dim, spec.decision_dim, hidden_dim=32, dropout=0.1)
        return RDFLImplicit(
            predictor,
            spec.perturbed_layer,
            rootfind_steps=args.steps,
            backward_steps=args.backward_steps,
        )
    if name == "rdfl-relaxed-u":
        predictor = FeedbackMLP(spec.feature_dim, spec.decision_dim, hidden_dim=32, dropout=0.1)
        return RDFLUnrolled(predictor, spec.relaxed_layer, unroll_steps=args.steps)
    if name == "rdfl-relaxed-i":
        predictor = FeedbackMLP(spec.feature_dim, spec.decision_dim, hidden_dim=32, dropout=0.1)
        return RDFLImplicit(
            predictor,
            spec.relaxed_layer,
            rootfind_steps=args.steps,
            backward_steps=args.backward_steps,
        )
    raise ValueError(f"unknown model: {name}")


def spo_plus_loss(pred_costs: torch.Tensor, true_costs: torch.Tensor, oracle) -> torch.Tensor:
    """SPO+ surrogate for min_{w in S} c^T w.

    loss = max_w (c - 2 c_hat)^T w + 2 c_hat^T w*(c) - c^T w*(c)
    """

    with torch.no_grad():
        true_decisions = oracle(-true_costs)
        adversarial = oracle(-(2.0 * pred_costs - true_costs))
    max_term = ((true_costs - 2.0 * pred_costs) * adversarial).sum(dim=-1)
    correction = (2.0 * pred_costs * true_decisions).sum(dim=-1)
    constant = (true_costs * true_decisions).sum(dim=-1)
    return max_term + correction - constant


def discrete_regret(decisions: torch.Tensor, true_costs: torch.Tensor, oracle) -> torch.Tensor:
    with torch.no_grad():
        true_decisions = oracle(-true_costs)
    return (true_costs * decisions).sum(dim=-1) - (true_costs * true_decisions).sum(dim=-1)


def train_one(name: str, model, spec: DiscreteSpec, train_loader, args: argparse.Namespace) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    start = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for features, true_costs in train_loader:
            if name == "pto":
                _, pred_costs = model(features, return_costs=True)
                loss = F.mse_loss(pred_costs, true_costs)
            elif name == "spo-plus":
                _, pred_costs = model(features, return_costs=True)
                loss = spo_plus_loss(pred_costs, true_costs, spec.oracle).mean()
            else:
                decisions = model(features)
                loss = discrete_regret(decisions, true_costs, spec.oracle).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return time.perf_counter() - start


@torch.no_grad()
def evaluate(model, spec: DiscreteSpec, loader) -> dict[str, float]:
    model.eval()
    regret_sum = 0.0
    sqerr_sum = 0.0
    count = 0
    for features, true_costs in loader:
        decisions = model(features)
        true_decisions = spec.oracle(-true_costs)
        regret_sum += discrete_regret(decisions, true_costs, spec.oracle).sum().item()
        sqerr_sum += (decisions - true_decisions).square().sum().item()
        count += features.shape[0]
    return {
        "regret": regret_sum / count,
        "decision_rmse": (sqerr_sum / (count * spec.decision_dim)) ** 0.5,
    }


def summarize(rows: list[dict[str, float | str]]) -> None:
    names = []
    for row in rows:
        if row["model"] not in names:
            names.append(str(row["model"]))
    print("model,decision_rmse,regret,time_sec")
    for name in names:
        subset = [row for row in rows if row["model"] == name]
        parts = []
        for key in ["decision_rmse", "regret", "time_sec"]:
            values = [float(row[key]) for row in subset]
            parts.append(f"{statistics.mean(values):.6f}±{statistics.pstdev(values) if len(values) > 1 else 0.0:.6f}")
        print(f"{name},{parts[0]},{parts[1]},{parts[2]}")


def run(args: argparse.Namespace) -> None:
    default_models = [
        "pto",
        "spo-plus",
        "sdfl-perturbed",
        "rdfl-perturbed-u",
        "rdfl-perturbed-i",
        "rdfl-relaxed-u",
        "rdfl-relaxed-i",
    ]
    model_names = default_models if args.models == ["all"] else args.models
    rows = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        torch.manual_seed(seed)
        spec = build_spec(args)
        if args.data_mode == "recursive":
            dataset = make_recursive_synthetic_costs(
                args.samples,
                spec.feature_dim,
                spec.decision_dim,
                spec.oracle,
                seed,
                feedback_strength=args.feedback_strength,
                fixed_point_steps=args.fixed_point_steps,
                fixed_point_mode=args.fixed_point_mode,
                convergence_tol=args.convergence_tol,
                max_fixed_point_steps=args.max_fixed_point_steps,
            )
        else:
            dataset = make_synthetic_costs(args.samples, spec.feature_dim, spec.decision_dim, seed)
        train_set, _, test_set = split_dataset(dataset, seed)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)
        for name in model_names:
            torch.manual_seed(seed)
            spec = build_spec(args)
            model = build_model(name, spec, args)
            elapsed = train_one(name, model, spec, train_loader, args)
            metrics = evaluate(model, spec, test_loader)
            rows.append({"model": name, "time_sec": elapsed, **metrics})
            print(
                f"repeat={repeat + 1} model={name} "
                f"decision_rmse={metrics['decision_rmse']:.6f} "
                f"regret={metrics['regret']:.6f} time={elapsed:.2f}s"
            )
    summarize(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["topk", "matching"], default="topk")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "all",
            "pto",
            "spo-plus",
            "sdfl-perturbed",
            "rdfl-perturbed-u",
            "rdfl-perturbed-i",
            "rdfl-relaxed-u",
            "rdfl-relaxed-i",
        ],
        default=["all"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--data-mode", choices=["recursive", "one_way"], default="recursive")
    parser.add_argument("--feedback-strength", type=float, default=0.7)
    parser.add_argument("--fixed-point-steps", type=int, default=20)
    parser.add_argument("--fixed-point-mode", choices=["fixed", "converged"], default="fixed")
    parser.add_argument("--convergence-tol", type=float, default=1e-4)
    parser.add_argument("--max-fixed-point-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decision-dim", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--side", type=int, default=4)
    parser.add_argument("--perturbations", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--relaxed-l2", type=float, default=1.0)
    parser.add_argument("--relaxed-iters", type=int, default=60)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backward-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
