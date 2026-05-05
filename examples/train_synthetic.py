from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

from rdfl import BipartiteMatchingLayer, FeedbackMLP, NewsvendorLayer, RDFLImplicit, RDFLUnrolled


def make_synthetic_data(
    n_samples: int,
    feature_dim: int,
    decision_dim: int,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, feature_dim, generator=generator)
    weights = torch.randn(feature_dim, decision_dim, generator=generator) / feature_dim**0.5
    true_costs = features @ weights + 0.15 * torch.randn(n_samples, decision_dim, generator=generator)
    return TensorDataset(features, true_costs)


def build_problem(name: str):
    if name == "newsvendor":
        decision_dim = 10
        layer = NewsvendorLayer(
            decision_dim=decision_dim,
            sum_lower=3.0,
            sum_upper=6.0,
            lower=0.0,
            upper=1.0,
            l2_strength=1.0,
        )
        return decision_dim, layer
    if name == "matching":
        side = 4
        layer = BipartiteMatchingLayer(
            num_left=side,
            num_right=side,
            min_matches=2.0,
            l2_strength=0.1,
            step_size=0.2,
        )
        return side * side, layer
    raise ValueError(f"unknown problem: {name}")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    feature_dim = 8
    decision_dim, opt_layer = build_problem(args.problem)
    dataset = make_synthetic_data(args.samples, feature_dim, decision_dim, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    predictor = FeedbackMLP(feature_dim, decision_dim, hidden_dim=32, dropout=0.1)
    if args.model == "unrolled":
        model = RDFLUnrolled(predictor, opt_layer, unroll_steps=args.steps)
    elif args.model == "implicit":
        model = RDFLImplicit(predictor, opt_layer, rootfind_steps=args.steps, backward_steps=args.backward_steps)
    else:
        raise ValueError(f"unknown model: {args.model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for features, true_costs in loader:
            decisions = model(features)
            regret = opt_layer.regret(decisions, true_costs)
            loss = regret.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * features.shape[0]
        print(f"epoch={epoch:03d} regret={total / len(dataset):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["newsvendor", "matching"], default="newsvendor")
    parser.add_argument("--model", choices=["unrolled", "implicit"], default="unrolled")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backward-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
