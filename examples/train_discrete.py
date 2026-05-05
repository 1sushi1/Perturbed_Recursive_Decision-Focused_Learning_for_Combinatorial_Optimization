from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

from rdfl import FeedbackMLP, PerturbedBipartiteMatchingLayer, PerturbedTopKLayer, RDFLImplicit, RDFLUnrolled


def make_synthetic_costs(n_samples: int, feature_dim: int, decision_dim: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, feature_dim, generator=generator)
    weights = torch.randn(feature_dim, decision_dim, generator=generator) / feature_dim**0.5
    costs = features @ weights + 0.15 * torch.randn(n_samples, decision_dim, generator=generator)
    return TensorDataset(features, costs)


def build_discrete_layer(args: argparse.Namespace):
    if args.problem == "topk":
        return args.decision_dim, PerturbedTopKLayer(
            decision_dim=args.decision_dim,
            k=args.k,
            num_samples=args.perturbations,
            sigma=args.sigma,
        )
    if args.problem == "matching":
        decision_dim = args.side * args.side
        return decision_dim, PerturbedBipartiteMatchingLayer(
            side=args.side,
            num_samples=args.perturbations,
            sigma=args.sigma,
        )
    raise ValueError(f"unknown problem: {args.problem}")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    feature_dim = 8
    decision_dim, opt_layer = build_discrete_layer(args)
    dataset = make_synthetic_costs(args.samples, feature_dim, decision_dim, args.seed)
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
            # The perturbed layers are argmax layers, while costs are minimized.
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
    parser.add_argument("--problem", choices=["topk", "matching"], default="topk")
    parser.add_argument("--model", choices=["unrolled", "implicit"], default="unrolled")
    parser.add_argument("--decision-dim", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--side", type=int, default=4)
    parser.add_argument("--perturbations", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=1.0)
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
