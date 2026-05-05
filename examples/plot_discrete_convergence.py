from __future__ import annotations

import argparse
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_discrete_benchmark import (
    build_model,
    build_spec,
    discrete_regret,
    evaluate,
    make_synthetic_costs,
    split_dataset,
    spo_plus_loss,
)


def train_epoch(name: str, model, spec, train_loader, optimizer) -> float:
    model.train()
    total = 0.0
    count = 0
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
        total += loss.item() * features.shape[0]
        count += features.shape[0]
    return total / count


def write_csv(rows: list[dict[str, float | int | str]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "model", "train_loss", "decision_rmse", "regret"])
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | int | str]], metric: str, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    models = []
    for row in rows:
        name = str(row["model"])
        if name not in models:
            models.append(name)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name in models:
        subset = [row for row in rows if row["model"] == name]
        epochs = [int(row["epoch"]) for row in subset]
        values = [float(row[metric]) for row in subset]
        plt.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label=name)
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " "))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


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

    torch.manual_seed(args.seed)
    spec = build_spec(args)
    dataset = make_synthetic_costs(args.samples, spec.feature_dim, spec.decision_dim, args.seed)
    train_set, _, test_set = split_dataset(dataset, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    rows: list[dict[str, float | int | str]] = []
    for name in model_names:
        torch.manual_seed(args.seed)
        model_spec = build_spec(args)
        model = build_model(name, model_spec, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(name, model, model_spec, train_loader, optimizer)
            metrics = evaluate(model, model_spec, test_loader)
            rows.append(
                {
                    "epoch": epoch,
                    "model": name,
                    "train_loss": train_loss,
                    "decision_rmse": metrics["decision_rmse"],
                    "regret": metrics["regret"],
                }
            )
            print(
                f"model={name} epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"decision_rmse={metrics['decision_rmse']:.6f} "
                f"regret={metrics['regret']:.6f}"
            )

    output_dir = pathlib.Path(args.output_dir)
    stem = f"discrete_convergence_{args.problem}"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}_{args.metric}.png"
    write_csv(rows, csv_path)
    plot(rows, args.metric, png_path)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=512)
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
    parser.add_argument("--metric", choices=["decision_rmse", "regret"], default="decision_rmse")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
