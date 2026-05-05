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

from run_benchmark import (
    build_model,
    build_problem,
    evaluate,
    make_recursive_synthetic_data,
    make_synthetic_data,
    split_dataset,
)


def train_epoch(model_name: str, model, opt_layer, train_loader, optimizer) -> float:
    model.train()
    total = 0.0
    count = 0
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
        total += loss.item() * features.shape[0]
        count += features.shape[0]
    return total / count


def write_csv(rows: list[dict[str, float | int | str]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "model", "train_loss", "test_regret", "decision_rmse", "cost_rmse"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_png(rows: list[dict[str, float | int | str]], path: pathlib.Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    models = []
    for row in rows:
        name = str(row["model"])
        if name not in models:
            models.append(name)

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
    torch.manual_seed(args.seed)
    spec = build_problem(args.problem, args.scale)
    if args.data_mode == "recursive":
        dataset = make_recursive_synthetic_data(
            args.samples,
            spec.feature_dim,
            spec.decision_dim,
            spec.optimizer_layer,
            args.seed,
            feedback_strength=args.feedback_strength,
        )
    else:
        dataset = make_synthetic_data(args.samples, spec.feature_dim, spec.decision_dim, args.seed)
    train_set, _, test_set = split_dataset(dataset, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model_names = ["pto", "sdfl", "rdfl-u", "rdfl-i"] if args.models == ["all"] else args.models
    rows: list[dict[str, float | int | str]] = []

    for model_name in model_names:
        torch.manual_seed(args.seed)
        model_spec = build_problem(args.problem, args.scale)
        model = build_model(model_name, model_spec, args.steps, args.backward_steps)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model_name, model, model_spec.optimizer_layer, train_loader, optimizer)
            metrics = evaluate(model, model_spec.optimizer_layer, test_loader)
            row = {
                "epoch": epoch,
                "model": model_name,
                "train_loss": train_loss,
                "test_regret": metrics["regret"],
                "decision_rmse": metrics["decision_rmse"],
                "cost_rmse": metrics["cost_rmse"],
            }
            rows.append(row)
            print(
                f"model={model_name} epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} test_regret={metrics['regret']:.6f} "
                f"decision_rmse={metrics['decision_rmse']:.6f}"
            )

    output_dir = pathlib.Path(args.output_dir)
    stem = f"convergence_{args.problem}_{args.scale}"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}_{args.metric}.png"
    write_csv(rows, csv_path)
    plot_png(rows, png_path, args.metric)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["newsvendor", "matching"], default="newsvendor")
    parser.add_argument("--scale", choices=["small", "mid", "large"], default="small")
    parser.add_argument("--models", nargs="+", choices=["all", "pto", "sdfl", "rdfl-u", "rdfl-i"], default=["all"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--data-mode", choices=["one_way", "recursive"], default="one_way")
    parser.add_argument("--feedback-strength", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backward-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--metric", choices=["test_regret", "decision_rmse", "cost_rmse"], default="test_regret")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
