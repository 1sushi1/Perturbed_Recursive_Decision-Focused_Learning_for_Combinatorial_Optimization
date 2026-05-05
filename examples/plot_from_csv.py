from __future__ import annotations

import argparse
import csv
import pathlib


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot(rows: list[dict[str, str]], metric: str, output: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    models = []
    for row in rows:
        name = row["model"]
        if name not in models:
            models.append(name)

    output.parent.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(output, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--metric", default="decision_rmse")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = pathlib.Path(args.csv_path)
    output = pathlib.Path(args.output) if args.output else csv_path.with_name(f"{csv_path.stem}_{args.metric}.png")
    rows = read_rows(csv_path)
    plot(rows, args.metric, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
