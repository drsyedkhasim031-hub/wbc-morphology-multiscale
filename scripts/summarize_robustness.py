"""Summarize materialized corruption evaluations against matched clean predictions.

Input catalog CSV: kind,severity,variant,predictions (path to prediction CSV).
Paired stain/gamma variants must both be present and are averaged equally.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wbc.cli import get_classes
from wbc.metrics import classification
from wbc.statistics import align


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    clean, catalog = pd.read_csv(args.clean), pd.read_csv(args.catalog)
    classes = get_classes(args.classes)
    rows = []
    for row in catalog.itertuples():
        corrupted = pd.read_csv(row.predictions)
        base, _, _ = align(clean, corrupted, classes)
        for seed, part in corrupted.groupby("seed"):
            clean_part = clean[clean.seed == seed]
            clean_acc = classification(clean_part.y_true, clean_part.prediction, classes)["accuracy"]
            acc = classification(part.y_true, part.prediction, classes)["accuracy"]
            rows.append(
                {
                    "kind": row.kind,
                    "severity": row.severity,
                    "variant": row.variant,
                    "seed": seed,
                    "accuracy": acc,
                    "clean_accuracy": clean_acc,
                    "delta_pp": (acc - clean_acc) * 100,
                }
            )
    frame = pd.DataFrame(rows)
    for (kind, severity, seed), group in frame.groupby(["kind", "severity", "seed"]):
        expected = {0, 1} if kind in ("stain", "gamma") else {0}
        if set(group.variant) != expected or len(group) != len(expected):
            raise ValueError(f"Missing or duplicate variants: {kind}, severity {severity}, seed {seed}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grouped = (
        frame.groupby(["kind", "severity", "seed"])[["accuracy", "clean_accuracy", "delta_pp"]]
        .mean()
        .reset_index()
    )
    grouped.to_csv(out / "robustness.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    for kind, group in grouped.groupby("kind"):
        curve = group.groupby("severity").delta_pp.mean()
        ax.plot([0] + curve.index.tolist(), [0] + curve.tolist(), "o-", label=kind)
    ax.set(
        xlabel="Severity",
        ylabel="Accuracy change from clean (percentage points)",
        title="Executed corruption evaluations",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
