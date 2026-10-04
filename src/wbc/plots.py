"""Regenerate evaluation plots from case-level predictions, never invented values."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import classification, calibration


def plot_predictions(frame, classes, out, title="Executed evaluation"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if frame.seed.nunique() != 1:
        raise ValueError("Plot one seed at a time; do not count seed replicates as independent cells")
    p = frame[[f"prob_{i}" for i in range(len(classes))]].to_numpy()
    metrics = classification(frame.y_true, frame.prediction, classes)
    reliability = calibration(frame.y_true, p)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = np.array(metrics["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="Reference label",
        title=title + "\nConfusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    bins = [b for b in reliability["bins"] if b["n"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(
        [b["confidence"] for b in bins],
        [b["accuracy"] for b in bins],
        "o-",
        color="#096c8a",
        label="Observed",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean confidence",
        ylabel="Fraction correct",
        title=f"{title}\nECE = {reliability['ece']:.4f}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(reliability["coverage"], reliability["risk"], color="#096c8a")
    ax.set(
        xlabel="Coverage", ylabel="Empirical error rate", title=title + "\nRetrospective risk–coverage curve"
    )
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=180)
    plt.close(fig)


def plot_history(csv, output):
    data = pd.read_csv(csv)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(data.epoch, data.train_loss)
    ax[0].set(xlabel="Epoch", ylabel="Training objective")
    ax[1].plot(data.epoch, data.validation_macro_f1)
    ax[1].set(xlabel="Epoch", ylabel="Validation macro-F1")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
