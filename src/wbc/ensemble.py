"""Prespecified probability ensembles with strict paired-case alignment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import validate_predictions, evaluate_frame
from .utils import sha256, write_json


def average_probabilities(frames, classes, weights=None):
    if len(frames) < 2:
        raise ValueError("An ensemble requires at least two prediction tables")
    weights = np.ones(len(frames)) if weights is None else np.asarray(weights, float)
    if (
        weights.shape != (len(frames),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Require one finite nonnegative weight per member, with positive sum")
    weights = weights / weights.sum()
    columns = [f"prob_{i}" for i in range(len(classes))]
    aligned = []
    for frame in frames:
        validate_predictions(frame, classes)
        if not set(columns) <= set(frame):
            raise ValueError("Every member needs all class probabilities")
        if {c for c in frame if c.startswith("prob_")} != set(columns):
            raise ValueError("Probability column count differs from the declared class space")
        aligned.append(frame.sort_values(["seed", "id"]).reset_index(drop=True))
    metadata = ["id", "seed", "group", "dataset", "y_true"]
    metadata += [key for key in ["split", "outer_fold"] if any(key in f for f in aligned)]
    reference = aligned[0]
    for frame in aligned[1:]:
        if not set(metadata) <= set(frame) or not set(metadata) <= set(reference):
            raise ValueError("Ensemble members disagree on partition/fold metadata")
        if not reference[metadata].equals(frame[metadata]):
            raise ValueError("Ensemble IDs, seeds, labels, groups, domains or partitions differ")
    result = reference[metadata].copy()
    probabilities = sum(weight * frame[columns].to_numpy(float) for weight, frame in zip(weights, aligned))
    result[columns] = probabilities
    result["prediction"] = probabilities.argmax(1)
    validate_predictions(result, classes)
    return result


def ensemble_runs(run_dirs, split, out, weights=None):
    """Read class-order provenance; caller must specify weights before testing."""
    if split not in ["validation", "test"]:
        raise ValueError("split must be validation or test")
    configs, frames, sources = [], [], []
    for directory in run_dirs:
        root = Path(directory)
        config = json.loads((root / "provenance.json").read_text(encoding="utf-8"))["config"]
        configs.append(config)
        path = root / f"{split}_predictions.csv"
        frames.append(pd.read_csv(path, dtype={"id": str, "group": str}))
        sources.append({"run": str(root), "predictions_sha256": sha256(path)})
    if len(configs) < 2 or any(c["classes"] != configs[0]["classes"] for c in configs):
        raise ValueError("Require >=2 runs with exactly matching class order")
    frame = average_probabilities(frames, configs[0]["classes"], weights)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Use an empty directory for ensemble artifacts")
    frame.to_csv(out / "predictions.csv", index=False)
    write_json(out / "metrics.json", evaluate_frame(frame, configs[0]["classes"]))
    raw_weights = np.ones(len(frames)) if weights is None else np.asarray(weights, float)
    write_json(
        out / "provenance.json",
        {
            "sources": sources,
            "classes": configs[0]["classes"],
            "split": split,
            "weights": raw_weights / raw_weights.sum(),
            "note": "Fixed probability average. Weights are not fitted by this command. Prespecify them or choose using validation only.",
        },
    )
    return frame
