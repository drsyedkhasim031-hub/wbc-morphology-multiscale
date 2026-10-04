"""Actual CPU experiment on six WBC classes in 28x28 BloodMNIST.

This is a logistic-regression demonstration, NOT the proposed model or paper split.
Fits scaling/model on train and selects C/calibration only on validation.
"""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wbc.data import pixel_hash
from wbc.labels import NATIVE
from wbc.metrics import evaluate_frame, fit_temperature, classification
from wbc.plots import plot_predictions
from wbc.statistics import paired_bootstrap
from wbc.utils import environment, sha256, write_json

MAP = {0: 0, 1: 1, 4: 2, 5: 3, 6: 4, 3: 5}


def features(images):
    x = images.astype(np.float32) / 255
    # 7x7 block means preserve coarse morphology; histogram describes stain distribution.
    spatial = x.reshape(len(x), 7, 4, 7, 4, 3).mean((2, 4)).reshape(len(x), -1)
    hist = np.stack(
        [
            np.concatenate(
                [
                    np.histogram(im[:, :, c], bins=16, range=(0, 1), density=False)[0] / (28 * 28)
                    for c in range(3)
                ]
            )
            for im in x
        ]
    )
    return np.concatenate([spatial, hist, x.mean((1, 2)), x.std((1, 2))], axis=1)


def load_data(path):
    arrays = np.load(path, allow_pickle=False)
    data = {}
    for split in ["train", "val", "test"]:
        y0 = arrays[f"{split}_labels"].ravel()
        indices = np.flatnonzero(np.isin(y0, list(MAP)))
        images = arrays[f"{split}_images"][indices]
        y = np.array([MAP[int(v)] for v in y0[indices]])
        hashes = [pixel_hash(Image.fromarray(im)) for im in images]
        data[split] = {"images": images, "y": y, "indices": indices, "hashes": np.array(hashes)}
    # Test takes precedence; validation takes precedence over fitting.
    excluded = {}
    for split, forbidden in [
        ("val", set(data["test"]["hashes"])),
        ("train", set(data["test"]["hashes"]) | set(data["val"]["hashes"])),
    ]:
        keep = np.array([h not in forbidden for h in data[split]["hashes"]])
        excluded[split] = int((~keep).sum())
        data[split] = {k: v[keep] for k, v in data[split].items()}
    return data, excluded


def prediction_frame(data, logits, temperature=1.0, split="test"):
    p = softmax(logits / temperature, axis=1)
    frame = pd.DataFrame(
        {
            "id": [f"bloodmnist:{split}:{i}" for i in data["indices"]],
            "group": data["hashes"],
            "dataset": "bloodmnist_demo",
            "seed": 1729,
            "split": split,
            "y_true": data["y"],
            "prediction": p.argmax(1),
            "temperature": temperature,
        }
    )
    for c in range(6):
        frame[f"prob_{c}"], frame[f"logit_{c}"] = p[:, c], logits[:, c]
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    data, excluded = load_data(args.data)
    x = {split: features(d["images"]) for split, d in data.items()}
    best, chosen, history = -1, None, []
    with threadpool_limits(limits=4):
        for c in [0.01, 0.1, 1.0]:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=1729))
            model.fit(x["train"], data["train"]["y"])
            score = classification(data["val"]["y"], model.predict(x["val"]), NATIVE["pbc"])["macro_f1"]
            history.append({"C": c, "validation_macro_f1": score, "iterations": model[-1].n_iter_.tolist()})
            print(history[-1], flush=True)
            if score > best:
                chosen, best = model, score
        temperature = fit_temperature(chosen.decision_function(x["val"]), data["val"]["y"])
        test_logits = chosen.decision_function(x["test"])
    predictions = prediction_frame(data["test"], test_logits, temperature)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    raw = prediction_frame(data["test"], test_logits)
    write_json(out / "raw_metrics.json", evaluate_frame(raw, NATIVE["pbc"]))
    write_json(out / "metrics.json", evaluate_frame(predictions, NATIVE["pbc"]))
    val = prediction_frame(data["val"], chosen.decision_function(x["val"]), temperature, "val")
    val.to_csv(out / "validation_predictions.csv", index=False)
    majority = np.bincount(data["train"]["y"], minlength=6).argmax()
    comparator = predictions.copy()
    comparator["prediction"] = majority
    for i in range(6):
        comparator[f"prob_{i}"] = float(i == majority)
        comparator[f"logit_{i}"] = 0.0 if i == majority else -1e10
    comparator.to_csv(out / "majority_predictions.csv", index=False)
    write_json(
        out / "paired_bootstrap_vs_majority.json",
        paired_bootstrap(predictions, comparator, NATIVE["pbc"], draws=2000),
    )
    plot_predictions(
        predictions, NATIVE["pbc"], out / "figures", "BloodMNIST 28x28 CPU baseline (six WBC classes)"
    )
    scaler, lr = chosen[0], chosen[-1]
    # Portable numeric coefficients avoid executable pickles.
    write_json(
        out / "model_coefficients.json",
        {
            "standardizer_mean": scaler.mean_,
            "standardizer_scale": scaler.scale_,
            "coefficients": lr.coef_,
            "intercept": lr.intercept_,
            "classes": lr.classes_,
            "temperature": temperature,
            "features": "7x7 RGB block means, 16-bin per-channel histograms, RGB mean and population std",
        },
    )
    write_json(
        out / "provenance.json",
        {
            "status": "EXECUTED CPU DEMONSTRATION, NOT A PAPER REPRODUCTION",
            "dataset": "BloodMNIST 28x28, six selected WBC classes",
            "source_url": "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1",
            "source_sha256": sha256(args.data),
            "source_md5": hashlib.md5(Path(args.data).read_bytes()).hexdigest(),
            "classes": NATIVE["pbc"],
            "split_counts": {s: len(d["y"]) for s, d in data.items()},
            "cross_split_exact_duplicates_excluded": excluded,
            "selection": history,
            "temperature": temperature,
            "environment": environment(),
            "elapsed_seconds": time.perf_counter() - start,
            "limits": "Official BloodMNIST split. Patient/slide independence and near-duplicate independence unverified. One deterministic classical-model run. Not evidence for proposed model or original-resolution PBC.",
        },
    )
    print("Demo completed", out, flush=True)


if __name__ == "__main__":
    main()
