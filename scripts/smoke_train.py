"""End-to-end proposed-path integration run with a deliberately reduced encoder.

Real BloodMNIST images, 6 classes, 120 cells, 64-pixel crops, 2 epochs.
No full-model accuracy inference is supported by this debug run.
"""

import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from wbc.engine import train
from wbc.labels import NATIVE
from wbc.utils import write_json
from demo_baseline import load_data


def prepare_images(data_path, work):
    """Materialize the fixed 72/24/24-cell integration subset for both demo scripts."""
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    data, _ = load_data(data_path)
    rows = []
    rng = np.random.Generator(np.random.PCG64(1729))
    for split, count in [("train", 12), ("val", 4), ("test", 4)]:
        d = data[split]
        for label in range(6):
            chosen = rng.choice(np.flatnonzero(d["y"] == label), count, replace=False)
            for index in chosen:
                name = f"{split}_{d['indices'][index]}.png"
                Image.fromarray(d["images"][index]).save(work / name)
                rows.append(
                    {
                        "id": name,
                        "path": name,
                        "dataset": "pbc",
                        "label": NATIVE["pbc"][label],
                        "original_label": NATIVE["pbc"][label],
                        "split": split,
                        "group": d["hashes"][index],
                        "exact_hash": d["hashes"][index],
                        "maturation": np.nan,
                    }
                )
    manifest = work / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--results", required=True)
    args = p.parse_args()
    work, results = Path(args.work).resolve(), Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    manifest = prepare_images(args.data, work)
    config = {
        "manifest": str(manifest),
        "roots": {"pbc": str(work)},
        "classes": NATIVE["pbc"],
        "seed": 1729,
        "device": "cpu",
        "threads": 4,
        "image_size": 64,
        "model": {"backbone": "debug", "pretrained": False, "factors": [0.8, 1.0, 1.2]},
        "training": {"epochs": 2, "warmup_epochs": 0, "batch_size": 4, "workers": 0},
    }
    run = train(config, work / "run")
    for name in [
        "history.csv",
        "test_predictions.csv",
        "test_metrics.json",
        "preprocessing.json",
        "run_summary.json",
        "manifest.csv",
    ]:
        shutil.copyfile(run / name, results / name)
    write_json(
        results / "scope.json",
        {
            "status": "EXECUTED INTEGRATION CHECK",
            "backbone": "debug (16/32/64/128 channels, one block per stage)",
            "input_size": 64,
            "epochs": 2,
            "train": 72,
            "validation": 24,
            "test": 24,
            "source": "BloodMNIST 28x28 enlarged to 64",
            "limitation": "Not a full ConvNeXt-T model experiment; accuracy has no manuscript reproduction interpretation.",
        },
    )


if __name__ == "__main__":
    main()
