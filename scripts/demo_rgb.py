"""Two-epoch RGB integration runs on real, tiny BloodMNIST subsets; not benchmarks."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from wbc.engine import train
from wbc.ensemble import ensemble_runs
from wbc.experiments import summarize_runs
from wbc.labels import NATIVE
from wbc.plots import plot_predictions, plot_history
from wbc.utils import sha256, write_json
import pandas as pd

from smoke_train import prepare_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    work, results = Path(args.work).resolve(), Path(args.results)
    if results.exists() and any(results.iterdir()):
        raise FileExistsError("Use a new results directory")
    manifest = prepare_images(args.data, work / "images")
    classes = NATIVE["pbc"]
    config = {
        "manifest": str(manifest),
        "roots": {"pbc": str(manifest.parent)},
        "classes": classes,
        "seed": 1729,
        "device": "cpu",
        "threads": 4,
        "image_size": 64,
        "training": {"epochs": 2, "warmup_epochs": 0, "batch_size": 8, "workers": 0},
    }
    runs = []
    for name in ["compact_cnn", "resnet18"]:
        config["model"] = {"architecture": "rgb", "backbone": name, "pretrained": False, "factors": [1.0]}
        run = train(deepcopy(config), work / name)
        runs.append(run)
        destination = results / name
        destination.mkdir(parents=True)
        for filename in [
            "history.csv",
            "manifest.csv",
            "test_predictions.csv",
            "validation_predictions.csv",
            "test_metrics.json",
            "validation_metrics.json",
            "calibration.json",
            "run_summary.json",
            "preprocessing.json",
        ]:
            shutil.copyfile(run / filename, destination / filename)
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        provenance["config"]["manifest"] = "<demo-work>/images/manifest.csv"
        provenance["config"]["roots"]["pbc"] = "<demo-work>/images"
        provenance["path_note"] = (
            "Local machine paths replaced with <demo-work>; hashes and executed hyperparameters unchanged."
        )
        write_json(destination / "provenance.json", provenance)
        plot_predictions(
            pd.read_csv(run / "test_predictions.csv"),
            classes,
            destination / "figures",
            f"{name}: 24-cell integration check",
        )
        plot_history(run / "history.csv", destination / "figures" / "training_history.png")
    summarize_runs([results / "compact_cnn", results / "resnet18"], results / "comparison.csv")
    ensemble_runs(runs, "test", results / "ensemble", weights=[1, 1])
    # Replace machine-specific run locations in the public ensemble provenance too.
    path = results / "ensemble" / "provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    for source, name in zip(provenance["sources"], ["compact_cnn", "resnet18"]):
        source["run"] = name
    write_json(path, provenance)
    write_json(
        results / "scope.json",
        {
            "status": "EXECUTED SOFTWARE INTEGRATION CHECK; NOT A BENCHMARK",
            "source": "BloodMNIST 28x28 images enlarged to 64; six WBC classes",
            "data_sha256": sha256(args.data),
            "train": 72,
            "validation": 24,
            "test": 24,
            "epochs": 2,
            "seed": 1729,
            "pretrained": False,
            "models": ["compact_cnn", "resnet18"],
            "ensemble_weights_prespecified": [1, 1],
            "limitation": "Tiny subset, short training, no patient-independent claim and no manuscript score reproduction.",
        },
    )


if __name__ == "__main__":
    main()
