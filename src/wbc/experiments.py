"""Baseline configuration planning and auditable run comparisons."""

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import yaml

from .models import MODEL_CATALOG
from .utils import write_json


def plan_baselines(base, backbones, seeds, out, pretrained=True, freeze_backbone=False):
    """Reuse the supplied task's split, labels and training schedule for every model."""
    if not backbones or not seeds or len(set(backbones)) != len(backbones) or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty, unique backbones and seeds")
    unknown = set(backbones) - MODEL_CATALOG.keys()
    if unknown:
        raise ValueError(f"Unknown backbones: {sorted(unknown)}")
    out = Path(out)
    plans = []
    for backbone in backbones:
        for seed in seeds:
            name = f"{backbone}_seed{seed}"
            config = deepcopy(base)
            config["seed"] = int(seed)
            config["model"] = {
                "architecture": "rgb",
                "backbone": backbone,
                "factors": [1.0],
                "pretrained": pretrained and backbone != "compact_cnn",
                "freeze_backbone": freeze_backbone,
                "dropout": 0.2,
                "weights_version": "IMAGENET1K_V1",
            }
            if freeze_backbone and not config["model"]["pretrained"]:
                raise ValueError("Baseline planner refuses a frozen randomly initialized encoder")
            config_path = out / "configs" / f"{name}.yaml"
            if config_path.exists():
                raise FileExistsError(f"Refusing to replace an existing plan: {config_path}")
            plans.append({"config": config, "config_path": config_path, "out": out / name})
    # Validate every request before writing any plan.
    for plan in plans:
        plan["config_path"].parent.mkdir(parents=True, exist_ok=True)
        plan["config_path"].write_text(yaml.safe_dump(plan["config"], sort_keys=False), encoding="utf-8")
    write_json(
        out / "plan.json",
        {
            "status": "planned; no experiments executed by planning",
            "scope": "standalone RGB extensions; not manuscript result reproduction",
            "runs": [{"config": str(p["config_path"]), "out": str(p["out"])} for p in plans],
        },
    )
    return plans


def summarize_runs(run_dirs, out):
    """Report validation and test results without ranking or selecting on test."""
    rows = []
    for directory in run_dirs:
        root = Path(directory)
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        config = provenance["config"]
        summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
        for split in ["validation", "test"]:
            path = root / f"{split}_metrics.json"
            if not path.exists():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics["classes"] != config["classes"]:
                raise ValueError(f"Class order differs between provenance and metrics: {root}")
            for seed, result in metrics["runs"].items():
                rows.append(
                    {
                        "run": str(root),
                        "architecture": config["model"].get("architecture", "proposed"),
                        "backbone": config["model"]["backbone"],
                        "seed": int(seed),
                        "split": split,
                        "selected_epoch": summary["selected_epoch"],
                        "n": result["n"],
                        "classes": json.dumps(config["classes"]),
                        "manifest_sha256": provenance["manifest_sha256"],
                        "checkpoint_sha256": summary["checkpoint_sha256"],
                        "parameters_total": provenance.get("parameters_total"),
                        "parameters_trainable": provenance.get("parameters_trainable"),
                        **{
                            key: result[key]
                            for key in ["accuracy", "macro_f1", "balanced_accuracy", "equal_domain_macro_f1"]
                        },
                        **{
                            key: result.get("calibration", {}).get(key)
                            for key in ["nll", "brier", "ece", "aurc"]
                        },
                    }
                )
    if not rows:
        raise ValueError("No completed run metrics found")
    frame = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame
