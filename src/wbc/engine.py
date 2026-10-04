"""Training, source-only validation selection, calibration and frozen evaluation."""

import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.special import softmax
import yaml

from .data import Cells, assert_no_leakage
from .losses import effective_weights, objective
from .metrics import classification, fit_temperature, evaluate_frame
from .models import build_model
from .checkpoints import atomic_save, cpu_state, capture_rng, restore_rng
from .preprocessing import fit_from_cells, fit_descriptors
from .utils import environment, seed_everything, sha256, write_json, clean_json


def read_config(path):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if "extends" in config:
        parent = read_config(Path(path).parent / config.pop("extends"))
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(parent.get(key), dict):
                parent[key].update(value)
            else:
                parent[key] = value
        config = parent
    return config


def seed_worker(worker_id):
    value = torch.initial_seed() % 2**32
    random.seed(value)
    np.random.seed(value)


def make_loader(cells, config, shuffle=False):
    return DataLoader(
        cells,
        batch_size=config.get("batch_size", 24),
        shuffle=shuffle,
        num_workers=config.get("workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.get("seed", 1729)),
    )


def dataset_for(frame, config, augment=False):
    return Cells(
        frame,
        config["roots"],
        config["classes"],
        config["model"].get(
            "factors", [1.0] if config["model"].get("architecture") == "rgb" else [0.8, 1.0, 1.2]
        ),
        config.get("image_size", 224),
        augment,
    )


def selection_score(frame, classes, lodo=False):
    if lodo:
        return float(
            np.mean(
                [
                    classification(p.y_true, p.prediction, classes)["macro_f1"]
                    for _, p in frame.groupby("dataset")
                ]
            )
        )
    return float(classification(frame.y_true, frame.prediction, classes)["macro_f1"])


@torch.no_grad()
def predict(model, loader, device, seed, temperature=1.0, split="test"):
    model.eval()
    rows = []
    for batch in loader:
        logits = (
            model(*[batch[k].to(device) for k in ["crops", "validity", "extents"]])["logits"].cpu().numpy()
        )
        probabilities = softmax(logits / temperature, axis=1)
        for j, index in enumerate(batch["index"].tolist()):
            record = loader.dataset.frame.iloc[index]
            row = {key: record[key] for key in ["id", "group", "dataset"]}
            row.update(
                {
                    "y_true": int(batch["label"][j]),
                    "prediction": int(probabilities[j].argmax()),
                    "seed": seed,
                    "split": split,
                    "temperature": temperature,
                }
            )
            if "outer_fold" in record:
                row["outer_fold"] = int(record.outer_fold)
            row.update({f"logit_{k}": float(v) for k, v in enumerate(logits[j])})
            row.update({f"prob_{k}": float(v) for k, v in enumerate(probabilities[j])})
            rows.append(row)
    if not rows:
        raise ValueError(f"No {split} examples")
    return pd.DataFrame(rows)


def parameter_groups(model, config):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or any(key in name for key in ["prototypes", "cls", "scale_embeddings"])
        peak = (
            config.get("backbone_lr", 1e-4) if name.startswith("backbone.") else config.get("head_lr", 3e-4)
        )
        decay = 0.0 if no_decay else config.get("weight_decay", 0.05)
        groups.setdefault((peak, decay), []).append(p)
    return [
        {"params": parameters, "lr": peak, "peak_lr": peak, "weight_decay": decay}
        for (peak, decay), parameters in groups.items()
    ]


def update_lr(optimizer, step, warmup, total, minimum=1e-6):
    for group in optimizer.param_groups:
        if step <= warmup and warmup:
            value = step / warmup * group["peak_lr"]
        else:
            progress = (step - warmup) / max(1, total - warmup)
            value = minimum + 0.5 * (group["peak_lr"] - minimum) * (1 + math.cos(math.pi * progress))
        group["lr"] = value


def train(config, out, resume=None, stop_after_epoch=None):
    """Train and finalize, or stop at an epoch boundary without accessing test data.

    Resume restores optimizer, RNGs and loader generators, keeping the originally
    configured schedule. Changing epochs, data, or hyperparameters is a new run.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if resume is None and any((out / name).exists() for name in ["best.pt", "resume.pt", "history.csv"]):
        raise FileExistsError("Use a fresh output directory to preserve prior run provenance")
    resumed = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if resumed:
        if resumed.get("format_version") != 1 or resumed.get("config") != config:
            raise ValueError("Resume requires the original complete configuration and a resume.pt checkpoint")
        if resumed["manifest_sha256"] != sha256(config["manifest"]):
            raise ValueError("Manifest changed since the saved checkpoint")
        if Path(resume).resolve().parent != out.resolve() and any(out.iterdir()):
            raise FileExistsError("Resume into its original directory or a new empty directory")
    seed = config.get("seed", 1729)
    seed_everything(seed)
    torch.set_num_threads(config.get("threads", 4))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if resumed and (resumed["device"] != str(device) or resumed["torch_version"] != str(torch.__version__)):
        raise ValueError(
            "Resume requires the same device and PyTorch version; use evaluate for portable inference"
        )
    manifest = pd.read_csv(config["manifest"], dtype={"id": str, "group": str})
    assert_no_leakage(manifest)
    target = config.get("target_dataset")
    if target and ((manifest.dataset == target) & manifest.split.isin(["train", "val"])).any():
        raise ValueError("Target dataset leaked into fitting or model selection")
    frames = {
        s: manifest[manifest.split == s].sort_values("id").reset_index(drop=True)
        for s in ["train", "val", "test"]
    }
    if not all(len(frames[s]) for s in ["train", "val"]):
        raise ValueError("Nonempty train and validation partitions are required")
    train_cfg = {**config.get("training", {}), "seed": seed}
    epochs = train_cfg.get("epochs", 80)
    if epochs < 1 or not 0 <= train_cfg.get("warmup_epochs", 5) < epochs:
        raise ValueError("Require epochs>=1 and 0<=warmup_epochs<epochs")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise ValueError("stop_after_epoch must be within the configured training schedule")
    start_epoch = resumed["epoch"] + 1 if resumed else 1
    if stop_after_epoch is not None and stop_after_epoch < start_epoch:
        raise ValueError("stop_after_epoch must be later than the saved epoch")
    clip_norm = train_cfg.get("clip_grad_norm")
    if clip_norm is not None and (not math.isfinite(clip_norm) or clip_norm <= 0):
        raise ValueError("clip_grad_norm must be positive and finite")
    model = build_model(config, initialize=not bool(resumed)).to(device)
    if resumed:
        model.load_state_dict(resumed["state_dict"], strict=True)
    fitting = dataset_for(frames["train"], config)
    preprocessing = resumed["preprocessing"] if resumed else {}
    if not resumed and model.use_stain and not model.rgb_only:
        canonical = Cells(
            frames["train"], config["roots"], config["classes"], [1.0], config.get("image_size", 224)
        )
        basis, preprocessing["stain_fit"] = fit_from_cells(canonical, out / "cache")
        model.basis.copy_(torch.from_numpy(basis).to(device))
    if not resumed and model.use_morphology and not model.rgb_only:
        preprocessing["descriptors"] = fit_descriptors(model, make_loader(fitting, train_cfg), device)
    if hasattr(model, "basis"):
        preprocessing["basis"] = model.basis.detach().cpu().tolist()
    else:
        preprocessing["normalization"] = (
            "fixed ImageNet RGB mean/std; no fitted stain or descriptor statistics"
        )
    # NumPy fitting metadata must remain readable with torch.load(weights_only=True).
    preprocessing = clean_json(preprocessing)
    write_json(out / "preprocessing.json", preprocessing)
    counts = [int((frames["train"].label == c).sum()) for c in config["classes"]]
    weights = effective_weights(counts, train_cfg.get("beta", 0.999)).to(device)
    train_loader = make_loader(dataset_for(frames["train"], config, True), train_cfg, True)
    val_loader = make_loader(dataset_for(frames["val"], config), train_cfg)
    optimizer = torch.optim.AdamW(parameter_groups(model, train_cfg), betas=(0.9, 0.999), eps=1e-8)
    steps, warmup = epochs * len(train_loader), train_cfg.get("warmup_epochs", 5) * len(train_loader)
    provenance = {
        "config": config,
        "environment": environment(),
        "manifest_sha256": sha256(config["manifest"]),
        "fit_ids": frames["train"].id.tolist(),
        "validation_ids": frames["val"].id.tolist(),
        "class_counts": dict(zip(config["classes"], counts)),
        "class_weights": weights.cpu().tolist(),
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "status": "executed run; scope determined by config; not manuscript result verification",
    }
    write_json(out / "provenance.json", provenance)
    manifest.to_csv(out / "manifest.csv", index=False)
    history = resumed["history"] if resumed else []
    chosen = resumed["best_checkpoint"] if resumed else None
    best = chosen["score"] if chosen else -float("inf")
    step = resumed["step"] if resumed else 0
    elapsed_before = history[-1]["elapsed_seconds"] if history else 0.0
    loaders = {"train": train_loader, "val": val_loader}
    if resumed:
        optimizer.load_state_dict(resumed["optimizer"])
        atomic_save(chosen, out / "best.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        restore_rng(resumed["rng"], loaders)
    start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        component_sums = {key: 0.0 for key in ["classification", "scale", "prototype", "maturation"]}
        for batch in train_loader:
            step += 1
            update_lr(optimizer, step, warmup, steps, train_cfg.get("min_lr", 1e-6))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(*[batch[k].to(device) for k in ["crops", "validity", "extents"]])
            loss, parts = objective(
                outputs,
                batch["label"].to(device),
                batch["maturation"].float().to(device),
                model,
                weights,
                train_cfg,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(batch["label"])
            for key, value in parts.items():
                component_sums[key] += value.detach().item() * len(batch["label"])
            n += len(batch["label"])
        validation = predict(model, val_loader, device, seed, split="val")
        score = selection_score(validation, config["classes"], bool(target))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / n,
                "validation_macro_f1": score,
                "elapsed_seconds": elapsed_before + time.perf_counter() - start,
                "learning_rate_min": min(g["lr"] for g in optimizer.param_groups),
                "learning_rate_max": max(g["lr"] for g in optimizer.param_groups),
                **{f"loss_{key}": value / n for key, value in component_sums.items()},
            }
        )
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if score > best:
            best = score
            chosen = {"state_dict": cpu_state(model), "config": config, "epoch": epoch, "score": best}
            atomic_save(chosen, out / "best.pt")
            validation.to_csv(out / "validation_raw.csv", index=False)
        atomic_save(
            {
                "format_version": 1,
                "state_dict": cpu_state(model),
                "config": config,
                "epoch": epoch,
                "step": step,
                "optimizer": optimizer.state_dict(),
                "best_checkpoint": chosen,
                "history": history,
                "preprocessing": preprocessing,
                "rng": capture_rng(loaders),
                "manifest_sha256": sha256(config["manifest"]),
                "device": str(device),
                "torch_version": str(torch.__version__),
            },
            out / "resume.pt",
        )
        print(f"epoch={epoch}/{epochs} loss={total_loss / n:.5f} validation_macro_f1={score:.5f}", flush=True)
        if stop_after_epoch == epoch and epoch < epochs:
            write_json(
                out / "run_state.json",
                {"status": "paused", "completed_epoch": epoch, "test_evaluated": False},
            )
            return out
    atomic_save({"state_dict": cpu_state(model), "config": config, "epoch": epochs}, out / "last.pt")
    chosen = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(chosen["state_dict"])
    validation = predict(model, val_loader, device, seed, split="val")
    validation.to_csv(out / "validation_raw.csv", index=False)
    temperature = fit_temperature(
        validation[[f"logit_{i}" for i in range(len(config["classes"]))]], validation.y_true
    )
    write_json(
        out / "calibration.json",
        {"temperature": temperature, "fit_partition": "val", "checkpoint_sha256": sha256(out / "best.pt")},
    )
    chosen["temperature"] = temperature
    atomic_save(chosen, out / "calibrated.pt")
    validation_calibrated = predict(model, val_loader, device, seed, temperature, "val")
    validation_calibrated.to_csv(out / "validation_predictions.csv", index=False)
    write_json(out / "validation_metrics.json", evaluate_frame(validation_calibrated, config["classes"]))
    if len(frames["test"]):
        loader = make_loader(dataset_for(frames["test"], config), train_cfg)
        test = predict(model, loader, device, seed, temperature)
        test["checkpoint_sha256"] = sha256(out / "calibrated.pt")
        test.to_csv(out / "test_predictions.csv", index=False)
        write_json(out / "test_metrics.json", evaluate_frame(test, config["classes"]))
    write_json(
        out / "run_summary.json",
        {
            "selected_epoch": chosen["epoch"],
            "validation_macro_f1": best,
            "temperature": temperature,
            "elapsed_seconds": elapsed_before + time.perf_counter() - start,
            "checkpoint_sha256": sha256(out / "calibrated.pt"),
            "manifest_sha256": sha256(out / "manifest.csv"),
        },
    )
    write_json(
        out / "run_state.json",
        {"status": "complete", "completed_epoch": epochs, "test_evaluated": bool(len(frames["test"]))},
    )
    return out


def load_model(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    config = payload["config"]
    model = build_model(config, initialize=False).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, config, payload.get("temperature", 1.0)


def evaluate_checkpoint(checkpoint, manifest_path, split, out, roots=None, device="cpu"):
    model, config, temperature = load_model(checkpoint, device)
    if roots:
        config["roots"] = roots
    frame = pd.read_csv(manifest_path)
    assert_no_leakage(frame)
    frame = frame[frame.split == split]
    loader = make_loader(dataset_for(frame, config), config.get("training", {}))
    predictions = predict(model, loader, device, config.get("seed", 1729), temperature, split)
    predictions["checkpoint_sha256"] = sha256(checkpoint)
    Path(out).mkdir(parents=True, exist_ok=True)
    predictions.to_csv(Path(out) / "predictions.csv", index=False)
    write_json(Path(out) / "metrics.json", evaluate_frame(predictions, config["classes"]))
