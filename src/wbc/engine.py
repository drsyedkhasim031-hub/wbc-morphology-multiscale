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
from .model import WBCModel
from .preprocessing import fit_from_cells, fit_descriptors
from .utils import environment, seed_everything, sha256, write_json


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
        config["model"].get("factors", [0.8, 1.0, 1.2]),
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


def train(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError("Use a fresh output directory to preserve prior run provenance")
    seed = config.get("seed", 1729)
    seed_everything(seed)
    torch.set_num_threads(config.get("threads", 4))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
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
    model = WBCModel(len(config["classes"]), seed=seed, **config["model"]).to(device)
    fitting = dataset_for(frames["train"], config)
    preprocessing = {}
    if model.use_stain and not model.rgb_only:
        canonical = Cells(
            frames["train"], config["roots"], config["classes"], [1.0], config.get("image_size", 224)
        )
        basis, preprocessing["stain_fit"] = fit_from_cells(canonical, out / "cache")
        model.basis.copy_(torch.from_numpy(basis).to(device))
    if model.use_morphology and not model.rgb_only:
        preprocessing["descriptors"] = fit_descriptors(model, make_loader(fitting, train_cfg), device)
    preprocessing["basis"] = model.basis.detach().cpu().tolist()
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
        "status": "executed run; scope determined by config; not manuscript result verification",
    }
    write_json(out / "provenance.json", provenance)
    manifest.to_csv(out / "manifest.csv", index=False)
    history, best, step = [], -float("inf"), 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
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
            optimizer.step()
            total_loss += loss.item() * len(batch["label"])
            n += len(batch["label"])
        validation = predict(model, val_loader, device, seed, split="val")
        score = selection_score(validation, config["classes"], bool(target))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / n,
                "validation_macro_f1": score,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if score > best:
            best = score
            torch.save(
                {"state_dict": model.state_dict(), "config": config, "epoch": epoch, "score": best},
                out / "best.pt",
            )
            validation.to_csv(out / "validation_raw.csv", index=False)
        print(f"epoch={epoch}/{epochs} loss={total_loss / n:.5f} validation_macro_f1={score:.5f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epochs}, out / "last.pt")
    chosen = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(chosen["state_dict"])
    validation = predict(model, val_loader, device, seed, split="val")
    temperature = fit_temperature(
        validation[[f"logit_{i}" for i in range(len(config["classes"]))]], validation.y_true
    )
    write_json(
        out / "calibration.json",
        {"temperature": temperature, "fit_partition": "val", "checkpoint_sha256": sha256(out / "best.pt")},
    )
    chosen["temperature"] = temperature
    torch.save(chosen, out / "calibrated.pt")
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
            "elapsed_seconds": time.perf_counter() - start,
            "checkpoint_sha256": sha256(out / "calibrated.pt"),
            "manifest_sha256": sha256(out / "manifest.csv"),
        },
    )
    return out


def load_model(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    config = payload["config"]
    cfg = dict(config["model"])
    cfg["pretrained"] = False
    model = WBCModel(len(config["classes"]), seed=config.get("seed", 1729), **cfg).to(device)
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
