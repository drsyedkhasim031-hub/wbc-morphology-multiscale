"""Batched directory inference and numeric representation export."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .crops import make_crops
from .engine import load_model
from .utils import sha256, write_json


def collect_images(paths):
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    found = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            found.extend(
                p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in extensions
            )
        elif path.is_file() and path.suffix.lower() in extensions:
            found.append(path.resolve())
        else:
            raise ValueError(f"Input is not an image or directory: {path}")
    found = sorted(set(found))
    if not found:
        raise ValueError("No supported images found")
    return found


@torch.no_grad()
def infer_images(checkpoint, paths, out, batch_size=16, device="cpu", features=False):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    paths = collect_images(paths)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Use an empty output directory for inference")
    model, config, temperature = load_model(checkpoint, device)
    torch.set_num_threads(config.get("threads", 4))
    classes = config["classes"]
    rows, embeddings = [], []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        inputs = []
        for path in batch_paths:
            with Image.open(path) as image:
                inputs.append(make_crops(image.convert("RGB"), model.factors, config.get("image_size", 224)))
        batch = [torch.stack(values).to(device) for values in zip(*inputs)]
        output = model(*batch)
        logits = output["logits"].cpu().numpy()
        probabilities = (output["logits"] / temperature).softmax(-1).cpu().numpy()
        if features:
            # Pre-classifier per-scale tokens, before the proposed scale transformer.
            embeddings.append(output["tokens"].cpu().numpy())
        for index, path in enumerate(batch_paths):
            winner = int(probabilities[index].argmax())
            rows.append(
                {
                    "id": f"image_{start + index:07d}",
                    "input_file": str(path),
                    "input_sha256": sha256(path),
                    "prediction": winner,
                    "predicted_class": classes[winner],
                    "confidence": float(probabilities[index, winner]),
                    **{f"prob_{k}": float(v) for k, v in enumerate(probabilities[index])},
                    **{f"logit_{k}": float(v) for k, v in enumerate(logits[index])},
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "predictions.csv", index=False)
    if features:
        np.savez_compressed(
            out / "features.npz", ids=frame.id.to_numpy(dtype=str), tokens=np.concatenate(embeddings)
        )
    write_json(
        out / "provenance.json",
        {
            "checkpoint_sha256": sha256(checkpoint),
            "classes": classes,
            "temperature": temperature,
            "model": config["model"],
            "images": len(frame),
            "batch_size": batch_size,
            "features": "per-scale pre-classifier tokens; proposed model tokens precede scale transformer"
            if features
            else None,
            "scope": "unlabeled cropped-cell inference; no accuracy or clinical validation is implied",
        },
    )
    return frame
