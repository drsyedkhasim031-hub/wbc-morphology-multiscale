"""Frozen single-cell inference with provisional compartment overlays."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from wbc.crops import make_crops
from wbc.engine import load_model
from wbc.utils import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, config, temperature = load_model(args.checkpoint, args.device)
    results = []
    palette = np.array([[38, 95, 185], [237, 143, 38], [240, 240, 240]], dtype=np.uint8)
    with torch.no_grad():
        for index, path in enumerate(args.images):
            with Image.open(path) as image:
                crops, valid, extent = make_crops(
                    image.convert("RGB"), model.factors, config.get("image_size", 224)
                )
            output = model(
                crops[None].to(args.device), valid[None].to(args.device), extent[None].to(args.device)
            )
            probabilities = (output["logits"] / temperature).softmax(-1)[0].cpu().numpy()
            record = {
                "input": str(path),
                "predicted_class": config["classes"][probabilities.argmax()],
                "probabilities": dict(zip(config["classes"], probabilities)),
                "temperature": temperature,
            }
            if "masks" in output:
                canonical = min(range(len(model.factors)), key=lambda i: abs(model.factors[i] - 1))
                soft = output["masks"][0, canonical].cpu().numpy()
                hard = soft.argmax(0)
                np.savez_compressed(
                    out / f"cell_{index:04d}_masks.npz",
                    soft_masks=soft,
                    hard_labels=hard,
                    valid=output["validity_grid"][0, canonical].cpu().numpy(),
                )
                rgb = Image.fromarray(
                    np.uint8(np.clip(crops[canonical].permute(1, 2, 0).numpy() * 255, 0, 255))
                )
                color = Image.fromarray(palette[hard]).resize(rgb.size, Image.Resampling.NEAREST)
                Image.blend(rgb, color, 0.35).save(out / f"cell_{index:04d}_provisional_overlay.png")
                record["compartment_note"] = (
                    "Unsupervised bank identities are provisional; this is not a validated anatomical segmentation."
                )
            results.append(record)
    write_json(out / "predictions.json", results)


if __name__ == "__main__":
    main()
