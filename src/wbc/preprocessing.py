"""Bounded-RAM exact fitting-pixel selection using a temporary disk array."""

from pathlib import Path
import numpy as np
import torch
from .stain import optical_density, fit_od_basis


def fit_from_cells(cells, cache_dir, seed=1729, max_pixels=100000):
    """Three deterministic passes; up to N*size^2*8 bytes temporary disk space.

    All valid unaugmented canonical pixels contribute to the 99th percentile.
    Sampling is uniform over the remaining filename/row-major ordered pixels.
    """
    if cells.augment or len(cells.factors) != 1 or cells.factors[0] != 1.0:
        raise ValueError("Stain estimation requires unaugmented canonical crops")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    norm_file = cache / "fitting_norms.dat"
    norms = np.memmap(norm_file, dtype="float64", mode="w+", shape=(len(cells) * cells.size * cells.size,))

    def pixels(i):
        item = cells[i]
        rgb = item["crops"][0].permute(1, 2, 0).numpy() * 255
        return optical_density(rgb[item["validity"][0, 0].numpy().astype(bool)])

    n = 0
    for i in range(len(cells)):
        v = np.linalg.norm(pixels(i), axis=1)
        v = v[v > 0.15]
        norms[n : n + len(v)] = v
        n += len(v)
    if not n:
        del norms
        norm_file.unlink()
        raise ValueError("No non-white valid fitting pixels")
    cutoff = np.quantile(norms[:n], 0.99, method="linear", overwrite_input=True)
    del norms
    norm_file.unlink()
    counts = []
    for i in range(len(cells)):
        v = np.linalg.norm(pixels(i), axis=1)
        counts.append(int(((v > 0.15) & (v <= cutoff)).sum()))
    total = sum(counts)
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = np.sort(rng.choice(total, min(max_pixels, total), replace=False))
    samples, offset = [], 0
    for i, count in enumerate(counts):
        choice = selected[(selected >= offset) & (selected < offset + count)] - offset
        if len(choice):
            od = pixels(i)
            v = np.linalg.norm(od, axis=1)
            samples.append(od[(v > 0.15) & (v <= cutoff)][choice])
        offset += count
    basis, metadata = fit_od_basis(np.concatenate(samples), seed)
    metadata.update({"norm_cutoff_99": cutoff, "eligible_pixels": total, "fitting_cells": len(cells)})
    return basis, metadata


@torch.no_grad()
def fit_descriptors(model, loader, device):
    model.eval()
    sums, squares, counts = np.zeros(8), np.zeros(8), np.zeros(8)
    for batch in loader:
        output = model(*[batch[key].to(device) for key in ["crops", "validity", "extents"]])
        if output["raw_descriptors"] is None:
            continue
        raw = output["raw_descriptors"].cpu().numpy().astype(float)
        good = np.isfinite(raw)
        sums += np.where(good, raw, 0).sum(0)
        squares += np.where(good, raw * raw, 0).sum(0)
        counts += good.sum(0)
    mean = np.divide(sums, counts, out=np.zeros(8), where=counts > 0)
    variance = np.divide(squares, counts, out=np.ones(8), where=counts > 0) - mean * mean
    std = np.maximum(np.sqrt(np.maximum(variance, 0)), 1e-6)
    model.descriptor_mean.copy_(torch.tensor(mean, device=device))
    model.descriptor_std.copy_(torch.tensor(std, device=device))
    return {
        "mean": mean,
        "std": std,
        "valid_counts": counts,
        "fit_stage": "initialized model before optimization",
    }
