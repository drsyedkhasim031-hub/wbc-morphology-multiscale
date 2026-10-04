"""End-to-end in-memory RGB profiling, including cropping and all stream passes."""

import time
import numpy as np
import torch
from .crops import make_crops
from .engine import load_model
from .utils import environment


def profile(checkpoint, images, batch_size=1, warmup=50, repetitions=200, device="cpu"):
    model, config, _ = load_model(checkpoint, device)
    if repetitions < 1 or warmup < 0 or batch_size < 1:
        raise ValueError("Invalid profiling counts")
    image_batch = [images[i % len(images)] for i in range(batch_size)]

    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def step():
        items = [make_crops(im, model.factors, config.get("image_size", 224)) for im in image_batch]
        batch = [torch.stack([item[j] for item in items]).to(device) for j in range(3)]
        return model(*batch)

    for _ in range(warmup):
        step()
    synchronize()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(repetitions):
        synchronize()
        start = time.perf_counter()
        step()
        synchronize()
        elapsed.append(time.perf_counter() - start)
    return {
        "environment": environment(),
        "batch_cells": batch_size,
        "warmup": warmup,
        "repetitions": repetitions,
        "unique_parameters": sum(p.numel() for p in model.parameters()),
        "mean_batch_latency_ms": np.mean(elapsed) * 1000,
        "mean_latency_per_cell_ms": np.mean(elapsed) * 1000 / batch_size,
        "measured_throughput_cells_s": batch_size * repetitions / sum(elapsed),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
        "scope": "CPU crop geometry and validity, RGB decomposition, transfer, all encoder passes, geometry, classifier. Disk reads excluded.",
        "flops": None,
        "flops_note": "No unsupported FLOP estimate. Nonlinear preprocessing and CPU geometry are included in timing.",
    }
