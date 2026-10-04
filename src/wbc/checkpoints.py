"""Atomic checkpoint writes and safe, weights_only-compatible epoch resume state."""

from pathlib import Path
import random
import uuid

import numpy as np
import torch


def atomic_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def capture_rng(loaders):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": {key: loader.generator.get_state() for key, loader in loaders.items()},
    }


def restore_rng(state, loaders):
    random.setstate(state["python"])
    name, values, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the original CUDA device count")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    for key, loader in loaders.items():
        loader.generator.set_state(state["loaders"][key].cpu())
