"""Training-only alternating NNLS stain fit and exact two-column NNLS inference."""

import numpy as np
import torch
from scipy.optimize import nnls


def optical_density(rgb):
    return -np.log((np.clip(np.asarray(rgb, dtype=np.float64), 0, 255) + 1.0) / 256.0)


def nnls_two(x, basis):
    """Solve min ||B c - x||^2, c>=0, for every row of x; enumerate active sets."""
    b = np.asarray(basis, dtype=np.float64)
    gram = b.T @ b
    unconstrained = np.linalg.solve(gram, (np.asarray(x) @ b).T).T
    one = np.stack([np.maximum(x @ b[:, 0] / gram[0, 0], 0), np.zeros(len(x))], axis=1)
    two = np.stack([np.zeros(len(x)), np.maximum(x @ b[:, 1] / gram[1, 1], 0)], axis=1)
    candidates = np.stack([np.maximum(unconstrained, 0), one, two], axis=1)
    errors = ((candidates @ b.T - x[:, None]) ** 2).sum(-1)
    errors[(unconstrained < 0).any(1), 0] = np.inf
    return candidates[np.arange(len(x)), errors.argmin(1)]


def fit_basis(pixel_arrays, seed=1729, max_pixels=100_000, iterations=100):
    """Input: filename-ordered valid, canonical RGB pixel arrays from fitting cells only.

    Callers may supply a list of disk-backed arrays for large datasets. Norm filtering
    and percentile use every supplied pixel before deterministic subsampling.
    """
    x = optical_density(np.concatenate(pixel_arrays, axis=0))
    norms = np.linalg.norm(x, axis=1)
    x, norms = x[norms > 0.15], norms[norms > 0.15]
    if not len(x):
        raise ValueError("No non-white fitting pixels available for stain estimation")
    x = x[norms <= np.quantile(norms, 0.99, method="linear")]
    if len(x) > max_pixels:
        x = x[np.random.Generator(np.random.PCG64(seed)).choice(len(x), max_pixels, replace=False)]
    return fit_od_basis(x, seed, iterations)


def fit_od_basis(x, seed=1729, iterations=100):
    """Fit a basis to an already filtered and sampled N x 3 OD matrix."""
    b = np.array([[0.70, 0.20], [0.65, 0.70], [0.25, 0.68]], dtype=np.float64)
    b /= np.linalg.norm(b, axis=0)
    previous = np.inf
    history = []
    for _ in range(iterations):
        c = nnls_two(x, b)
        b = np.stack([nnls(c, x[:, j])[0] for j in range(3)])
        norms = np.linalg.norm(b, axis=0)
        if np.any(norms < 1e-8):
            raise ValueError("Degenerate stain basis")
        b /= norms
        c *= norms
        sv = np.linalg.svd(b, compute_uv=False)
        if sv[-1] / sv[0] < 1e-8:
            raise ValueError("Collinear stain basis")
        objective = float(np.square(x - c @ b.T).sum())
        history.append(objective)
        if np.isfinite(previous) and (previous - objective) / max(previous, 1e-12) < 1e-6:
            break
        previous = objective
    order = np.argsort(b[2] / np.maximum(b[0], 1e-8), kind="stable")
    return b[:, order].astype(np.float32), {
        "sampled_pixels": len(x),
        "objectives": history,
        "seed": seed,
        "white_reference": 255,
    }


def decompose(rgb, basis):
    """BCHW in [0,1] -> B,2,3,H,W reconstructed components; no residual stream."""
    x = -torch.log((rgb.clamp(0, 1) * 255 + 1) / 256)
    b = basis.to(x)
    flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
    gram = b.T @ b
    uc = torch.linalg.solve(gram, (flat @ b).T).T
    one = torch.stack([(flat @ b[:, 0] / gram[0, 0]).clamp_min(0), torch.zeros_like(uc[:, 0])], -1)
    two = torch.stack([torch.zeros_like(uc[:, 0]), (flat @ b[:, 1] / gram[1, 1]).clamp_min(0)], -1)
    candidates = torch.stack([uc.clamp_min(0), one, two], 1)
    errors = (candidates @ b.T - flat[:, None]).square().sum(-1)
    errors[:, 0] = errors[:, 0].masked_fill((uc < 0).any(-1), float("inf"))
    c = candidates[torch.arange(len(flat), device=x.device), errors.argmin(-1)]
    component = (torch.exp(-c[:, :, None] * b.T[None]) * 256 - 1).clamp(0, 255) / 255
    n, _, h, w = rgb.shape
    return component.reshape(n, h, w, 2, 3).permute(0, 3, 4, 1, 2)
