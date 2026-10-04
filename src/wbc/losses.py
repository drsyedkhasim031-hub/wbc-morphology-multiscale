"""Manuscript reductions: true-class weighted smoothed CE and eligible Huber."""

import torch
from torch.nn import functional as F


def effective_weights(counts, beta=0.999):
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if (counts <= 0).any():
        raise ValueError("Every task class must occur in the fitting partition")
    weights = (1 - beta) / (1 - beta**counts)
    return weights / weights.mean()


def objective(outputs, labels, maturation_targets, model, weights, cfg):
    logp = outputs["logits"].log_softmax(-1)
    smooth = cfg.get("label_smoothing", 0.05)
    ce = -(1 - smooth) * logp.gather(1, labels[:, None]).squeeze(1) - smooth * logp.mean(-1)
    classification = (ce * weights[labels]).mean()
    if model.rgb_only:
        zero = classification * 0
        return classification, {
            "classification": classification,
            "scale": zero,
            "prototype": zero,
            "maturation": zero,
        }
    tokens = F.normalize(outputs["tokens"], dim=-1, eps=1e-8)
    pair = torch.triu_indices(tokens.shape[1], tokens.shape[1], offset=1, device=tokens.device)
    scale = (
        (1 - (tokens[:, pair[0]] * tokens[:, pair[1]]).sum(-1)).mean() if pair.numel() else classification * 0
    )
    p = F.normalize(model.prototypes, dim=-1, eps=1e-8)
    pair = torch.triu_indices(p.shape[1], p.shape[1], offset=1, device=p.device)
    proto = (
        (p[:, pair[0]] * p[:, pair[1]]).sum(-1).clamp_min(0).square().mean()
        if pair.numel()
        else classification * 0
    )
    eligible = torch.isfinite(maturation_targets)
    ordinal = (
        F.huber_loss(outputs["maturation"][eligible], maturation_targets[eligible], delta=0.1)
        if eligible.any() and model.maturation_head is not None
        else classification * 0
    )
    total = (
        classification
        + cfg.get("scale_weight", 0.15) * scale
        + cfg.get("prototype_weight", 0.05) * proto
        + cfg.get("maturation_weight", 0.2) * ordinal
    )
    return total, {
        "classification": classification,
        "scale": scale,
        "prototype": proto,
        "maturation": ordinal,
    }
