"""Paired group bootstrap and optional paired cluster randomization."""

import numpy as np
from scipy.stats import binomtest

from .metrics import validate_predictions


def _score(y, prediction, k, metric):
    cm = np.bincount(y * k + prediction, minlength=k * k).reshape(k, k)
    if metric == "accuracy":
        return np.trace(cm) / cm.sum()
    denominator = cm.sum(0) + cm.sum(1)
    return np.divide(2 * np.diag(cm), denominator, out=np.zeros(k, float), where=denominator > 0).mean()


def align(a, b, classes):
    validate_predictions(a, classes)
    validate_predictions(b, classes)
    keys = ["seed", "id"]
    a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
    for column in keys + ["y_true", "group", "dataset"]:
        if not a[column].equals(b[column]):
            raise ValueError(f"Models must have identical paired {column}")
    seeds = sorted(a.seed.unique())
    parts = [a[a.seed == s].sort_values("id").reset_index(drop=True) for s in seeds]
    base = parts[0]
    for part in parts[1:]:
        for col in ["id", "group", "dataset", "y_true"]:
            if not part[col].equals(base[col]):
                raise ValueError("Every seed must evaluate the same cells and labels")
    pa = np.stack([part.prediction.to_numpy(int) for part in parts])
    pb = np.stack([b[b.seed == s].sort_values("id").prediction.to_numpy(int) for s in seeds])
    return base, pa, pb


def paired_bootstrap(a, b, classes, draws=2000, seed=1729, equal_domains=False):
    base, pa, pb = align(a, b, classes)
    if draws < 2:
        raise ValueError("At least two bootstrap draws required")
    if equal_domains and base.groupby("group").dataset.nunique().max() > 1:
        raise ValueError("A group spans domains; separate-domain resampling would violate dependence")
    y = base.y_true.to_numpy(int)
    domains = sorted(base.dataset.unique()) if equal_domains else [None]
    strata = []
    for domain in domains:
        use = np.ones(len(base), bool) if domain is None else (base.dataset == domain).to_numpy()
        strata.append(
            [
                np.flatnonzero(use & (base.group.to_numpy() == g))
                for g in sorted(base.loc[use, "group"].unique())
            ]
        )
    rng = np.random.Generator(np.random.PCG64(seed))

    def difference(indices, metric):
        return np.mean(
            [
                _score(y[indices], p[indices], len(classes), metric)
                - _score(y[indices], q[indices], len(classes), metric)
                for p, q in zip(pa, pb)
            ]
        )

    result = {
        "orientation": "model_a minus model_b",
        "draws": draws,
        "seed": seed,
        "groups": base.group.nunique(),
        "grouping_scope": "Only as independent as the supplied verified grouping; image hashes do not establish patient independence.",
        "equal_domain_weighting": equal_domains,
        "cells": len(base),
        "training_seeds": len(pa),
    }
    samples = {m: [] for m in ["accuracy", "macro_f1"]}
    for _ in range(draws):
        indices = [
            np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            for groups in strata
        ]
        for metric in samples:
            samples[metric].append(np.mean([difference(ix, metric) for ix in indices]))
    for metric, values in samples.items():
        estimate = np.mean([difference(np.concatenate(groups), metric) for groups in strata])
        result[metric] = {
            "difference": estimate,
            "ci95": np.quantile(values, [0.025, 0.975], method="linear").tolist(),
        }
    return result


def cluster_randomization(a, b, classes, draws=10000, seed=1729):
    """Two-sided sign permutation of group accuracy contributions, shared over seeds.

    Optional extension, distinct from the paper's bootstrap. Assumes exchangeability
    of the two model outcomes under the null at the supplied independent-group level.
    """
    base, pa, pb = align(a, b, classes)
    y = base.y_true.to_numpy()
    difference = ((pa == y).astype(float) - (pb == y).astype(float)).mean(0)
    contributions = np.array(
        [difference[base.group.to_numpy() == g].sum() for g in sorted(base.group.unique())]
    )
    observed = abs(contributions.sum())
    rng = np.random.Generator(np.random.PCG64(seed))
    exceed = sum(
        abs((contributions * rng.choice([-1, 1], len(contributions))).sum()) >= observed - 1e-12
        for _ in range(draws)
    )
    return {
        "test": "paired cluster sign randomization, pooled accuracy",
        "p_value": (exceed + 1) / (draws + 1),
        "draws": draws,
        "seed": seed,
        "difference": difference.mean(),
    }


def mcnemar_exact(a, b, classes):
    base, pa, pb = align(a, b, classes)
    if len(pa) != 1 or base.group.duplicated().any():
        raise ValueError("Exact image-level McNemar requires one seed and independent singleton groups")
    ca, cb = pa[0] == base.y_true.to_numpy(), pb[0] == base.y_true.to_numpy()
    wins, losses = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return {
        "a_only_correct": wins,
        "b_only_correct": losses,
        "p_value": binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0,
    }


def holm(pvalues):
    p = np.asarray(pvalues, float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("P values must be finite and between zero and one")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    out = np.empty_like(p)
    out[order] = adjusted
    return out
