"""Fixed-label classification, calibration, selective prediction and difficult pairs."""

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix, roc_auc_score


def classification(y, predicted, classes):
    y, predicted = np.asarray(y, int), np.asarray(predicted, int)
    k = len(classes)
    if not len(y) or len(predicted) != len(y):
        raise ValueError("Metrics need equally sized nonempty arrays")
    if min(y.min(), predicted.min()) < 0 or max(y.max(), predicted.max()) >= k:
        raise ValueError("Label outside fixed task label space")
    cm = confusion_matrix(y, predicted, labels=np.arange(k))
    tp = np.diag(cm)
    support, guesses = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, guesses, out=np.zeros(k, float), where=guesses > 0)
    recall = np.divide(tp, support, out=np.zeros(k, float), where=support > 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(k, float), where=precision + recall > 0
    )
    return {
        "n": len(y),
        "accuracy": float(tp.sum() / len(y)),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "balanced_accuracy": recall.mean(),
        "macro_f1": f1.mean(),
        "weighted_f1": (f1 * support).sum() / len(y),
        "confusion_matrix": cm,
        "per_class": [
            {"label": c, "support": n, "precision": p, "recall": r, "f1": f}
            for c, n, p, r, f in zip(classes, support, precision, recall, f1)
        ],
    }


def fit_temperature(logits, y):
    logits, y = np.asarray(logits, float), np.asarray(y, int)
    if not len(y) or not np.isfinite(logits).all():
        raise ValueError("Calibration needs finite validation logits")
    temperatures = np.geomspace(0.05, 20, 1001)
    losses = [np.mean(logsumexp(logits / t, axis=1) - logits[np.arange(len(y)), y] / t) for t in temperatures]
    return float(temperatures[np.argmin(losses)])


def calibration(y, probabilities, bins=15):
    y, p = np.asarray(y, int), np.asarray(probabilities, float)
    if (
        len(y) != len(p)
        or not len(y)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1)
    ):
        raise ValueError("Invalid probabilities")
    confidence = p.max(1)
    correct = p.argmax(1) == y
    bin_id = np.maximum(0, np.ceil(confidence * bins).astype(int) - 1)
    entries, ece = [], 0.0
    for b in range(bins):
        mask = bin_id == b
        accuracy = correct[mask].mean() if mask.any() else 0.0
        conf = confidence[mask].mean() if mask.any() else 0.0
        ece += mask.mean() * abs(accuracy - conf)
        entries.append(
            {
                "lower": b / bins,
                "upper": (b + 1) / bins,
                "n": int(mask.sum()),
                "accuracy": accuracy,
                "confidence": conf,
            }
        )
    order = np.argsort(-confidence, kind="stable")
    sorted_conf, errors = confidence[order], ~correct[order]
    endpoints = np.r_[np.flatnonzero(sorted_conf[:-1] != sorted_conf[1:]) + 1, len(y)]
    coverage = np.r_[0.0, endpoints / len(y)]
    risk = np.r_[0.0, np.cumsum(errors)[endpoints - 1] / endpoints]
    trapezoid = getattr(np, "trapezoid", np.trapz if hasattr(np, "trapz") else None)
    return {
        "nll": -np.log(np.maximum(p[np.arange(len(y)), y], 1e-300)).mean(),
        "brier": np.square(p - np.eye(p.shape[1])[y]).sum(1).mean(),
        "ece": ece,
        "aurc": trapezoid(risk, coverage),
        "coverage_at_empirical_risk_1pct": coverage[risk <= 0.01].max(),
        "bins": entries,
        "coverage": coverage,
        "risk": risk,
    }


def difficult_pair(y, p, first, second):
    y, p = np.asarray(y), np.asarray(p)
    use = np.isin(y, [first, second])
    y, p = y[use], p[use]
    pred = p.argmax(1)
    recalls = [
        float((pred[y == label] == label).mean()) if (y == label).any() else 0.0 for label in [first, second]
    ]
    return {
        "n": len(y),
        "accuracy": float((y == pred).mean()) if len(y) else None,
        "balanced_accuracy": np.mean(recalls),
        "auroc": roc_auc_score(y == second, p[:, second]) if len(np.unique(y)) == 2 else None,
        "outside_pair_predictions": int((~np.isin(pred, [first, second])).sum()),
    }


def validate_predictions(frame, classes):
    required = {"id", "group", "dataset", "seed", "y_true", "prediction"}
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {required - set(frame)}")
    if frame[["seed", "id"]].duplicated().any():
        raise ValueError("Repeated image within a seed (invalid OOF or duplicate predictions)")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing prediction metadata")
    pcols = [f"prob_{i}" for i in range(len(classes))]
    if set(pcols) <= set(frame):
        p = frame[pcols].to_numpy()
        calibration(frame.y_true, p)
        if not np.array_equal(p.argmax(1), frame.prediction.to_numpy()):
            raise ValueError("Predicted class disagrees with probabilities")
    classification(frame.y_true, frame.prediction, classes)


def evaluate_frame(frame, classes):
    validate_predictions(frame, classes)
    results = {}
    for seed, part in frame.groupby("seed", sort=True):
        metrics = classification(part.y_true, part.prediction, classes)
        pcols = [f"prob_{i}" for i in range(len(classes))]
        if set(pcols) <= set(part):
            metrics["calibration"] = calibration(part.y_true, part[pcols])
        domain = {
            name: classification(d.y_true, d.prediction, classes) for name, d in part.groupby("dataset")
        }
        metrics["domains"] = domain
        metrics["equal_domain_accuracy"] = np.mean([d["accuracy"] for d in domain.values()])
        metrics["equal_domain_macro_f1"] = np.mean([d["macro_f1"] for d in domain.values()])
        results[str(seed)] = metrics
    summary = {}
    for key in [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "equal_domain_accuracy",
        "equal_domain_macro_f1",
    ]:
        values = [r[key] for r in results.values()]
        summary[key] = {
            "mean": np.mean(values),
            "sample_std": np.std(values, ddof=1) if len(values) > 1 else None,
        }
    return {
        "classes": classes,
        "runs": results,
        "summary": summary,
        "seeds": len(results),
        "note": "Seed standard deviation is not sampling uncertainty. Metrics are fractions, not percentages.",
    }
