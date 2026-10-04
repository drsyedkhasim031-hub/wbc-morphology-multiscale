"""Independent mask validation; labels 0 nucleus, 1 cytoplasm, 2 background."""

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import pearsonr
import pandas as pd

from .morphology import descriptors, selected_masks


def reference_grid(nucleus, whole_cell, size=28):
    nucleus, whole_cell = np.asarray(nucleus, bool), np.asarray(whole_cell, bool)
    if nucleus.shape != whole_cell.shape or (nucleus & ~whole_cell).any():
        raise ValueError("Reference nucleus must be a subset of whole cell on the same grid")

    # Exact fractional pixel overlap; unlike adaptive pooling this weights boundary pixels.
    def weights(original):
        lo = np.arange(size) * original / size
        hi = (np.arange(size) + 1) * original / size
        px = np.arange(original)
        return np.maximum(0, np.minimum(hi[:, None], px + 1) - np.maximum(lo[:, None], px))

    wy, wx = weights(nucleus.shape[0]), weights(nucleus.shape[1])
    labels = [nucleus, whole_cell & ~nucleus, ~whole_cell]
    areas = np.stack([wy @ m.astype(float) @ wx.T for m in labels])
    return areas.argmax(0).astype(np.uint8)


def overlap(pred, ref):
    p, r = np.asarray(pred, bool), np.asarray(ref, bool)
    if not p.any() and not r.any():
        return {"dice": 1.0, "iou": 1.0, "boundary_f1": 1.0}
    if not p.any() or not r.any():
        return {"dice": 0.0, "iou": 0.0, "boundary_f1": 0.0}
    intersection = (p & r).sum()
    structure = ndi.generate_binary_structure(2, 1)
    pb = p & ~ndi.binary_erosion(p, structure=structure, border_value=0)
    rb = r & ~ndi.binary_erosion(r, structure=structure, border_value=0)
    precision = (ndi.distance_transform_edt(~rb)[pb] <= 1).mean()
    recall = (ndi.distance_transform_edt(~pb)[rb] <= 1).mean()
    return {
        "dice": 2 * intersection / (p.sum() + r.sum()),
        "iou": intersection / (p | r).sum(),
        "boundary_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def evaluate_masks(predictions, references, classes, validities=None, extents=None):
    if predictions.shape != references.shape or len(predictions) != len(classes):
        raise ValueError("Mask arrays and classes must align")
    rows, predicted_nc, reference_nc = [], [], []
    for i, (pred, ref, label) in enumerate(zip(predictions, references, classes)):
        if not np.isin(ref, [0, 1, 2]).all() or not np.isin(pred, [0, 1, 2]).all():
            raise ValueError("Use labels nucleus=0, cytoplasm=1, background=2")
        valid = validities[i] if validities is not None else np.ones_like(pred, bool)
        extent = extents[i] if extents is not None else [pred.shape[1], pred.shape[0]]
        pn, pc = selected_masks(pred, valid, extent)
        rn, rc = ref == 0, ref == 1
        row = {"index": i, "class": str(label)}
        for name, p, r in [("nucleus", pn, rn), ("cytoplasm", pc, rc), ("whole_cell", pn | pc, rn | rc)]:
            row.update({f"{name}_{key}": value for key, value in overlap(p, r).items()})
        pnc, rnc = descriptors(pred, valid, extent)[0], descriptors(ref, valid, extent)[0]
        row.update({"predicted_nc": pnc, "reference_nc": rnc})
        predicted_nc.append(pnc)
        reference_nc.append(rnc)
        rows.append(row)
    p, r = np.array(predicted_nc), np.array(reference_nc)
    usable = np.isfinite(p) & np.isfinite(r)
    eligible = np.isfinite(r)
    table = pd.DataFrame(rows)
    metric_columns = [c for c in table if c.endswith(("dice", "iou", "boundary_f1"))]
    by_class = table.groupby("class")[metric_columns].mean()
    supports = table.groupby("class").size()
    weighted = (by_class.mul(supports, axis=0).sum() / supports.sum()).to_dict()
    return {
        "cells": rows,
        "per_class": by_class.to_dict("index"),
        "class_supports": supports.to_dict(),
        "support_weighted_means": weighted,
        "reference_eligible": int(eligible.sum()),
        "valid_pairs": int(usable.sum()),
        "measurement_coverage": usable.sum() / eligible.sum() if eligible.any() else None,
        "nc_bias": (p[usable] - r[usable]).mean() if usable.any() else None,
        "nc_mae": np.abs(p[usable] - r[usable]).mean() if usable.any() else None,
        "nc_pearson": pearsonr(p[usable], r[usable]).statistic
        if usable.sum() >= 2 and np.ptp(p[usable]) > 0 and np.ptp(r[usable]) > 0
        else None,
    }
