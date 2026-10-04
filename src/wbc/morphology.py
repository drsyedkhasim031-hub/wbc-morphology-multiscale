"""Nondifferentiable geometry on provisional compartment assignments."""

import numpy as np
from scipy import ndimage as ndi

NAMES = [
    "nc_ratio",
    "nuclear_fraction",
    "cell_fraction",
    "nuclear_perimeter_normalized",
    "nuclear_eccentricity",
    "cell_eccentricity",
    "centroid_offset_x",
    "centroid_offset_y",
]


def selected_masks(labels, valid, extent=None):
    components, n = ndi.label((labels < 2) & valid, structure=np.ones((3, 3)))
    if not n:
        empty = np.zeros_like(valid, dtype=bool)
        return empty, empty
    centre = (np.array(labels.shape) - 1) / 2
    spacing = (
        np.array([extent[1] / labels.shape[0], extent[0] / labels.shape[1]])
        if extent is not None
        else np.ones(2)
    )
    candidates = []
    for i in range(1, n + 1):
        points = np.argwhere(components == i)
        candidates.append(
            (
                np.square((points.mean(0) - centre) * spacing).sum(),
                np.ravel_multi_index(points[0], labels.shape),
                i,
            )
        )
    cell = components == min(candidates)[2]
    return cell & (labels == 0), cell & (labels == 1)


def descriptors(labels, valid, extent):
    valid = np.asarray(valid, dtype=bool)
    nucleus, cyto = selected_masks(labels, valid, extent)
    cell = nucleus | cyto
    bad_border = np.zeros_like(cell)
    bad_border[[0, -1], :] = True
    bad_border[:, [0, -1]] = True
    bad = bad_border | ndi.binary_dilation(~valid, structure=np.ones((3, 3)))
    if nucleus.sum() < 4 or cyto.sum() < 4 or (cell & bad).any():
        return np.full(8, np.nan, dtype=np.float32)
    dx, dy = float(extent[0]) / labels.shape[1], float(extent[1]) / labels.shape[0]
    an, ac = nucleus.sum() * dx * dy, cell.sum() * dx * dy

    def moments(mask):
        y, x = np.nonzero(mask)
        coords = np.column_stack([(x + 0.5) * dx, (y + 0.5) * dy])
        centroid = coords.mean(0)
        centred = coords - centroid
        cov = centred.T @ centred / len(coords) + np.diag([dx * dx, dy * dy]) / 12
        eigen = np.linalg.eigvalsh(cov)
        return centroid, np.sqrt(max(0, 1 - eigen[0] / eigen[1]))

    cn, en = moments(nucleus)
    cc, ec = moments(cell)
    m = np.pad(nucleus.astype(np.int8), 1)
    perimeter = np.abs(np.diff(m, axis=0)).sum() * dx + np.abs(np.diff(m, axis=1)).sum() * dy
    return np.array(
        [
            an / (cyto.sum() * dx * dy),
            nucleus.sum() / valid.sum(),
            cell.sum() / valid.sum(),
            perimeter / np.sqrt(an),
            en,
            ec,
            *((cn - cc) / np.sqrt(ac)),
        ],
        dtype=np.float32,
    )
