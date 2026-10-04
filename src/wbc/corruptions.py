"""Deterministic perturbations on acquired RGB, before crop extraction/padding."""

from io import BytesIO
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from .stain import optical_density, nnls_two


def corrupt(image, kind, severity, variant=0, basis=None):
    if severity not in (1, 2, 3):
        raise ValueError("Severity must be 1, 2, or 3")
    rgb = np.asarray(image, dtype=float)
    i = severity - 1
    if kind in ("gamma", "stain") and variant not in (0, 1):
        raise ValueError("Paired corruptions have variants 0 and 1")
    if kind == "gamma":
        gamma = [(0.8, 1.2), (0.6, 1.4), (0.4, 1.6)][i][variant]
        out = 255 * (rgb / 255) ** gamma
    elif kind == "blur":
        sigma = [0.5, 1.0, 1.5][i]
        out = gaussian_filter(
            rgb,
            (sigma, sigma, 0),
            mode="reflect",
            radius=(int(np.ceil(3 * sigma)), int(np.ceil(3 * sigma)), 0),
        )
    elif kind == "jpeg":
        buffer = BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(
            buffer, format="JPEG", quality=[90, 70, 50][i], subsampling=0
        )
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    elif kind == "stain":
        if basis is None:
            raise ValueError("A fixed source-fitting reference basis is required")
        od = optical_density(rgb).reshape(-1, 3)
        c = nnls_two(od, basis)
        residual = od - c @ basis.T
        angle = np.deg2rad([5, 10, 15][i]) * (-1 if variant == 0 else 1)
        rotation = Rotation.from_rotvec(np.ones(3) / np.sqrt(3) * angle).as_matrix()
        shifted = np.maximum(c @ (rotation @ basis).T + residual, 0)
        out = (256 * np.exp(-shifted) - 1).reshape(rgb.shape)
    else:
        raise ValueError("Unknown corruption")
    return Image.fromarray(np.floor(np.clip(out, 0, 255) + 0.5).astype(np.uint8))
