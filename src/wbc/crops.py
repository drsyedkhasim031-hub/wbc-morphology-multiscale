"""Coordinated geometry, white padding and conservative validity propagation."""

import math
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF, InterpolationMode


def make_crops(image, factors=(0.8, 1.0, 1.2), size=224, augment=False):
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    _, h, w = image.shape
    valid = torch.ones(1, h, w, dtype=image.dtype)
    dx = dy = 0.0
    sigma = None
    if augment:
        angle = float(torch.empty(()).uniform_(-12, 12))
        image = TF.rotate(image, angle, InterpolationMode.BILINEAR, fill=1.0)
        valid = TF.rotate(valid, angle, InterpolationMode.BILINEAR, fill=0.0)
        valid = (valid >= 1.0 - 1e-6).float()
        dx, dy = (torch.rand(2).numpy() * 0.1 - 0.05) * [w, h]
        if torch.rand(()) < 0.5:
            sigma = float(torch.empty(()).uniform_(0.1, 0.5))
    crops, validity, extents = [], [], []
    for factor in factors:
        cw, ch = int(math.floor(w * factor + 0.5)), int(math.floor(h * factor + 0.5))
        left, top = math.floor(w / 2 + dx - cw / 2), math.floor(h / 2 + dy - ch / 2)
        rgb = torch.ones(3, ch, cw)
        v = torch.zeros(1, ch, cw)
        x0, y0, x1, y1 = max(left, 0), max(top, 0), min(left + cw, w), min(top + ch, h)
        if x1 > x0 and y1 > y0:
            rgb[:, y0 - top : y1 - top, x0 - left : x1 - left] = image[:, y0:y1, x0:x1]
            v[:, y0 - top : y1 - top, x0 - left : x1 - left] = valid[:, y0:y1, x0:x1]
        rgb = F.interpolate(rgb[None], (size, size), mode="bilinear", align_corners=False, antialias=False)[0]
        v = F.interpolate(v[None], (size, size), mode="bilinear", align_corners=False, antialias=False)[0]
        v = (v >= 1.0 - 1e-6).float()
        if sigma is not None:
            rgb = TF.gaussian_blur(rgb, [5, 5], [sigma, sigma])
            invalid = F.pad(1 - v[None], (2, 2, 2, 2), mode="reflect")
            v = 1 - F.max_pool2d(invalid, 5, stride=1)[0]
        crops.append(rgb)
        validity.append(v)
        extents.append([cw, ch])
    return torch.stack(crops), torch.stack(validity), torch.tensor(extents, dtype=torch.float32)
