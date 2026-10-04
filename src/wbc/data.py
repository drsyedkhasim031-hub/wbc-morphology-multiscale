"""Auditable manifests and connected-component group splitting."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .crops import make_crops
from .labels import NATIVE, EXPECTED, normalize_label, harmonize, maturation
from .utils import write_json


def pixel_hash(image):
    rgb = np.asarray(image.convert("RGB"))
    return hashlib.sha256(np.array(rgb.shape, dtype="<i8").tobytes() + rgb.tobytes()).hexdigest()


def perceptual_hash(image):
    rgb = np.asarray(image.convert("RGB"), dtype=float)
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    gray = F.interpolate(
        torch.from_numpy(gray)[None, None], (32, 32), mode="bilinear", align_corners=False, antialias=False
    )[0, 0].numpy()
    coefficients = dctn(gray, type=2, norm="ortho")[:8, :8].ravel()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    return f"{sum(int(b) << i for i, b in enumerate(bits)):016x}"


def build_manifest(root, dataset, output, metadata=None):
    root = Path(root).resolve()
    meta = pd.read_csv(metadata, dtype=str).fillna("").set_index("path").to_dict("index") if metadata else {}
    rows, excluded = [], []
    for file in sorted(root.rglob("*")):
        if (
            file.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
            or "__MACOSX" in file.parts
        ):
            continue
        rel = file.relative_to(root).as_posix()
        info = meta.get(rel, {})
        label = info.get("original_label", "")
        if not label:
            candidates = [
                normalize_label(part, dataset) for part in reversed(file.relative_to(root).parts[:-1])
            ]
            label = next((x for x in candidates if x in NATIVE[dataset]), "")
            if not label and dataset == "aml":
                label = file.stem.split("_")[0].upper()
        label = normalize_label(label, dataset)
        collection = info.get("collection", "")
        if not collection:
            normalized = [p.lower().replace("_", "").replace("-", "").replace(" ", "") for p in file.parts]
            collection = next(
                (
                    name
                    for name, tag in [("test_a", "testa"), ("test_b", "testb"), ("train", "train")]
                    if tag in normalized
                ),
                "unspecified",
            )
        if label not in NATIVE[dataset] or (dataset == "raabin" and collection == "test_b"):
            excluded.append({"path": rel, "label": label, "collection": collection})
            continue
        with Image.open(file) as image:
            image.load()
            exact, phash = pixel_hash(image), perceptual_hash(image)
            width, height = image.size
        maturation_value = maturation(label, dataset)
        if str(info.get("maturation_eligible", "true")).lower() in ("false", "0", "missing", "ambiguous"):
            maturation_value = float("nan")
        rows.append(
            {
                "id": f"{dataset}:{rel}",
                "path": rel,
                "dataset": dataset,
                "original_label": label,
                "label": label,
                "collection": collection,
                "patient_id": info.get("patient_id", ""),
                "slide_id": info.get("slide_id", ""),
                "exact_hash": exact,
                "phash": phash,
                "width": width,
                "height": height,
                "maturation": maturation_value,
                "dataset_version": info.get("dataset_version", "unspecified"),
            }
        )
    if not rows:
        raise ValueError("No eligible images found. Supply metadata CSV if folder labels are ambiguous.")
    frame = pd.DataFrame(rows)
    frame = connect_groups(frame)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    observed = frame.original_label.value_counts().to_dict()
    write_json(
        str(output) + ".audit.json",
        {
            "dataset": dataset,
            "observed": observed,
            "expected_manuscript": EXPECTED[dataset],
            "counts_match": observed == EXPECTED[dataset],
            "excluded": excluded,
            "patient_ids_available": int((frame.patient_id != "").sum()),
            "slide_ids_available": int((frame.slide_id != "").sum()),
            "warning": "Matching counts do not verify original experiment membership or patient independence.",
        },
    )
    return frame


def connect_groups(frame, confirmed_links=None):
    frame = frame.copy().reset_index(drop=True)
    if frame.id.duplicated().any():
        raise ValueError("Duplicate image IDs")
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(b)] = find(a)

    for col in ["patient_id", "slide_id", "exact_hash", "group"]:
        if col not in frame:
            continue
        seen = {}
        for i, row in frame.iterrows():
            value = row[col]
            if pd.isna(value) or str(value) == "":
                continue
            key = (row.dataset, str(value)) if col in ("patient_id", "slide_id") else str(value)
            if key in seen:
                union(i, seen[key])
            seen[key] = i
    if confirmed_links is not None:
        ids = {v: i for i, v in enumerate(frame.id)}
        for link in confirmed_links.itertuples():
            if link.id_a not in ids or link.id_b not in ids:
                raise ValueError("Confirmed link contains unknown ID")
            union(ids[link.id_a], ids[link.id_b])
    groups = {}
    for i in range(len(frame)):
        groups.setdefault(find(i), []).append(i)
    for indices in groups.values():
        name = min(frame.iloc[indices].id)
        frame.loc[indices, "group"] = "g:" + hashlib.sha256(name.encode()).hexdigest()[:20]
    return frame


def duplicate_candidates(frame, max_distance=4):
    """Five disjoint hash bands guarantee candidates for <=4 differing bits."""
    if max_distance != 4:
        raise ValueError("This index implements the manuscript's Hamming threshold 4")
    bands = [dict() for _ in range(5)]
    hashes, rows = [], []
    for i, row in enumerate(frame.itertuples()):
        h = int(str(row.phash), 16)
        candidates = set()
        for b, (start, length) in enumerate([(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]):
            key = (h >> start) & ((1 << length) - 1)
            candidates.update(bands[b].get(key, []))
            bands[b].setdefault(key, []).append(i)
        for j in sorted(candidates):
            distance = (h ^ hashes[j]).bit_count()
            if distance <= 4:
                rows.append({"id_a": frame.iloc[j].id, "id_b": row.id, "hamming": distance, "confirmed": ""})
        hashes.append(h)
    return pd.DataFrame(rows, columns=["id_a", "id_b", "hamming", "confirmed"])


def greedy_allocate(frame, fractions, seed=1729, domains=False):
    """Whole groups, filename order then PCG shuffle then stable descending size."""
    names = list(fractions)
    fraction = np.asarray(list(fractions.values()), float)
    if not np.isclose(fraction.sum(), 1) or (fraction <= 0).any():
        raise ValueError("Split fractions must be positive and sum to one")
    columns = [np.ones(len(frame))]
    columns += [(frame.label == label).to_numpy(float) for label in sorted(frame.label.unique())]
    if domains:
        columns += [(frame.dataset == d).to_numpy(float) for d in sorted(frame.dataset.unique())]
    matrix = np.stack(columns, axis=1)
    groups = sorted(frame.group.unique(), key=lambda g: min(frame.loc[frame.group == g, "id"]))
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(groups)
    indices = {g: np.flatnonzero(frame.group.to_numpy() == g) for g in groups}
    groups.sort(key=lambda g: -len(indices[g]))
    targets = fraction[:, None] * matrix.sum(0)[None]
    counts = np.zeros_like(targets)
    assignment = {}
    for group in groups:
        addition = matrix[indices[group]].sum(0)
        costs = []
        for p in range(len(names)):
            proposal = counts.copy()
            proposal[p] += addition
            costs.append(np.sum(np.abs(proposal - targets) / np.maximum(targets, 1)))
        chosen = int(np.argmin(costs))
        counts[chosen] += addition
        assignment[group] = names[chosen]
    return frame.group.map(assignment)


def split_manifest(frame, protocol, target=None, fold=1, seed=1729, links=None):
    frame = connect_groups(frame, links)
    frame["split"] = "excluded"
    if protocol == "lodo":
        if target not in NATIVE:
            raise ValueError("LODO requires target raabin, pbc, or aml")
        frame["label"] = [harmonize(r.original_label, r.dataset) for r in frame.itertuples()]
        frame = frame[frame.label.notna()].copy()
        target_rows = frame.dataset == target
        frame.loc[target_rows, "split"] = "test"
        overlap = frame.group.isin(frame.loc[target_rows, "group"])
        source = ~target_rows & ~overlap
        if set(frame.loc[source, "dataset"]) != set(NATIVE) - {target}:
            raise ValueError("LODO needs both source datasets")
        frame.loc[source, "split"] = greedy_allocate(
            frame.loc[source], {"train": 0.85, "val": 0.15}, seed, domains=True
        )
    elif protocol == "aml":
        frame = frame[frame.dataset == "aml"].copy()
        if fold not in range(1, 6):
            raise ValueError("AML fold must be 1..5")
        outer = greedy_allocate(frame, {str(i): 0.2 for i in range(1, 6)}, seed)
        frame["outer_fold"] = outer.astype(int)
        frame.loc[outer == str(fold), "split"] = "test"
        source = outer != str(fold)
        frame.loc[source, "split"] = greedy_allocate(frame.loc[source], {"train": 0.9, "val": 0.1}, seed)
    elif protocol == "raabin":
        frame = frame[frame.dataset == "raabin"].copy()
        test = frame.collection == "test_a"
        if not test.any():
            raise ValueError("Raabin native requires explicit test_a collection; use metadata CSV")
        frame.loc[test, "split"] = "test"
        source = ~test & ~frame.group.isin(frame.loc[test, "group"])
        frame.loc[source, "split"] = greedy_allocate(
            frame.loc[source], {"train": 8649 / 10175, "val": 1526 / 10175}, seed
        )
    elif protocol == "pbc":
        frame = frame[frame.dataset == "pbc"].copy()
        frame["split"] = greedy_allocate(
            frame, {"train": 9235 / 13193, "val": 1979 / 13193, "test": 1979 / 13193}, seed
        )
    else:
        raise ValueError("Unknown protocol")
    assert_no_leakage(frame)
    return frame


def assert_no_leakage(frame):
    active = frame[frame.split.isin(["train", "val", "test"])]
    for column in ("id", "group", "exact_hash"):
        if column in active and active.groupby(column).split.nunique().max() > 1:
            raise ValueError(f"Cross-partition leakage via {column}")
    if active.id.duplicated().any():
        raise ValueError("Repeated cell ID")


class Cells(Dataset):
    def __init__(self, frame, roots, labels, factors=(0.8, 1.0, 1.2), size=224, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.roots, self.labels, self.factors, self.size, self.augment = roots, labels, factors, size, augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        root = Path(self.roots[row.dataset]).resolve()
        path = (root / row.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path leaves configured dataset root")
        with Image.open(path) as image:
            crops, valid, extent = make_crops(image.convert("RGB"), self.factors, self.size, self.augment)
        mat = float(row.maturation) if "maturation" in row else maturation(row.original_label, row.dataset)
        return {
            "crops": crops,
            "validity": valid,
            "extents": extent,
            "label": self.labels.index(row.label),
            "maturation": mat,
            "index": index,
        }
