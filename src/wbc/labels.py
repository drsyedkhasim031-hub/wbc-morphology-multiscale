"""Explicit label spaces; MYB is myelocyte and MYO is myeloblast."""

COMMON = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
AML = [
    "BAS",
    "EBO",
    "EOS",
    "KSC",
    "LYA",
    "LYT",
    "MMZ",
    "MOB",
    "MON",
    "MYB",
    "MYO",
    "NGB",
    "NGS",
    "PMB",
    "PMO",
]
NATIVE = {"raabin": COMMON, "pbc": COMMON + ["immature_granulocyte"], "aml": AML}
AML_COMMON = {
    "BAS": "basophil",
    "EOS": "eosinophil",
    "LYA": "lymphocyte",
    "LYT": "lymphocyte",
    "MON": "monocyte",
    "NGB": "neutrophil",
    "NGS": "neutrophil",
}
EXPECTED = {
    "raabin": dict(zip(COMMON, [301, 1066, 3461, 795, 8891])),
    "pbc": dict(zip(NATIVE["pbc"], [1218, 3117, 1214, 1420, 3329, 2895])),
    "aml": dict(zip(AML, [79, 78, 424, 15, 11, 3937, 15, 26, 1789, 42, 3268, 109, 8484, 18, 70])),
}
ALIASES = {
    "basophils": "basophil",
    "eosinophils": "eosinophil",
    "lymphocytes": "lymphocyte",
    "monocytes": "monocyte",
    "neutrophils": "neutrophil",
    "ig": "immature_granulocyte",
    "immature granulocytes": "immature_granulocyte",
    "immature_granulocytes": "immature_granulocyte",
    "erythroblasts": "erythroblast",
    "platelets": "platelet",
}


def normalize_label(label, dataset):
    label = str(label).strip()
    return label.upper() if dataset == "aml" else ALIASES.get(label.lower(), label.lower())


def harmonize(label, dataset):
    label = normalize_label(label, dataset)
    return AML_COMMON.get(label) if dataset == "aml" else label if label in COMMON else None


def maturation(label, dataset):
    return {"MMZ": 0.0, "NGB": 0.5, "NGS": 1.0}.get(label, float("nan")) if dataset == "aml" else float("nan")
