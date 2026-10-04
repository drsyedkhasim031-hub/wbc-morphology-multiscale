"""Model factory shared by training, evaluation and inference."""

from .rgb import MODEL_CATALOG, RGBClassifier


def build_model(config, initialize=True):
    """Build from the full run config; loading never downloads pretrained weights."""
    from ..model import WBCModel

    options = dict(config["model"])
    architecture = options.pop("architecture", "proposed")
    if not initialize:
        options["pretrained"] = False
    if architecture == "proposed":
        return WBCModel(len(config["classes"]), seed=config.get("seed", 1729), **options)
    if architecture == "rgb":
        name = options.get("backbone", "resnet18")
        if name not in MODEL_CATALOG:
            raise ValueError(f"Unknown RGB backbone: {name}")
        minimum = MODEL_CATALOG[name]["minimum_size"]
        size = config.get("image_size", 224)
        if size < minimum or (name == "vit_b_16" and size != 224):
            raise ValueError(f"{name} requires size >= {minimum}; ViT requires exactly 224")
        return RGBClassifier(len(config["classes"]), **options)
    raise ValueError(f"Unknown architecture: {architecture!r}; choose proposed or rgb")


__all__ = ["build_model", "RGBClassifier", "MODEL_CATALOG"]
