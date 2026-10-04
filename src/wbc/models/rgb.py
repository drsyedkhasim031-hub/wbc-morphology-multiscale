"""Standalone RGB baselines with explicit weights and no dormant morphology branches.

These are repository extensions, not additional verified manuscript experiments.
All models use the common WBC crop protocol and ImageNet channel normalization.
"""

import torch
from torch import nn


MODEL_CATALOG = {
    "compact_cnn": {"features": 128, "minimum_size": 16, "weights": None},
    "resnet18": {"features": 512, "minimum_size": 32, "weights": "IMAGENET1K_V1"},
    "resnet34": {"features": 512, "minimum_size": 32, "weights": "IMAGENET1K_V1"},
    "resnet50": {"features": 2048, "minimum_size": 32, "weights": "IMAGENET1K_V1"},
    "densenet121": {"features": 1024, "minimum_size": 32, "weights": "IMAGENET1K_V1"},
    "mobilenet_v3_large": {"features": 1280, "minimum_size": 32, "weights": "IMAGENET1K_V1"},
    "efficientnet_b0": {"features": 1280, "minimum_size": 32, "weights": "IMAGENET1K_V1"},
    "efficientnet_v2_s": {"features": 1280, "minimum_size": 32, "weights": "IMAGENET1K_V1"},
    "convnext_tiny": {"features": 768, "minimum_size": 32, "weights": "IMAGENET1K_V1"},
    "swin_t": {"features": 768, "minimum_size": 32, "weights": "IMAGENET1K_V1"},
    "vit_b_16": {"features": 768, "minimum_size": 224, "weights": "IMAGENET1K_V1"},
}


class ResidualBlock(nn.Module):
    """Small batch-friendly residual block using GroupNorm."""

    def __init__(self, incoming, outgoing):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, outgoing),
            nn.GELU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing),
        )
        self.skip = nn.Conv2d(incoming, outgoing, 1, stride=2, bias=False)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class CompactCNN(nn.Sequential):
    """Three residual stages; a CPU-friendly baseline trained from scratch."""

    def __init__(self):
        super().__init__(
            ResidualBlock(3, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )


def make_encoder(name, pretrained, weights_version):
    if name == "compact_cnn":
        if pretrained:
            raise ValueError("compact_cnn has no pretrained weights; set pretrained: false")
        return CompactCNN()
    from torchvision.models import get_model, get_model_weights

    weights = get_model_weights(name)[weights_version] if pretrained else None
    encoder = get_model(name, weights=weights)
    if name.startswith("resnet"):
        encoder.fc = nn.Identity()
    elif name == "densenet121":
        encoder.classifier = nn.Identity()
    elif name == "swin_t":
        encoder.head = nn.Identity()
    elif name == "vit_b_16":
        encoder.heads.head = nn.Identity()
    else:
        encoder.classifier[-1] = nn.Identity()
    return encoder


class RGBClassifier(nn.Module):
    """Classify a single crop, or average features across specified crop scales."""

    rgb_only = True
    use_stain = False
    use_morphology = False
    maturation_head = None

    def __init__(
        self,
        num_classes,
        backbone="resnet18",
        pretrained=False,
        factors=(1.0,),
        freeze_backbone=False,
        dropout=0.2,
        weights_version="IMAGENET1K_V1",
    ):
        super().__init__()
        if backbone not in MODEL_CATALOG:
            raise ValueError(f"Unknown RGB backbone {backbone!r}; choose {list(MODEL_CATALOG)}")
        if num_classes < 2 or not factors or any(f <= 0 for f in factors):
            raise ValueError("Require >=2 classes and nonempty positive crop factors")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.name = backbone
        self.factors = tuple(factors)
        self.freeze_backbone = freeze_backbone
        self.backbone = make_encoder(backbone, pretrained, weights_version)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(MODEL_CATALOG[backbone]["features"], num_classes)
        )
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # Frozen BatchNorm statistics and dropout remain frozen too.
        return self

    def forward(self, crops, validity=None, extents=None):
        if crops.ndim != 5 or crops.shape[1] != len(self.factors) or crops.shape[2] != 3:
            raise ValueError("Expected crops shaped [batch, configured scales, 3, height, width]")
        batch, scales, channels, height, width = crops.shape
        x = crops.reshape(batch * scales, channels, height, width)
        features = self.backbone((x - self.image_mean) / self.image_std)
        tokens = features.reshape(batch, scales, -1)
        return {"logits": self.head(tokens.mean(1)), "tokens": tokens}
