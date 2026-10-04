"""Shared ConvNeXt-T, compartment pooling, explicit geometry and scale Transformer."""

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .morphology import descriptors
from .stain import decompose

OFFICIAL_WEIGHTS = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        y = x.permute(0, 2, 3, 1)
        return F.layer_norm(y, (x.shape[1],), self.weight, self.bias, 1e-6).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.dwconv = nn.Conv2d(width, width, 7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pwconv1 = nn.Linear(width, 4 * width)
        self.pwconv2 = nn.Linear(4 * width, width)
        self.gamma = nn.Parameter(torch.full((width,), 1e-6))

    def forward(self, x):
        y = self.norm(self.dwconv(x).permute(0, 2, 3, 1))
        y = self.pwconv2(F.gelu(self.pwconv1(y))) * self.gamma
        return x + y.permute(0, 3, 1, 2)


class ConvNeXt(nn.Module):
    """Architecture-compatible encoder; no classifier, final norm, or stochastic depth."""

    def __init__(self, tiny_debug=False, pretrained=False):
        super().__init__()
        widths = [16, 32, 64, 128] if tiny_debug else [96, 192, 384, 768]
        depths = [1, 1, 1, 1] if tiny_debug else [3, 3, 9, 3]
        self.local_channels, self.global_channels = widths[1], widths[3]
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(3, widths[0], 4, stride=4), ChannelNorm(widths[0]))]
        )
        self.downsample_layers.extend(
            [
                nn.Sequential(ChannelNorm(widths[i - 1]), nn.Conv2d(widths[i - 1], widths[i], 2, stride=2))
                for i in range(1, 4)
            ]
        )
        self.stages = nn.ModuleList(
            [nn.Sequential(*[ConvNeXtBlock(w) for _ in range(d)]) for w, d in zip(widths, depths)]
        )
        if pretrained:
            if tiny_debug:
                raise ValueError("Debug backbone has no pretrained weights")
            state = torch.hub.load_state_dict_from_url(
                OFFICIAL_WEIGHTS, map_location="cpu", check_hash=False, weights_only=True
            )["model"]
            state = {k: v for k, v in state.items() if k.startswith(("downsample_layers.", "stages."))}
            self.load_state_dict(state, strict=True)

    def forward(self, x):
        local = None
        for i, (down, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            x = stage(down(x))
            if i == 1:
                local = x
        return local, x


class WBCModel(nn.Module):
    def __init__(
        self,
        num_classes,
        factors=(0.8, 1.0, 1.2),
        backbone="convnext_tiny",
        pretrained=False,
        prototypes=12,
        scale_encoding=True,
        stain=True,
        morphology=True,
        maturation=True,
        seed=1729,
        rgb_only=False,
        use_descriptors=True,
        mask_grid=None,
    ):
        super().__init__()
        self.factors = list(factors)
        self.use_stain, self.use_morphology = stain, morphology
        self.use_scale_encoding, self.rgb_only = scale_encoding, rgb_only
        self.use_descriptors, self.mask_grid = use_descriptors, mask_grid
        if backbone in ("convnext_tiny", "debug"):
            self.backbone = ConvNeXt(backbone == "debug", pretrained)
        else:
            from .backbones import TorchvisionBackbone

            self.backbone = TorchvisionBackbone(backbone, pretrained)
        lc, gc = self.backbone.local_channels, self.backbone.global_channels
        self.spatial = nn.Conv2d(2 * lc, 64, 1)
        rng = np.random.Generator(np.random.PCG64(seed))
        proto = rng.normal(size=(3, prototypes, 64)).astype(np.float32)
        proto /= np.linalg.norm(proto, axis=-1, keepdims=True)
        self.prototypes = nn.Parameter(torch.from_numpy(proto))
        self.appearance = nn.Sequential(nn.Linear(2 * gc + 192, 512), nn.GELU())
        self.morphology_embed = nn.Sequential(nn.Linear(8, 128), nn.GELU())
        self.projection = nn.Linear(640, 384)
        self.scale_embeddings = nn.Parameter(torch.empty(1, len(factors), 384))
        self.cls = nn.Parameter(torch.empty(1, 1, 384))
        block = nn.TransformerEncoderLayer(
            384, 6, 1536, 0.1, activation="gelu", batch_first=True, norm_first=False
        )
        self.transformer = nn.ModuleList([copy.deepcopy(block) for _ in range(2)])
        self.classifier = nn.Sequential(
            nn.Linear(384, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, num_classes)
        )
        self.maturation_head = (
            nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 1)) if maturation else None
        )
        self.rgb_classifier = nn.Linear(gc, num_classes) if rgb_only else None
        self.register_buffer("basis", torch.tensor([[0.7, 0.2], [0.65, 0.7], [0.25, 0.68]]))
        self.basis /= self.basis.norm(dim=0)
        self.register_buffer("descriptor_mean", torch.zeros(8))
        self.register_buffer("descriptor_std", torch.ones(8))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for name, module in self.named_modules():
            if name.startswith("backbone"):
                continue
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.MultiheadAttention):
                # Each of Q/K/V contains six independent 384 -> 64 head maps.
                head_width = module.embed_dim // module.num_heads
                for start in range(0, 3 * module.embed_dim, head_width):
                    nn.init.xavier_uniform_(module.in_proj_weight[start : start + head_width])
                nn.init.zeros_(module.in_proj_bias)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.scale_embeddings, std=0.02)

    def forward(self, crops, validity, extents):
        batch, scales, _, height, width = crops.shape
        if scales != len(self.factors):
            raise ValueError("Crop count differs from checkpoint configuration")
        if self.rgb_only:
            canonical = min(range(scales), key=lambda i: abs(self.factors[i] - 1))
            _, global_map = self.backbone((crops[:, canonical] - self.image_mean) / self.image_std)
            logits = self.rgb_classifier(global_map.mean((-2, -1)))
            return {
                "logits": logits,
                "maturation": logits[:, 0] * 0,
                "tokens": logits[:, None],
                "raw_descriptors": None,
            }
        flat = crops.reshape(batch * scales, 3, height, width)
        streams = decompose(flat, self.basis) if self.use_stain else flat[:, None].expand(-1, 2, -1, -1, -1)
        local, global_map = self.backbone(
            (streams.reshape(-1, 3, height, width) - self.image_mean) / self.image_std
        )
        gh, gw = local.shape[-2:]
        local = local.reshape(batch * scales, 2, -1, gh, gw).flatten(1, 2)
        feature = self.spatial(local)
        if self.mask_grid is not None:
            feature = F.interpolate(
                feature, (self.mask_grid, self.mask_grid), mode="bilinear", align_corners=False
            )
            gh = gw = self.mask_grid
        scores = torch.einsum(
            "bdhw,kpd->bkphw",
            F.normalize(feature, dim=1, eps=1e-8),
            F.normalize(self.prototypes, dim=-1, eps=1e-8),
        )
        masks = (scores.max(2).values / 0.1).softmax(1)
        v = 1 - F.adaptive_max_pool2d(1 - validity.reshape(batch * scales, 1, height, width), (gh, gw))
        weights = masks * v
        mass = weights.sum((-2, -1))
        pooled = torch.einsum("bkhw,bdhw->bkd", weights, feature) / mass.clamp_min(1e-8)[..., None]
        pooled = torch.where((mass >= 1e-8)[..., None], pooled, 0.0)
        raw = None
        if self.use_morphology and self.use_descriptors:
            hard = masks.detach().argmax(1).cpu().numpy()
            val = v.detach()[:, 0].cpu().numpy().astype(bool)
            ex = extents.detach().reshape(-1, 2).cpu().numpy()
            raw = torch.as_tensor(
                np.stack([descriptors(a, b, c) for a, b, c in zip(hard, val, ex)]), device=crops.device
            )
            standardized = torch.nan_to_num((raw - self.descriptor_mean) / self.descriptor_std, nan=0.0)
            morph = self.morphology_embed(standardized)
        else:
            morph = torch.zeros(batch * scales, 128, device=crops.device)
        if not self.use_morphology:
            pooled = pooled * 0
        global_vec = global_map.mean((-2, -1)).reshape(batch * scales, -1)
        appearance = self.appearance(torch.cat([global_vec, pooled.flatten(1)], -1))
        tokens = self.projection(torch.cat([appearance, morph], -1)).reshape(batch, scales, 384)
        sequence = tokens + self.scale_embeddings if self.use_scale_encoding else tokens
        sequence = torch.cat([self.cls.expand(batch, -1, -1), sequence], 1)
        for layer in self.transformer:
            sequence = layer(sequence)
        h = sequence[:, 0]
        return {
            "logits": self.classifier(h),
            "maturation": self.maturation_head(h).squeeze(-1) if self.maturation_head else h[:, 0] * 0,
            "tokens": tokens,
            "masks": masks.reshape(batch, scales, 3, gh, gw),
            "validity_grid": v.reshape(batch, scales, gh, gw),
            "raw_descriptors": raw,
        }
