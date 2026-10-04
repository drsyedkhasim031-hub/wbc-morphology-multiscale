"""Explicit comparator adapters. Spatial maps are resized to 28x28 at 224 input.

These branch assignments complete missing comparator details in the manuscript;
they are declared implementation choices, not recovered experimental code.
"""

from torch import nn
from torch.nn import functional as F
from torchvision import models


class TorchvisionBackbone(nn.Module):
    def __init__(self, name, pretrained):
        super().__init__()
        builders = {
            "resnet50": models.resnet50,
            "mobilenet_v3_large": models.mobilenet_v3_large,
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_v2_s": models.efficientnet_v2_s,
            "swin_t": models.swin_t,
        }
        if name not in builders:
            raise ValueError(f"Unknown backbone: {name}")
        net = builders[name](weights="DEFAULT" if pretrained else None)
        self.name = name
        if name == "resnet50":
            self.features = nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
            )
            self.local_index, self.local_channels, self.global_channels = 5, 512, 2048
        else:
            self.features = net.features
            self.local_index, self.local_channels, self.global_channels = {
                "mobilenet_v3_large": (6, 40, 960),
                "efficientnet_b0": (3, 40, 1280),
                "efficientnet_v2_s": (3, 64, 1280),
                "swin_t": (3, 192, 768),
            }[name]
        from torchvision.ops import StochasticDepth

        for module in self.modules():
            if isinstance(module, StochasticDepth):
                module.p = 0.0

    def forward(self, x):
        target = (x.shape[-2] // 8, x.shape[-1] // 8)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.local_index:
                local = x.permute(0, 3, 1, 2) if self.name == "swin_t" else x
        if self.name == "swin_t":
            x = x.permute(0, 3, 1, 2)
        return F.interpolate(local, target, mode="bilinear", align_corners=False), x
