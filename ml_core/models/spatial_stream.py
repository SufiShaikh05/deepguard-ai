"""
Stream 1: looks at the image the way a normal computer vision model
does - raw RGB pixels - using a backbone that already learned strong
general-purpose visual features from ImageNet.
"""

from __future__ import annotations

import timm
import torch
from torch import nn


class SpatialStream(nn.Module):
    def __init__(
        self,
        backbone_name: str = "xception",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.feature_dim: int = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
