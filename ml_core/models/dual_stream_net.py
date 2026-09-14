"""
The complete DeepGuard AI model: two streams feeding one fusion head.
"""

from __future__ import annotations

import torch
from torch import nn

from models.fusion import FusionHead
from models.frequency_stream import FrequencyStream
from models.spatial_stream import SpatialStream


class DualStreamNet(nn.Module):
    def __init__(
        self,
        spatial_backbone: str = "xception",
        spatial_pretrained: bool = True,
        freeze_spatial_backbone: bool = False,
        frequency_in_channels: int = 3,
        frequency_feature_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.spatial_stream = SpatialStream(
            backbone_name=spatial_backbone,
            pretrained=spatial_pretrained,
            freeze_backbone=freeze_spatial_backbone,
        )
        self.frequency_stream = FrequencyStream(
            in_channels=frequency_in_channels,
            feature_dim=frequency_feature_dim,
        )
        self.fusion = FusionHead(
            spatial_dim=self.spatial_stream.feature_dim,
            frequency_dim=self.frequency_stream.feature_dim,
            hidden_dim=fusion_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial_stream(rgb)
        frequency_features = self.frequency_stream(freq)
        return self.fusion(spatial_features, frequency_features)

    @classmethod
    def from_config(cls, config: dict) -> "DualStreamNet":
        """Builds the model directly from the parsed config.yaml dictionary."""
        model_cfg = config["model"]
        return cls(
            spatial_backbone=model_cfg["spatial_backbone"],
            spatial_pretrained=model_cfg["spatial_pretrained"],
            freeze_spatial_backbone=model_cfg["freeze_spatial_backbone"],
            frequency_in_channels=model_cfg["frequency_in_channels"],
            frequency_feature_dim=model_cfg["frequency_feature_dim"],
            fusion_hidden_dim=model_cfg["fusion_hidden_dim"],
            dropout=model_cfg["dropout"],
            num_classes=model_cfg["num_classes"],
        )
