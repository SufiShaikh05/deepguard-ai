"""
Stream 2: looks at the DCT representation instead of raw pixels.
"""

from __future__ import annotations

import torch
from torch import nn


class FrequencyStream(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            self._conv_block(in_channels, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, feature_dim),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return torch.flatten(x, start_dim=1)
