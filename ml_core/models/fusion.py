"""
Combines both streams' opinions into one real/fake decision.
"""

from __future__ import annotations

import torch
from torch import nn


class FusionHead(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        frequency_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        combined_dim = spatial_dim + frequency_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, spatial_features: torch.Tensor, frequency_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([spatial_features, frequency_features], dim=1)
        return self.classifier(combined)
