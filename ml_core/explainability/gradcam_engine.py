"""
Generates Grad-CAM heatmaps explaining WHERE in a face image the
Dual-Stream model found evidence for its decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.dual_stream_net import DualStreamNet
from utils.constants import CLASS_NAMES


@dataclass
class GradCAMResult:
    heatmap: np.ndarray               # (H, W) float32 in [0, 1]
    predicted_class: int              # 0 = real, 1 = fake
    confidence: float                 # softmax probability of predicted_class
    probabilities: dict[str, float]   # e.g. {"real": 0.12, "fake": 0.88}


class _SpatialStreamView(nn.Module):
    """
    pytorch-grad-cam expects a model whose forward() takes exactly ONE
    tensor. This wrapper looks like a single-input model to Grad-CAM,
    while secretly still feeding both streams underneath - holding the
    frequency tensor for the CURRENT image fixed while Grad-CAM varies
    only the RGB input to compute gradients.
    """

    def __init__(self, dual_stream_model: DualStreamNet) -> None:
        super().__init__()
        self.dual_stream_model = dual_stream_model
        self._fixed_freq: torch.Tensor | None = None

    def set_frequency_input(self, freq_tensor: torch.Tensor) -> None:
        self._fixed_freq = freq_tensor

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if self._fixed_freq is None:
            raise RuntimeError("Call set_frequency_input() before running Grad-CAM.")

        batch_size = rgb.shape[0]
        freq = self._fixed_freq.expand(batch_size, -1, -1, -1)
        return self.dual_stream_model(rgb, freq)


def find_last_conv_layer(module: nn.Module) -> nn.Module:
    """
    Finds the last nn.Conv2d layer inside a module. Works for any CNN
    backbone without hardcoding architecture-specific layer names, so
    swapping spatial_backbone in config.yaml never breaks this.
    """
    conv_layers = [layer for layer in module.modules() if isinstance(layer, nn.Conv2d)]

    if not conv_layers:
        raise ValueError("No Conv2d layers found - cannot pick a Grad-CAM target layer.")

    return conv_layers[-1]


class GradCAMExplainer:
    """
    A reusable Grad-CAM engine around a trained DualStreamNet. Build
    one instance (e.g. once, when the backend starts up) and call
    .explain() for every uploaded image afterwards.
    """

    def __init__(self, model: DualStreamNet, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

        self._view = _SpatialStreamView(self.model)
        target_layer = find_last_conv_layer(self.model.spatial_stream.backbone)

        self.cam = GradCAM(model=self._view, target_layers=[target_layer])

    def explain(
        self,
        rgb_tensor: torch.Tensor,
        freq_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> GradCAMResult:
        """
        target_class=None (the default) explains whatever class the
        model actually predicted - pass 0 or 1 to force an explanation
        for "real" or "fake" specifically instead.
        """
        rgb_batch = rgb_tensor.unsqueeze(0).to(self.device)
        rgb_batch.requires_grad_(True)  # needed even if the backbone is frozen
        freq_batch = freq_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(rgb_batch, freq_batch)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())
        explain_class = predicted_class if target_class is None else target_class

        self._view.set_frequency_input(freq_batch)
        targets = [ClassifierOutputTarget(explain_class)]

        grayscale_cam = self.cam(input_tensor=rgb_batch, targets=targets)
        heatmap = grayscale_cam[0]  # drop batch dimension -> (H, W)

        return GradCAMResult(
            heatmap=heatmap,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities={
                CLASS_NAMES[0]: float(probabilities[0].item()),
                CLASS_NAMES[1]: float(probabilities[1].item()),
            },
        )
