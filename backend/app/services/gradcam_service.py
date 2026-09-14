"""
The full explainable path: PyTorch + Grad-CAM + OpenCV blend, packaged
as a ready-to-send response.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import torch

from explainability.gradcam_engine import GradCAMExplainer
from explainability.heatmap_blender import blend_heatmap_with_image
from models.dual_stream_net import DualStreamNet
from utils.constants import CLASS_NAMES


class GradCAMService:
    def __init__(self, model: DualStreamNet, device: torch.device) -> None:
        self.explainer = GradCAMExplainer(model, device)

    def predict_and_explain(
        self, rgb_tensor: torch.Tensor, freq_tensor: torch.Tensor, original_image: np.ndarray
    ) -> dict:
        result = self.explainer.explain(rgb_tensor, freq_tensor)
        blended = blend_heatmap_with_image(original_image, result.heatmap)

        return {
            "label": CLASS_NAMES[result.predicted_class],
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "heatmap_image": _encode_image_as_data_uri(blended),
        }


def _encode_image_as_data_uri(image_rgb: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", image_bgr)

    if not success:
        raise RuntimeError("Failed to encode the heatmap image as PNG.")

    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
