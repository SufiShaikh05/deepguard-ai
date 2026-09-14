"""
Overlays a Grad-CAM heatmap onto the original photo using OpenCV.
Pure image processing - no PyTorch dependency, so this file can be
reused anywhere a heatmap array and an image are available.
"""

from __future__ import annotations

import cv2
import numpy as np


def blend_heatmap_with_image(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    """
    original_image : (H, W, 3) uint8, RGB, 0-255
    heatmap        : (H, W) float array, values in [0, 1] - straight
                     out of GradCAMExplainer.explain()
    Returns an (H, W, 3) uint8 RGB image ready to display or encode.
    """
    target_height, target_width = original_image.shape[:2]

    if heatmap.shape[:2] != (target_height, target_width):
        heatmap = cv2.resize(heatmap, (target_width, target_height))

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))

    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(original_image, 1 - alpha, heatmap_color_rgb, alpha, 0)
