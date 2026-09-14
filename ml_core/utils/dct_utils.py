"""
Frequency-domain utilities for DeepGuard AI's Stream 2.
"""

from __future__ import annotations

import cv2
import numpy as np


def _dct_2d(channel: np.ndarray) -> np.ndarray:
    """Runs a 2D DCT on a single-channel float32 image.

    cv2.dct can reject odd-sized inputs, so we pad by one pixel on the
    bottom/right if needed and crop back afterwards.
    """
    height, width = channel.shape
    pad_bottom = height % 2
    pad_right = width % 2

    if pad_bottom or pad_right:
        channel = cv2.copyMakeBorder(channel, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)

    transformed = cv2.dct(channel)

    if pad_bottom or pad_right:
        transformed = transformed[:height, :width]

    return transformed


def image_to_dct(image_rgb: np.ndarray) -> np.ndarray:
    """Converts an (H, W, 3) RGB image (0-255) into its per-channel,
    log-scaled DCT representation, also shaped (H, W, 3).
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image_rgb.shape}")

    channels = cv2.split(image_rgb.astype(np.float32))
    dct_channels = [np.log1p(np.abs(_dct_2d(channel))) for channel in channels]

    return cv2.merge(dct_channels)


def normalize_dct(dct_image: np.ndarray) -> np.ndarray:
    """Rescales a log-scaled DCT image to [0, 1] per channel."""
    normalized = np.zeros_like(dct_image, dtype=np.float32)

    for c in range(dct_image.shape[2]):
        channel = dct_image[:, :, c]
        channel_min = float(channel.min())
        channel_max = float(channel.max())
        span = max(channel_max - channel_min, 1e-8)
        normalized[:, :, c] = (channel - channel_min) / span

    return normalized
