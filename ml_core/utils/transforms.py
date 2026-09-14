"""
Shared preprocessing for both streams of DeepGuard AI.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image

# ImageNet statistics - required because the spatial stream starts from
# ImageNet-pretrained weights (Xception / EfficientNet via timm).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_image(image: Image.Image, image_size: int) -> Image.Image:
    """Resizes to a square (image_size, image_size).

    Deepfake datasets are almost always pre-cropped to just the face, so
    a direct square resize is simpler and more predictable here than
    aspect-ratio-preserving letterboxing.
    """
    return image.resize((image_size, image_size), Image.Resampling.BILINEAR)


def apply_geometric_augmentation(image: Image.Image) -> Image.Image:
    """Light, label-preserving augmentation for training only."""
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if random.random() < 0.3:
        angle = random.uniform(-10, 10)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

    return image


def normalize_rgb(image_array: np.ndarray) -> np.ndarray:
    """Scales pixels to [0, 1], then applies ImageNet mean/std normalization."""
    scaled = image_array.astype(np.float32) / 255.0
    return (scaled - IMAGENET_MEAN) / IMAGENET_STD


def to_chw_tensor_array(image_array: np.ndarray) -> np.ndarray:
    """Converts an (H, W, C) array into PyTorch's expected (C, H, W) layout."""
    return np.ascontiguousarray(np.transpose(image_array, (2, 0, 1)))
