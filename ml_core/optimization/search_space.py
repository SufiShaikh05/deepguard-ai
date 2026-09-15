"""
Defines the hyperparameter genome and encode/decode routines for DualStreamNet.
"""

from __future__ import annotations

from typing import Any
import numpy as np

SPATIAL_BACKBONES: list[str] = ["xception", "efficientnet_b0", "efficientnet_b1"]
FREQUENCY_FEATURE_DIMS: list[int] = [64, 128, 256, 512]
FUSION_HIDDEN_DIMS: list[int] = [64, 128, 256, 512]
DROPOUT_BOUNDS: tuple[float, float] = (0.1, 0.5)

MODEL_CONFIG_KEYS: tuple[str, ...] = (
    "spatial_backbone",
    "spatial_pretrained",
    "freeze_spatial_backbone",
    "frequency_in_channels",
    "frequency_feature_dim",
    "fusion_hidden_dim",
    "dropout",
    "num_classes",
)


def get_search_space_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Returns lower and upper bounds (xl, xu) for the continuous/index genome vector."""
    xl = np.array([0.0, 0.0, 0.0, DROPOUT_BOUNDS[0]], dtype=float)
    xu = np.array(
        [
            float(len(SPATIAL_BACKBONES) - 1),
            float(len(FREQUENCY_FEATURE_DIMS) - 1),
            float(len(FUSION_HIDDEN_DIMS) - 1),
            DROPOUT_BOUNDS[1],
        ],
        dtype=float,
    )
    return xl, xu


def decode_genome(genome: np.ndarray | list[float] | dict[str, Any]) -> dict[str, Any]:
    """Decodes a genome representation into a model config dict.

    The returned dictionary contains EXACTLY the keys expected by
    DualStreamNet.from_config()'s model section:
      - spatial_backbone
      - spatial_pretrained
      - freeze_spatial_backbone
      - frequency_in_channels
      - frequency_feature_dim
      - fusion_hidden_dim
      - dropout
      - num_classes
    """
    if isinstance(genome, dict):
        # Extract fields directly or from indices
        if "spatial_backbone" in genome:
            backbone = str(genome["spatial_backbone"])
        else:
            b_idx = int(np.clip(round(float(genome.get("backbone_idx", 0))), 0, len(SPATIAL_BACKBONES) - 1))
            backbone = SPATIAL_BACKBONES[b_idx]

        if "frequency_feature_dim" in genome:
            freq_dim = int(genome["frequency_feature_dim"])
        else:
            f_idx = int(np.clip(round(float(genome.get("freq_idx", 2))), 0, len(FREQUENCY_FEATURE_DIMS) - 1))
            freq_dim = FREQUENCY_FEATURE_DIMS[f_idx]

        if "fusion_hidden_dim" in genome:
            fusion_dim = int(genome["fusion_hidden_dim"])
        else:
            h_idx = int(np.clip(round(float(genome.get("fusion_idx", 2))), 0, len(FUSION_HIDDEN_DIMS) - 1))
            fusion_dim = FUSION_HIDDEN_DIMS[h_idx]

        dropout = float(np.clip(float(genome.get("dropout", 0.3)), DROPOUT_BOUNDS[0], DROPOUT_BOUNDS[1]))
    else:
        g = np.asarray(genome, dtype=float).flatten()
        b_idx = int(np.clip(round(g[0]), 0, len(SPATIAL_BACKBONES) - 1))
        f_idx = int(np.clip(round(g[1]), 0, len(FREQUENCY_FEATURE_DIMS) - 1))
        h_idx = int(np.clip(round(g[2]), 0, len(FUSION_HIDDEN_DIMS) - 1))
        dropout = float(np.clip(g[3], DROPOUT_BOUNDS[0], DROPOUT_BOUNDS[1]))

        backbone = SPATIAL_BACKBONES[b_idx]
        freq_dim = FREQUENCY_FEATURE_DIMS[f_idx]
        fusion_dim = FUSION_HIDDEN_DIMS[h_idx]

    decoded = {
        "spatial_backbone": backbone,
        "spatial_pretrained": True,
        "freeze_spatial_backbone": True,
        "frequency_in_channels": 3,
        "frequency_feature_dim": freq_dim,
        "fusion_hidden_dim": fusion_dim,
        "dropout": float(round(dropout, 4)),
        "num_classes": 2,
    }

    # Verify exact keys
    assert tuple(decoded.keys()) == MODEL_CONFIG_KEYS, (
        f"Decoded config keys mismatch. Expected {MODEL_CONFIG_KEYS}, got {tuple(decoded.keys())}"
    )

    return decoded


def encode_genome(model_config: dict[str, Any]) -> np.ndarray:
    """Encodes a model config dictionary into the genome vector."""
    backbone = model_config.get("spatial_backbone", SPATIAL_BACKBONES[0])
    if backbone not in SPATIAL_BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Supported: {SPATIAL_BACKBONES}")
    b_idx = float(SPATIAL_BACKBONES.index(backbone))

    freq_dim = int(model_config.get("frequency_feature_dim", 256))
    if freq_dim not in FREQUENCY_FEATURE_DIMS:
        # snap to closest
        freq_dim = min(FREQUENCY_FEATURE_DIMS, key=lambda x: abs(x - freq_dim))
    f_idx = float(FREQUENCY_FEATURE_DIMS.index(freq_dim))

    fusion_dim = int(model_config.get("fusion_hidden_dim", 256))
    if fusion_dim not in FUSION_HIDDEN_DIMS:
        fusion_dim = min(FUSION_HIDDEN_DIMS, key=lambda x: abs(x - fusion_dim))
    h_idx = float(FUSION_HIDDEN_DIMS.index(fusion_dim))

    dropout = float(np.clip(float(model_config.get("dropout", 0.3)), DROPOUT_BOUNDS[0], DROPOUT_BOUNDS[1]))

    return np.array([b_idx, f_idx, h_idx, dropout], dtype=float)
