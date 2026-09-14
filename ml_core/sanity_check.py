"""
Fast, no-dataset-required check that the architecture is wired correctly.
Run with: python sanity_check.py
"""

from __future__ import annotations

import torch

from models.dual_stream_net import DualStreamNet

BATCH_SIZE = 4
IMAGE_SIZE = 224
NUM_CLASSES = 2


def main() -> None:
    print("Building DualStreamNet...")
    model = DualStreamNet(spatial_pretrained=False, num_classes=NUM_CLASSES)
    model.eval()

    print(f"Spatial stream feature size: {model.spatial_stream.feature_dim}")
    print(f"Frequency stream feature size: {model.frequency_stream.feature_dim}")

    dummy_rgb = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    dummy_freq = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        logits = model(dummy_rgb, dummy_freq)

    expected_shape = (BATCH_SIZE, NUM_CLASSES)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {tuple(logits.shape)}"

    print(f"Output shape: {tuple(logits.shape)}  <- correct!")
    print("Sanity check passed. The architecture is wired together correctly.")


if __name__ == "__main__":
    main()
