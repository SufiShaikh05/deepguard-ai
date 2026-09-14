"""The pytest version of sanity_check.py - runs automatically now."""

import torch

from models.dual_stream_net import DualStreamNet


def test_output_shape_matches_num_classes() -> None:
    model = DualStreamNet(spatial_pretrained=False, num_classes=2)
    model.eval()

    rgb = torch.randn(2, 3, 224, 224)
    freq = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits = model(rgb, freq)

    assert logits.shape == (2, 2)


def test_from_config_matches_manual_construction() -> None:
    config = {
        "model": {
            "spatial_backbone": "xception", "spatial_pretrained": False,
            "freeze_spatial_backbone": False, "frequency_in_channels": 3,
            "frequency_feature_dim": 128, "fusion_hidden_dim": 128,
            "dropout": 0.2, "num_classes": 2,
        }
    }
    model = DualStreamNet.from_config(config)
    assert model.frequency_stream.feature_dim == 128
