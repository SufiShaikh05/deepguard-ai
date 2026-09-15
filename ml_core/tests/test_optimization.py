"""
Tests for ml_core.optimization module.

Includes genome encode/decode verification, from_config schema compatibility,
real-dataset guard behavior, and an isolated toy NSGA-II search (pipeline crash validation).
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.dataset import DeepfakeDataset
from models.dual_stream_net import DualStreamNet
from optimization.search_space import (
    SPATIAL_BACKBONES,
    FREQUENCY_FEATURE_DIMS,
    FUSION_HIDDEN_DIMS,
    MODEL_CONFIG_KEYS,
    decode_genome,
    encode_genome,
    get_search_space_bounds,
)
from optimization.run_nsga2 import (
    verify_real_dataset_presence,
    run_nsga2_optimization,
)


def _write_dummy_face(path: Path) -> None:
    """Creates a tiny 16x16 dummy image for test-only crash validation."""
    arr = np.random.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_genome_encode_decode_roundtrip():
    """Verify that configuration encoding and decoding round-trips accurately."""
    for b_idx, backbone in enumerate(SPATIAL_BACKBONES):
        for f_idx, freq_dim in enumerate(FREQUENCY_FEATURE_DIMS):
            original_cfg = {
                "spatial_backbone": backbone,
                "spatial_pretrained": True,
                "freeze_spatial_backbone": True,
                "frequency_in_channels": 3,
                "frequency_feature_dim": freq_dim,
                "fusion_hidden_dim": 256,
                "dropout": 0.25,
                "num_classes": 2,
            }

            encoded = encode_genome(original_cfg)
            assert encoded[0] == pytest.approx(b_idx)
            assert encoded[1] == pytest.approx(f_idx)

            decoded = decode_genome(encoded)
            assert decoded["spatial_backbone"] == backbone
            assert decoded["frequency_feature_dim"] == freq_dim
            assert decoded["dropout"] == pytest.approx(0.25, abs=1e-3)
            assert tuple(decoded.keys()) == MODEL_CONFIG_KEYS


def test_decoded_genome_instantiates_dual_stream_net():
    """Verify that decoded configs have EXACTLY the keys DualStreamNet.from_config expects."""
    xl, xu = get_search_space_bounds()
    # Sample arbitrary point inside bounds
    x_test = (xl + xu) / 2.0
    decoded = decode_genome(x_test)

    # Must contain exactly the model config keys
    assert tuple(decoded.keys()) == MODEL_CONFIG_KEYS

    # Turn off pretrained weights for fast offline unit testing
    decoded["spatial_pretrained"] = False
    model = DualStreamNet.from_config({"model": decoded})
    assert isinstance(model, DualStreamNet)
    assert model.frequency_stream.feature_dim == decoded["frequency_feature_dim"]


def test_real_dataset_presence_guard(tmp_path):
    """Verify that verify_real_dataset_presence detects missing/empty data directories."""
    empty_train = tmp_path / "train"
    empty_val = tmp_path / "val"
    empty_train.mkdir()
    empty_val.mkdir()

    cfg = {
        "data": {
            "train_dir": str(empty_train),
            "val_dir": str(empty_val),
        }
    }

    is_present, msg = verify_real_dataset_presence(cfg, base_dir=tmp_path)
    assert is_present is False
    assert "do not contain real" in msg


def test_toy_nsga2_execution(tmp_path):
    """Pipeline crash validation: runs a 2-candidate, 1-generation toy search.

    NOTE: This test uses temporary synthetic images ONLY to validate that
    the search loop, model training pass, ONNX export, and ONNX Runtime latency
    profiling wire together without crashing. The accuracy metrics produced here
    are purely test artifacts and are never used for production/research Pareto front.
    """
    # Create tiny test-only mock dataset (2 real, 2 fake)
    test_data_dir = tmp_path / "test_only_mock_dataset"
    train_dir = test_data_dir / "train"
    val_dir = test_data_dir / "val"

    for d in (train_dir / "real", train_dir / "fake", val_dir / "real", val_dir / "fake"):
        d.mkdir(parents=True)
        _write_dummy_face(d / "sample_1.jpg")
        _write_dummy_face(d / "sample_2.jpg")

    train_ds = DeepfakeDataset(str(train_dir), image_size=32, train=True)
    val_ds = DeepfakeDataset(str(val_dir), image_size=32, train=False)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    toy_config = {
        "data": {"image_size": 32},
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "early_stopping_patience": 2,
            "device": "cpu",
        },
    }

    results_dir = tmp_path / "toy_results"

    results = run_nsga2_optimization(
        config=toy_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        pop_size=2,
        n_gen=1,
        epochs=1,
        output_dir=results_dir,
        seed=123,
    )

    # Validate output structure
    assert len(results["all_evaluations"]) >= 2
    assert len(results["pareto_front"]) >= 1

    json_file = Path(results["json_path"])
    png_file = Path(results["png_path"])

    assert json_file.exists(), "pareto_front.json was not created"
    assert png_file.exists(), "pareto_front.png was not created"

    with open(json_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        assert "pareto_front" in saved_data
        assert "all_evaluations" in saved_data
