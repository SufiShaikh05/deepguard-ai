"""Uses a temporary folder of tiny random images instead of the real dataset."""

import numpy as np
import pytest
from PIL import Image

from data.dataset import DeepfakeDataset


def _write_dummy_image(path) -> None:
    array = np.random.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_dataset_indexes_real_and_fake_folders(tmp_path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "fake").mkdir()
    _write_dummy_image(tmp_path / "real" / "one.jpg")
    _write_dummy_image(tmp_path / "fake" / "one.jpg")
    _write_dummy_image(tmp_path / "fake" / "two.jpg")

    dataset = DeepfakeDataset(str(tmp_path), image_size=32)
    assert len(dataset) == 3

    rgb_tensor, freq_tensor, label = dataset[0]
    assert rgb_tensor.shape == (3, 32, 32)
    assert freq_tensor.shape == (3, 32, 32)
    assert label in (0, 1)


def test_dataset_raises_clear_error_when_empty(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        DeepfakeDataset(str(tmp_path), image_size=32)
