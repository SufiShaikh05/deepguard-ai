"""Tests for the frequency math - the part most likely to break
silently if someone edits it later."""

import numpy as np

from utils.dct_utils import image_to_dct, normalize_dct


def test_image_to_dct_preserves_shape() -> None:
    image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    assert image_to_dct(image).shape == image.shape


def test_image_to_dct_handles_odd_dimensions() -> None:
    # 65x65 is exactly what would break cv2.dct without the padding
    # fix inside _dct_2d - this test exists to catch that regression.
    image = np.random.randint(0, 255, size=(65, 65, 3), dtype=np.uint8)
    assert image_to_dct(image).shape == image.shape


def test_normalize_dct_stays_within_unit_range() -> None:
    image = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    normalized = normalize_dct(image_to_dct(image))
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
