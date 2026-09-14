"""
Loads face images and prepares both tensors the Dual-Stream model needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.constants import CLASS_NAMES
from utils.dct_utils import image_to_dct, normalize_dct
from utils.transforms import (
    apply_geometric_augmentation,
    normalize_rgb,
    resize_image,
    to_chw_tensor_array,
)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class DeepfakeDataset(Dataset):
    """Returns (rgb_tensor, freq_tensor, label) for each sample.

    rgb_tensor  : FloatTensor, shape (3, H, W) - Stream 1 input
    freq_tensor : FloatTensor, shape (3, H, W) - Stream 2 input
    label       : int, 0 = real, 1 = fake
    """

    def __init__(self, root_dir: str, image_size: int = 224, train: bool = False) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.train = train
        self.samples: list[tuple[Path, int]] = self._index_samples()

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found under '{self.root_dir}'. Expected "
                f"'{self.root_dir}/real/' and '{self.root_dir}/fake/' "
                "subfolders containing images."
            )

    def _index_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []

        for label_index, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root_dir / class_name

            if not class_dir.is_dir():
                continue

            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    samples.append((file_path, label_index))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        file_path, label = self.samples[index]

        try:
            image = Image.open(file_path).convert("RGB")
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not read image at '{file_path}': {error}") from error

        image = resize_image(image, self.image_size)

        if self.train:
            image = apply_geometric_augmentation(image)

        image_array = np.array(image)  # (H, W, 3), uint8, 0-255

        rgb_array = normalize_rgb(image_array)
        rgb_tensor = torch.from_numpy(to_chw_tensor_array(rgb_array)).float()

        dct_array = normalize_dct(image_to_dct(image_array))
        freq_tensor = torch.from_numpy(to_chw_tensor_array(dct_array)).float()

        return rgb_tensor, freq_tensor, label
