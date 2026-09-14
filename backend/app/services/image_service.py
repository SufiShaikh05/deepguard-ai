"""
Validates uploaded files and prepares the tensors both models expect.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.utils.exceptions import InvalidImageError
from utils.dct_utils import image_to_dct, normalize_dct
from utils.transforms import normalize_rgb, resize_image, to_chw_tensor_array

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


async def read_and_validate_image(file: UploadFile) -> np.ndarray:
    """Returns the upload at its ORIGINAL resolution as (H, W, 3) uint8 RGB."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(
            f"Unsupported file type '{file.content_type}'. Please upload a JPEG, PNG, or WEBP image."
        )

    raw_bytes = await file.read()

    if len(raw_bytes) == 0:
        raise InvalidImageError("The uploaded file is empty.")

    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise InvalidImageError(
            f"File is too large ({len(raw_bytes) / 1024 / 1024:.1f} MB). "
            f"The limit is {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB."
        )

    try:
        Image.open(io.BytesIO(raw_bytes)).verify()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")  # re-open: verify() consumes the file
    except UnidentifiedImageError as error:
        raise InvalidImageError(
            "This file isn't a readable image. If you uploaded a PDF or another "
            "document format, please convert it to JPEG or PNG first."
        ) from error
    except Exception as error:  # noqa: BLE001 - any other decode failure lands here too
        raise InvalidImageError(f"Could not process the uploaded image: {error}") from error

    return np.array(image)


def preprocess_for_model(original_image: np.ndarray, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the model's input tensors from a RESIZED copy - the original stays untouched."""
    pil_image = Image.fromarray(original_image)
    resized_array = np.array(resize_image(pil_image, image_size))

    rgb_tensor = torch.from_numpy(to_chw_tensor_array(normalize_rgb(resized_array))).float()

    dct_array = normalize_dct(image_to_dct(resized_array))
    freq_tensor = torch.from_numpy(to_chw_tensor_array(dct_array)).float()

    return rgb_tensor, freq_tensor
