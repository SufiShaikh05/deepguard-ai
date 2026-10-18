from __future__ import annotations

import io
import random

from PIL import Image, ImageFilter


AUGMENTATION_PROBABILITY = 0.35


def apply_robustness_augmentation(
    image: Image.Image,
    probability: float = AUGMENTATION_PROBABILITY,
) -> tuple[Image.Image, str]:
    """Apply one optional, label-preserving image-quality perturbation.

    Apply this to the source image BEFORE the normal model resize.
    The function never writes to or modifies the original image file.

    Returns:
        A transformed RGB image and the operation name.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0.0 and 1.0.")

    result = image.convert("RGB")

    if random.random() >= probability:
        return result, "none"

    operation = random.choice(("jpeg", "blur", "downscale"))

    if operation == "jpeg":
        quality = random.randint(65, 95)
        buffer = io.BytesIO()

        result.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=2,
        )
        buffer.seek(0)

        with Image.open(buffer) as decoded:
            result = decoded.convert("RGB")

        return result, f"jpeg_q{quality}"

    if operation == "blur":
        radius = random.uniform(0.35, 0.75)
        result = result.filter(ImageFilter.GaussianBlur(radius=radius))
        return result, f"gaussian_blur_r{radius:.3f}"

    ratio = random.uniform(0.90, 0.99)
    width, height = result.size

    reduced_size = (
        max(1, round(width * ratio)),
        max(1, round(height * ratio)),
    )

    reduced = result.resize(
        reduced_size,
        Image.Resampling.BILINEAR,
    )
    result = reduced.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    return result, f"downscale_{ratio:.3f}"
