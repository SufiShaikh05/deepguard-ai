from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_CORE_DIR = PROJECT_ROOT / "ml_core"
sys.path.insert(0, str(ML_CORE_DIR))

from utils.constants import CLASS_NAMES
from utils.dct_utils import image_to_dct, normalize_dct
from utils.transforms import normalize_rgb, resize_image, to_chw_tensor_array


MODEL_PATH = PROJECT_ROOT / "backend" / "app" / "ml_assets" / "model.onnx"
CONFIG_PATH = ML_CORE_DIR / "configs" / "config.yaml"
TEST_DIR = ML_CORE_DIR / "data" / "processed" / "test"


def jpeg_reencode(image: Image.Image, quality: int) -> Image.Image:
    """Encode and decode an image in memory; never modify the source file."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def resize_down_then_restore(image: Image.Image) -> Image.Image:
    """Downsample each dimension by half, then restore original dimensions."""
    width, height = image.size
    small_size = (max(1, width // 2), max(1, height // 2))
    small = image.resize(small_size, Image.Resampling.BILINEAR)
    return small.resize((width, height), Image.Resampling.BILINEAR)


TRANSFORMS = [
    ("original", lambda image: image.copy()),
    ("jpeg_q75", lambda image: jpeg_reencode(image, 75)),
    ("jpeg_q50", lambda image: jpeg_reencode(image, 50)),
    ("resize_half_restore", resize_down_then_restore),
    ("brightness_0.8", lambda image: ImageEnhance.Brightness(image).enhance(0.8)),
    ("contrast_0.8", lambda image: ImageEnhance.Contrast(image).enhance(0.8)),
    ("saturation_0.7", lambda image: ImageEnhance.Color(image).enhance(0.7)),
    (
        "gaussian_smoothing_r1",
        lambda image: image.filter(ImageFilter.GaussianBlur(radius=1.0)),
    ),
    ("sharpen_2x", lambda image: ImageEnhance.Sharpness(image).enhance(2.0)),
]

TRANSFORM_DESCRIPTIONS = {
    "original": "Unmodified source image.",
    "jpeg_q75": "JPEG re-encoding at quality 75, chroma subsampling 4:2:0.",
    "jpeg_q50": "JPEG re-encoding at quality 50, chroma subsampling 4:2:0.",
    "resize_half_restore": "Downsample each dimension by half and resize back.",
    "brightness_0.8": "Brightness enhancement factor 0.8.",
    "contrast_0.8": "Contrast enhancement factor 0.8.",
    "saturation_0.7": "Color saturation enhancement factor 0.7.",
    "gaussian_smoothing_r1": (
        "Gaussian blur radius 1.0; a synthetic smoothing proxy, "
        "not a reproduction of a specific phone beauty filter."
    ),
    "sharpen_2x": "Sharpness enhancement factor 2.0.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_image(image: Image.Image, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Match the backend's shared RGB and DCT preprocessing."""
    resized = resize_image(image.convert("RGB"), image_size)
    resized_array = np.asarray(resized, dtype=np.uint8)

    rgb = to_chw_tensor_array(normalize_rgb(resized_array)).astype(np.float32)
    dct = normalize_dct(image_to_dct(resized_array))
    freq = to_chw_tensor_array(dct).astype(np.float32)

    return rgb, freq


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def list_samples(limit_per_class: int | None, seed: int):
    samples = []
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = TEST_DIR / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Test class directory not found: {class_dir}")

        paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        )

        if not paths:
            raise RuntimeError(f"No images found in {class_dir}")

        if limit_per_class is not None and len(paths) > limit_per_class:
            chooser = random.Random(seed + label)
            paths = sorted(chooser.sample(paths, limit_per_class))

        print(f"Selected {len(paths)} {class_name} images.")
        samples.extend((path, label, class_name) for path in paths)

    return samples


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via average ranks, including ties; positive class is fake."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1

        # Ranks are one-based; 'end' is the last rank in this tied group.
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    rank_sum = ranks[labels == 1].sum()
    return float(
        (rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_transform(name: str, rows: list[dict]) -> dict:
    labels = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["predicted_label_id"] for row in rows], dtype=np.int64)
    p_fake = np.asarray([row["p_fake"] for row in rows], dtype=np.float64)
    latency = np.asarray([row["onnx_inference_ms"] for row in rows], dtype=np.float64)
    delta = np.asarray(
        [row["p_fake_delta_vs_original"] for row in rows], dtype=np.float64
    )
    flips = np.asarray(
        [row["prediction_flip_vs_original"] for row in rows], dtype=np.int64
    )

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))

    precision_fake = safe_divide(tp, tp + fp)
    recall_fake = safe_divide(tp, tp + fn)
    f1_fake = safe_divide(
        2 * precision_fake * recall_fake, precision_fake + recall_fake
    )

    real_mask = labels == 0
    fake_mask = labels == 1

    return {
        "transform": name,
        "n_images": len(rows),
        "accuracy": safe_divide(tp + tn, len(rows)),
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "specificity_real": safe_divide(tn, tn + fp),
        "false_positive_rate_real": safe_divide(fp, tn + fp),
        "false_negative_rate_fake": safe_divide(fn, tp + fn),
        "roc_auc_fake": rank_auc(labels, p_fake),
        "true_positives_fake": tp,
        "false_negatives_fake": fn,
        "true_negatives_real": tn,
        "false_positives_real": fp,
        "mean_p_fake_on_real": float(p_fake[real_mask].mean()),
        "mean_p_fake_on_fake": float(p_fake[fake_mask].mean()),
        "mean_abs_p_fake_shift_real": float(np.abs(delta[real_mask]).mean()),
        "mean_abs_p_fake_shift_fake": float(np.abs(delta[fake_mask]).mean()),
        "prediction_flip_rate_vs_original": float(flips.mean()),
        "median_onnx_inference_ms": float(np.median(latency)),
        "p95_onnx_inference_ms": float(np.percentile(latency, 95)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired robustness benchmark for DeepGuard AI's ONNX model."
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional deterministic pilot size per class; omit for all images.",
    )
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument(
        "--output-dir",
        default="ml_core/experiments/results/robustness_v1",
        help="Output directory, absolute or relative to the project root.",
    )
    args = parser.parse_args()

    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be at least 1.")

    for required_path in (MODEL_PATH, CONFIG_PATH, TEST_DIR):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    image_size = int(config["data"]["image_size"])
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {"rgb", "freq"}:
        raise RuntimeError(f"Unexpected ONNX input names: {list(inputs)}")

    expected_shape = [1, 3, image_size, image_size]
    for name in ("rgb", "freq"):
        if inputs[name].shape != expected_shape:
            raise RuntimeError(
                f"ONNX input {name} has shape {inputs[name].shape}; "
                f"expected {expected_shape}. Stop rather than silently reshape."
            )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list_samples(args.limit_per_class, args.seed)
    if not samples:
        raise RuntimeError("No samples selected.")

    print(f"Model: {MODEL_PATH}")
    print(f"Image size: {image_size} x {image_size}")
    print(f"ONNX providers: {session.get_providers()}")
    print(f"Source images selected: {len(samples)}")
    print(f"Transform conditions: {len(TRANSFORMS)}")
    print("Warming up ONNX Runtime...")

    # Warm-up once so initialization overhead is not included in measured latency.
    with Image.open(samples[0][0]) as opened:
        warm_rgb, warm_freq = prepare_image(opened.convert("RGB"), image_size)
    session.run(
        None,
        {"rgb": warm_rgb[None, ...], "freq": warm_freq[None, ...]},
    )

    rows = []
    progress_every = max(1, len(samples) // 10)

    for index, (path, true_label, class_name) in enumerate(samples, start=1):
        with Image.open(path) as opened:
            original = opened.convert("RGB")

        image_id = f"{class_name}/{path.name}"

        for transform_name, apply_transform in TRANSFORMS:
            transformed = apply_transform(original)
            rgb, freq = prepare_image(transformed, image_size)
            model_inputs = {"rgb": rgb[None, ...], "freq": freq[None, ...]}

            start = time.perf_counter()
            logits = session.run(None, model_inputs)[0][0]
            inference_ms = (time.perf_counter() - start) * 1000.0

            probabilities = softmax(logits)
            predicted_label = int(np.argmax(probabilities))

            rows.append({
                "image_id": image_id,
                "source_filename": path.name,
                "true_class": class_name,
                "true_label": int(true_label),
                "transform": transform_name,
                "predicted_class": CLASS_NAMES[predicted_label],
                "predicted_label_id": predicted_label,
                "correct": int(predicted_label == true_label),
                "confidence": float(probabilities[predicted_label]),
                "p_real": float(probabilities[0]),
                "p_fake": float(probabilities[1]),
                "onnx_inference_ms": float(inference_ms),
            })

        if index % progress_every == 0 or index == len(samples):
            print(f"Processed {index}/{len(samples)} source images.")

    # Pair each transformed result with that same source image's original prediction.
    baseline = {
        row["image_id"]: row
        for row in rows
        if row["transform"] == "original"
    }

    for row in rows:
        reference = baseline[row["image_id"]]
        row["p_fake_delta_vs_original"] = row["p_fake"] - reference["p_fake"]
        row["prediction_flip_vs_original"] = int(
            row["predicted_label_id"] != reference["predicted_label_id"]
        )

    summary = [
        summarize_transform(name, [r for r in rows if r["transform"] == name])
        for name, _ in TRANSFORMS
    ]

    write_csv(output_dir / "predictions.csv", rows)
    write_csv(output_dir / "summary_by_transform.csv", summary)

    metadata = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "project": "DeepGuard AI",
        "experiment": "Robustness Benchmark v1",
        "model_path": str(MODEL_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
        "config_path": str(CONFIG_PATH),
        "image_size": image_size,
        "class_names": CLASS_NAMES,
        "class_mapping": {"real": 0, "fake": 1},
        "providers": session.get_providers(),
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "pillow_version": Image.__version__,
        "limit_per_class": args.limit_per_class,
        "sampling_seed": args.seed,
        "selected_source_image_count": len(samples),
        "transform_condition_count": len(TRANSFORMS),
        "predictions_row_count": len(rows),
        "transform_descriptions": TRANSFORM_DESCRIPTIONS,
        "notes": [
            "Original source image files are read only and never modified.",
            "RGB normalization and DCT preprocessing reuse the existing ml_core utilities.",
            "Latency measures ONNX session.run only; preprocessing and transformations are excluded.",
            "Variants from the same source image are paired stress tests, not independent images.",
            "Gaussian smoothing is a synthetic proxy, not an actual mobile beautification filter.",
            "Use these results for diagnosis; confirm findings with an independent dataset.",
        ],
        "outputs": ["predictions.csv", "summary_by_transform.csv"],
    }

    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    for item in summary:
        print(
            f"{item['transform']:22s} "
            f"accuracy={item['accuracy']:.4f} "
            f"FPR_real={item['false_positive_rate_real']:.4f} "
            f"recall_fake={item['recall_fake']:.4f} "
            f"AUC={item['roc_auc_fake']:.4f} "
            f"flip_rate={item['prediction_flip_rate_vs_original']:.4f} "
            f"median_inference={item['median_onnx_inference_ms']:.2f}ms"
        )

    print(f"\nResults saved under: {output_dir}")
    print("No model files or original dataset images were modified.")


if __name__ == "__main__":
    main()
