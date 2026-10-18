from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from robustness_benchmark import (
    PROJECT_ROOT,
    ML_CORE_DIR,
    MODEL_PATH,
    CONFIG_PATH,
    prepare_image,
    softmax,
    rank_auc,
    safe_divide,
    sha256_file,
    write_csv,
)
from utils.constants import CLASS_NAMES

VALIDATION_DIR = ML_CORE_DIR / "data" / "processed" / "val"


def make_jpeg_transform(quality: int):
    def apply(image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.convert("RGB").save(
            buffer, format="JPEG", quality=quality, subsampling=2
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")
    return apply


def make_blur_transform(radius: float):
    def apply(image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    return apply


def make_downscale_transform(ratio: float):
    def apply(image: Image.Image) -> Image.Image:
        width, height = image.size
        reduced_size = (
            max(1, round(width * ratio)),
            max(1, round(height * ratio)),
        )
        reduced = image.resize(
            reduced_size, Image.Resampling.BILINEAR
        )
        return reduced.resize(
            (width, height), Image.Resampling.BILINEAR
        )
    return apply


def build_transforms():
    transforms = [("original", lambda image: image.copy())]

    for quality in (95, 85, 75, 65, 50, 30):
        transforms.append((
            f"jpeg_quality_{quality}",
            make_jpeg_transform(quality),
        ))

    for radius in (0.25, 0.5, 0.75, 1.0, 1.5):
        transforms.append((
            f"gaussian_blur_r{radius:g}",
            make_blur_transform(radius),
        ))

    for ratio in (0.90, 0.75, 0.50, 0.25):
        transforms.append((
            f"downscale_{ratio:.2f}_restore",
            make_downscale_transform(ratio),
        ))

    return transforms


def collect_samples(limit_per_class: int | None, seed: int):
    samples = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for label, class_name in enumerate(CLASS_NAMES):
        folder = VALIDATION_DIR / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Validation folder missing: {folder}")

        paths = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        if limit_per_class is not None and len(paths) > limit_per_class:
            rng = random.Random(seed + label)
            paths = sorted(rng.sample(paths, limit_per_class))

        if not paths:
            raise RuntimeError(f"No validation images in {folder}")

        print(f"Selected {len(paths)} {class_name} images.")
        samples.extend((p, label, class_name) for p in paths)

    return samples


def infer(session, rgb, freq):
    inputs = {
        "rgb": rgb[None, ...].astype(np.float32),
        "freq": freq[None, ...].astype(np.float32),
    }

    start = time.perf_counter()
    logits = session.run(None, inputs)[0][0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    probabilities = softmax(logits)
    prediction = int(np.argmax(probabilities))

    return {
        "predicted_label_id": prediction,
        "p_real": float(probabilities[0]),
        "p_fake": float(probabilities[1]),
        "confidence": float(probabilities[prediction]),
        "inference_ms": elapsed_ms,
    }


def summarize(name: str, rows: list[dict]) -> dict:
    labels = np.asarray([r["true_label"] for r in rows], dtype=np.int64)
    predictions = np.asarray(
        [r["predicted_label_id"] for r in rows], dtype=np.int64
    )
    p_fake = np.asarray([r["p_fake"] for r in rows], dtype=np.float64)
    latency = np.asarray(
        [r["inference_ms"] for r in rows], dtype=np.float64
    )
    delta = np.asarray(
        [r["p_fake_delta_vs_original"] for r in rows], dtype=np.float64
    )
    flips = np.asarray(
        [r["prediction_flip_vs_original"] for r in rows], dtype=np.float64
    )

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    real_mask = labels == 0
    fake_mask = labels == 1

    return {
        "transform": name,
        "n_images": len(rows),
        "accuracy": safe_divide(tp + tn, len(rows)),
        "false_positive_rate_real": safe_divide(fp, tn + fp),
        "false_negative_rate_fake": safe_divide(fn, tp + fn),
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "roc_auc_fake": rank_auc(labels, p_fake),
        "true_positives_fake": tp,
        "false_negatives_fake": fn,
        "true_negatives_real": tn,
        "false_positives_real": fp,
        "mean_p_fake_on_real": float(p_fake[real_mask].mean()),
        "mean_p_fake_on_fake": float(p_fake[fake_mask].mean()),
        "mean_signed_p_fake_shift_real": float(delta[real_mask].mean()),
        "mean_abs_p_fake_shift_real": float(np.abs(delta[real_mask]).mean()),
        "mean_abs_p_fake_shift_fake": float(np.abs(delta[fake_mask]).mean()),
        "prediction_flip_rate_vs_original": float(flips.mean()),
        "median_onnx_inference_ms": float(np.median(latency)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure robustness versus image transformation severity."
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=100,
        help="Pilot sample size per class; default is 100.",
    )
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument(
        "--output-dir",
        default="ml_core/experiments/results/severity_sweep_pilot_100",
    )
    args = parser.parse_args()

    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be at least 1.")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    image_size = int(config["data"]["image_size"])

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    inputs = {item.name: item for item in session.get_inputs()}
    expected_shape = [1, 3, image_size, image_size]
    if set(inputs) != {"rgb", "freq"}:
        raise RuntimeError(f"Unexpected ONNX inputs: {list(inputs)}")

    for name in ("rgb", "freq"):
        if inputs[name].shape != expected_shape:
            raise RuntimeError(
                f"Unexpected {name} shape {inputs[name].shape}; "
                f"expected {expected_shape}"
            )

    samples = collect_samples(args.limit_per_class, args.seed)
    transforms = build_transforms()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dimensions = Counter()
    for path, _, _ in samples:
        with Image.open(path) as image:
            dimensions[f"{image.width}x{image.height}"] += 1

    print(f"Image size fed to model: {image_size}x{image_size}")
    print("Source dimensions (most common):", dimensions.most_common(8))
    print("ONNX providers:", session.get_providers())
    print("Transformation conditions:", len(transforms))
    print("Running severity sweep...")

    rows = []
    baseline_by_id = {}

    for index, (path, label, class_name) in enumerate(samples, start=1):
        with Image.open(path) as opened:
            original = opened.convert("RGB")

        image_id = f"{class_name}/{path.name}"

        for transform_name, transform in transforms:
            transformed = transform(original)
            rgb, freq = prepare_image(transformed, image_size)
            result = infer(session, rgb, freq)

            if transform_name == "original":
                baseline_by_id[image_id] = result

            rows.append({
                "image_id": image_id,
                "source_filename": path.name,
                "true_class": class_name,
                "true_label": int(label),
                "transform": transform_name,
                "predicted_class": CLASS_NAMES[result["predicted_label_id"]],
                "predicted_label_id": result["predicted_label_id"],
                "correct": int(result["predicted_label_id"] == label),
                "confidence": result["confidence"],
                "p_real": result["p_real"],
                "p_fake": result["p_fake"],
                "inference_ms": result["inference_ms"],
            })

        if index % max(1, len(samples) // 10) == 0 or index == len(samples):
            print(f"Processed {index}/{len(samples)} source images.")

    for row in rows:
        baseline = baseline_by_id[row["image_id"]]
        row["p_fake_delta_vs_original"] = row["p_fake"] - baseline["p_fake"]
        row["prediction_flip_vs_original"] = int(
            row["predicted_label_id"] != baseline["predicted_label_id"]
        )

    summary = [
        summarize(name, [r for r in rows if r["transform"] == name])
        for name, _ in transforms
    ]

    write_csv(output_dir / "predictions.csv", rows)
    write_csv(output_dir / "summary_by_transform.csv", summary)

    metadata = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "experiment": "DeepGuard AI severity-response sweep",
        "data_split": "validation",
        "model_path": str(MODEL_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
        "image_size": image_size,
        "class_names": CLASS_NAMES,
        "seed": args.seed,
        "limit_per_class": args.limit_per_class,
        "selected_source_images": len(samples),
        "source_dimension_counts": dict(dimensions),
        "transform_descriptions": {
            "jpeg_quality_*": "Re-encode at specified JPEG quality with subsampling 4:2:0.",
            "gaussian_blur_r*": "PIL Gaussian blur at the specified source-image pixel radius.",
            "downscale_*_restore": "Downscale by the specified ratio, then restore original dimensions.",
            "original": "Unmodified source image.",
        },
        "transform_order": [name for name, _ in transforms],
        "providers": session.get_providers(),
        "prediction_rows": len(rows),
        "notes": [
            "Transforms are applied before the project's normal 224x224 model preprocessing.",
            "Blur radii are expressed in source-image pixels; source dimensions are recorded above.",
            "All variants are paired transformations of the same source images.",
            "Only validation data is used in this diagnostic experiment.",
            "Latency measures ONNX inference only, excluding transformations and preprocessing.",
            "No model or original image file is modified.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n=== Severity Sweep Summary ===")
    for item in summary:
        print(
            f"{item['transform']:24s} "
            f"acc={item['accuracy']:.4f} "
            f"FPR_real={item['false_positive_rate_real']:.4f} "
            f"recall_fake={item['recall_fake']:.4f} "
            f"AUC={item['roc_auc_fake']:.4f} "
            f"mean_shift_real={item['mean_signed_p_fake_shift_real']:.4f} "
            f"flip_rate={item['prediction_flip_rate_vs_original']:.4f}"
        )

    print(f"\nSaved results to: {output_dir}")
    print("Model and original images were not modified.")


if __name__ == "__main__":
    main()
