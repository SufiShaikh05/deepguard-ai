from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from robustness_benchmark import (
    PROJECT_ROOT,
    ML_CORE_DIR,
    MODEL_PATH,
    CONFIG_PATH,
    TRANSFORMS,
    prepare_image,
    softmax,
    rank_auc,
    safe_divide,
    sha256_file,
    write_csv,
)
from utils.constants import CLASS_NAMES

VALIDATION_DIR = ML_CORE_DIR / "data" / "processed" / "val"

TRANSFORM_FUNCTIONS = dict(TRANSFORMS)
PERTURBATIONS = [
    "jpeg_q50",
    "resize_half_restore",
    "gaussian_smoothing_r1",
]


def collect_samples(limit_per_class: int | None, seed: int):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    samples = []

    for label, class_name in enumerate(CLASS_NAMES):
        folder = VALIDATION_DIR / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Validation folder not found: {folder}")

        paths = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

        if limit_per_class is not None and len(paths) > limit_per_class:
            rng = random.Random(seed + label)
            paths = sorted(rng.sample(paths, limit_per_class))

        if not paths:
            raise RuntimeError(f"No images found in {folder}")

        print(f"Selected {len(paths)} {class_name} validation images.")
        samples.extend((path, label, class_name) for path in paths)

    return samples


def infer(session, rgb, freq):
    start = time.perf_counter()
    logits = session.run(
        None,
        {
            "rgb": rgb[None, ...].astype(np.float32),
            "freq": freq[None, ...].astype(np.float32),
        },
    )[0][0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    probabilities = softmax(logits)

    return {
        "predicted_label_id": int(np.argmax(probabilities)),
        "p_real": float(probabilities[0]),
        "p_fake": float(probabilities[1]),
        "confidence": float(np.max(probabilities)),
        "inference_ms": float(elapsed_ms),
    }


def make_row(image_id, class_name, true_label, condition, variant, result, baseline):
    pred = result["predicted_label_id"]
    return {
        "image_id": image_id,
        "true_class": class_name,
        "true_label": int(true_label),
        "condition": condition,
        "variant": variant,
        "predicted_class": CLASS_NAMES[pred],
        "predicted_label_id": pred,
        "correct": int(pred == true_label),
        "p_real": result["p_real"],
        "p_fake": result["p_fake"],
        "confidence": result["confidence"],
        "p_fake_delta_vs_original": result["p_fake"] - baseline["p_fake"],
        "prediction_flip_vs_original": int(
            pred != baseline["predicted_label_id"]
        ),
        "onnx_inference_ms": result["inference_ms"],
    }


def summarize_condition(condition, rows):
    labels = np.asarray([r["true_label"] for r in rows], dtype=np.int64)
    predictions = np.asarray(
        [r["predicted_label_id"] for r in rows], dtype=np.int64
    )
    p_fake = np.asarray([r["p_fake"] for r in rows], dtype=np.float64)
    latency = np.asarray(
        [r["onnx_inference_ms"] for r in rows], dtype=np.float64
    )
    flips = np.asarray(
        [r["prediction_flip_vs_original"] for r in rows], dtype=np.float64
    )
    deltas = np.asarray(
        [r["p_fake_delta_vs_original"] for r in rows], dtype=np.float64
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
        "condition": condition,
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
        "mean_abs_p_fake_delta_real": float(np.abs(deltas[real_mask]).mean()),
        "mean_abs_p_fake_delta_fake": float(np.abs(deltas[fake_mask]).mean()),
        "prediction_flip_rate_vs_original": float(flips.mean()),
        "median_onnx_inference_ms": float(np.median(latency)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic input-stream sensitivity experiment."
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=100,
        help="Pilot size per class; use no flag only when ready for the full validation set.",
    )
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument(
        "--output-dir",
        default="ml_core/experiments/results/stream_sensitivity_pilot_100",
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
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_functions = {
        name: TRANSFORM_FUNCTIONS[name] for name in PERTURBATIONS
    }
    rows = []

    print(f"Model: {MODEL_PATH}")
    print(f"Image size: {image_size} x {image_size}")
    print(f"Providers: {session.get_providers()}")
    print("Running baseline and RGB/frequency input interventions...")

    for index, (path, label, class_name) in enumerate(samples, start=1):
        with Image.open(path) as opened:
            original = opened.convert("RGB")

        image_id = f"{class_name}/{path.name}"
        original_rgb, original_freq = prepare_image(original, image_size)

        baseline = infer(session, original_rgb, original_freq)
        rows.append(
            make_row(
                image_id, class_name, label, "original",
                "original", baseline, baseline
            )
        )

        for transform_name, transform in selected_functions.items():
            transformed = transform(original)
            transformed_rgb, transformed_freq = prepare_image(
                transformed, image_size
            )

            interventions = [
                ("both", transformed_rgb, transformed_freq),
                ("rgb_only", transformed_rgb, original_freq),
                ("freq_only", original_rgb, transformed_freq),
            ]

            for variant, rgb, freq in interventions:
                result = infer(session, rgb, freq)
                condition = f"{transform_name}__{variant}"
                rows.append(
                    make_row(
                        image_id, class_name, label, condition,
                        variant, result, baseline
                    )
                )

        if index % max(1, len(samples) // 10) == 0 or index == len(samples):
            print(f"Processed {index}/{len(samples)} validation images.")

    conditions = ["original"] + [
        f"{name}__{variant}"
        for name in PERTURBATIONS
        for variant in ("both", "rgb_only", "freq_only")
    ]
    summary = [
        summarize_condition(
            condition,
            [row for row in rows if row["condition"] == condition],
        )
        for condition in conditions
    ]

    write_csv(output_dir / "predictions.csv", rows)
    write_csv(output_dir / "summary_by_condition.csv", summary)

    metadata = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "experiment": "DeepGuard AI stream sensitivity diagnostic",
        "data_split": "validation",
        "model_path": str(MODEL_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
        "config_path": str(CONFIG_PATH),
        "image_size": image_size,
        "class_names": CLASS_NAMES,
        "sampling_seed": args.seed,
        "limit_per_class": args.limit_per_class,
        "selected_source_images": len(samples),
        "conditions": conditions,
        "provider": session.get_providers(),
        "predictions_rows": len(rows),
        "notes": [
            "Validation images only; test data is not used by this script.",
            "RGB and frequency preprocessing reuses existing project utilities.",
            "Single-stream interventions deliberately pair inputs from different versions of an image.",
            "Single-stream results are diagnostic counterfactuals, not deployment performance or causal proof.",
            "Inference latency excludes preprocessing and image transformation time.",
            "No model checkpoint, ONNX model, or source image is modified.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n=== Summary by Condition ===")
    for item in summary:
        print(
            f"{item['condition']:34s} "
            f"acc={item['accuracy']:.4f} "
            f"FPR_real={item['false_positive_rate_real']:.4f} "
            f"recall_fake={item['recall_fake']:.4f} "
            f"AUC={item['roc_auc_fake']:.4f} "
            f"mean_delta_real={item['mean_abs_p_fake_delta_real']:.4f} "
            f"flip_rate={item['prediction_flip_rate_vs_original']:.4f}"
        )

    print(f"\nSaved results to: {output_dir}")
    print("Model and original images were not modified.")


if __name__ == "__main__":
    main()

