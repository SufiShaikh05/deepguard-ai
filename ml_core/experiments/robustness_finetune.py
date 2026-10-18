from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_CORE_DIR = PROJECT_ROOT / "ml_core"

sys.path.insert(0, str(ML_CORE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import DeepfakeDataset, SUPPORTED_EXTENSIONS
from engine.trainer import Trainer
from models.dual_stream_net import DualStreamNet
from robustness_augmentations import apply_robustness_augmentation
from utils.constants import CLASS_NAMES
from utils.dct_utils import image_to_dct, normalize_dct
from utils.transforms import (
    apply_geometric_augmentation,
    normalize_rgb,
    resize_image,
    to_chw_tensor_array,
)


def resolve_path(path_value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class RobustnessTrainingDataset(DeepfakeDataset):
    """Training-only degradation augmentation before normal model resizing."""

    def __init__(
        self,
        root_dir: str,
        image_size: int,
        robustness_probability: float,
    ) -> None:
        super().__init__(root_dir, image_size=image_size, train=True)
        self.robustness_probability = robustness_probability

    def __getitem__(self, index: int):
        file_path, label = self.samples[index]

        try:
            with Image.open(file_path) as opened:
                image = opened.convert("RGB")
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Could not read training image '{file_path}': {error}"
            ) from error

        # Apply the robustness transformation to the source image first.
        image, _ = apply_robustness_augmentation(
            image,
            probability=self.robustness_probability,
        )

        # Preserve the original project's preprocessing sequence.
        image = resize_image(image, self.image_size)
        image = apply_geometric_augmentation(image)
        image_array = np.asarray(image, dtype=np.uint8)

        rgb_array = normalize_rgb(image_array)
        rgb_tensor = torch.from_numpy(
            to_chw_tensor_array(rgb_array)
        ).float()

        dct_array = normalize_dct(image_to_dct(image_array))
        freq_tensor = torch.from_numpy(
            to_chw_tensor_array(dct_array)
        ).float()

        return rgb_tensor, freq_tensor, label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated DeepGuard AI robustness fine-tuning experiment."
    )
    parser.add_argument(
        "--config",
        default="ml_core/configs/config.yaml",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="ml_core/checkpoints/best_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        default="ml_core/experiments/results/robustness_finetune_v1",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--robustness-probability", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Explicitly allow real training on CPU; normally use Colab GPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data/model and run one forward pass without training or saving checkpoints.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must both be at least 1.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if not 0.0 <= args.robustness_probability <= 1.0:
        parser.error("--robustness-probability must be between 0 and 1.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    config_path = resolve_path(args.config)
    baseline_path = resolve_path(args.baseline_checkpoint)
    output_dir = resolve_path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {baseline_path}"
        )
    if checkpoint_dir.resolve() == baseline_path.parent.resolve():
        raise RuntimeError(
            "Output checkpoint directory matches the baseline directory. "
            "Choose a separate output directory."
        )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    image_size = int(config["data"]["image_size"])

    def configured_data_path(key: str) -> Path:
        return resolve_path(config["data"][key], base=ML_CORE_DIR)

    train_dir = configured_data_path("train_dir")
    val_dir = configured_data_path("val_dir")

    for path in (train_dir, val_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {path}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. No training started."
            )
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if (
        not args.dry_run
        and device.type != "cuda"
        and not args.allow_cpu
    ):
        raise RuntimeError(
            "No GPU is available. Training has not started. "
            "Run the real experiment in Colab with a GPU, or explicitly "
            "pass --allow-cpu if you accept a potentially very slow run."
        )

    seed_everything(args.seed)

    train_dataset = RobustnessTrainingDataset(
        str(train_dir),
        image_size,
        args.robustness_probability,
    )
    val_dataset = DeepfakeDataset(
        str(val_dir),
        image_size=image_size,
        train=False,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Training or validation dataset is empty.")

    model_config = copy.deepcopy(config)

    # The baseline state dict contains trained weights for the full model.
    # Do not download or substitute new pretrained weights before loading it.
    model_config["model"]["spatial_pretrained"] = False
    model = DualStreamNet.from_config(model_config)

    state_dict = torch.load(
        baseline_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    print("=== Robustness Fine-Tuning Setup ===")
    print(f"Device: {device}")
    print(f"Configuration: {config_path}")
    print(f"Baseline checkpoint: {baseline_path}")
    print(f"Baseline SHA-256: {sha256_file(baseline_path)}")
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Image size: {image_size} x {image_size}")
    print(f"Robustness augmentation probability: {args.robustness_probability}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {output_dir}")

    # Validate both data paths and model input compatibility without training.
    train_sample = train_dataset[0]
    val_sample = val_dataset[0]

    for name, sample in (("training", train_sample), ("validation", val_sample)):
        rgb, freq, label = sample
        assert tuple(rgb.shape) == (3, image_size, image_size), (
            name, tuple(rgb.shape)
        )
        assert tuple(freq.shape) == (3, image_size, image_size), (
            name, tuple(freq.shape)
        )
        assert int(label) in (0, 1), (name, label)
        print(
            f"{name.capitalize()} sample verified: "
            f"rgb={tuple(rgb.shape)}, freq={tuple(freq.shape)}, label={label}"
        )

    model.eval()
    with torch.inference_mode():
        logits = model(
            train_sample[0].unsqueeze(0).to(device),
            train_sample[1].unsqueeze(0).to(device),
        )

    if tuple(logits.shape) != (1, 2):
        raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")

    print(f"Baseline model forward pass verified: logits={tuple(logits.shape)}")

    if args.dry_run:
        print("DRY RUN PASSED. No training was performed and no checkpoints were saved.")
        return

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=train_generator,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting fine-tuning from the existing baseline checkpoint.")
    print("Validation images use the original, non-augmented pipeline.")
    print("Checkpoints will be saved only inside the experiment output folder.")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=float(config["training"]["weight_decay"]),
        checkpoint_dir=str(checkpoint_dir),
        early_stopping_patience=args.patience,
    )
    trainer.fit(epochs=args.epochs)

    metadata = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "experiment": "DeepGuard AI robustness augmentation fine-tuning",
        "baseline_checkpoint": str(baseline_path),
        "baseline_sha256": sha256_file(baseline_path),
        "config_path": str(config_path),
        "training_directory": str(train_dir),
        "validation_directory": str(val_dir),
        "image_size": image_size,
        "class_names": CLASS_NAMES,
        "device": str(device),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate,
        "robustness_probability": args.robustness_probability,
        "early_stopping_patience": args.patience,
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "checkpoint_directory": str(checkpoint_dir),
        "checkpoint_selection": (
            "The existing Trainer selects best_model.pth by clean validation accuracy. "
            "Robustness must be evaluated separately; this is not a robustness-aware selector."
        ),
        "outputs": ["checkpoints/best_model.pth", "checkpoints/last_model.pth"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nTraining finished. Results saved under: {output_dir}")
    print("The original baseline checkpoint and deployed ONNX model were not overwritten.")


if __name__ == "__main__":
    main()
