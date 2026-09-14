"""
Entry point for training. Run with: python train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import DeepfakeDataset
from engine.trainer import Trainer
from models.dual_stream_net import DualStreamNet


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available - falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def main(config_path: str) -> None:
    set_seed()
    config = load_config(config_path)

    device = resolve_device(config["training"]["device"])
    print(f"Using device: {device}")

    data_cfg = config["data"]
    train_dataset = DeepfakeDataset(data_cfg["train_dir"], data_cfg["image_size"], train=True)
    val_dataset = DeepfakeDataset(data_cfg["val_dir"], data_cfg["image_size"], train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=data_cfg["batch_size"], shuffle=True,
        num_workers=data_cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=data_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )

    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    model = DualStreamNet.from_config(config)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        checkpoint_dir=config["paths"]["checkpoint_dir"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
    )
    trainer.fit(epochs=config["training"]["epochs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepGuard AI's Dual-Stream model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
