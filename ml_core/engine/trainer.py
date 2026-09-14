"""
The training loop: run every batch, update weights, check validation
performance each epoch, and keep the best checkpoint automatically.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.evaluator import evaluate


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 5,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping_patience = early_stopping_patience

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0

    def _run_one_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0

        for rgb, freq, labels in self.train_loader:
            rgb, freq, labels = rgb.to(self.device), freq.to(self.device), labels.to(self.device).long()

            self.optimizer.zero_grad()
            logits = self.model(rgb, freq)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * rgb.size(0)

        return running_loss / len(self.train_loader.dataset)

    def fit(self, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            train_loss = self._run_one_epoch()
            val_metrics = evaluate(self.model, self.val_loader, self.device, self.criterion)

            print(
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | val_accuracy={val_metrics['accuracy']:.4f}"
            )

            self._save_checkpoint("last_model.pth")

            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.epochs_without_improvement = 0
                self._save_checkpoint("best_model.pth")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"No improvement for {self.early_stopping_patience} epochs - stopping early.")
                break

    def _save_checkpoint(self, filename: str) -> None:
        torch.save(self.model.state_dict(), self.checkpoint_dir / filename)
