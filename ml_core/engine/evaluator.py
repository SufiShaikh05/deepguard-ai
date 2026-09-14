"""
Evaluates a model on a dataloader without updating its weights.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()

    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_fake_probabilities: list[float] = []
    running_loss = 0.0

    for rgb, freq, labels in data_loader:
        rgb, freq, labels = rgb.to(device), freq.to(device), labels.to(device).long()

        logits = model(rgb, freq)

        if criterion is not None:
            running_loss += criterion(logits, labels).item() * rgb.size(0)

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_fake_probabilities.extend(probabilities[:, 1].cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="binary", zero_division=0
    )

    metrics = {
        "loss": running_loss / len(data_loader.dataset) if criterion is not None else 0.0,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    try:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_fake_probabilities)
    except ValueError:
        metrics["roc_auc"] = float("nan")  # only one class present so far

    return metrics
