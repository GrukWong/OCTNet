"""Classification evaluation and report generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect ground-truth labels and predictions from a dataloader."""
    model.eval()
    predictions: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            outputs = model(inputs)
            predictions.append(outputs.argmax(dim=1).cpu().numpy())
            labels_list.append(labels.numpy())

    if not predictions:
        raise ValueError("Cannot evaluate an empty dataloader")
    return np.concatenate(labels_list), np.concatenate(predictions)


def build_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> tuple[float, str, np.ndarray]:
    """Compute global accuracy, a formatted report, and confusion matrix."""
    label_ids = list(range(len(class_names)))
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        labels=label_ids,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=label_ids)
    return float(accuracy), report, matrix
