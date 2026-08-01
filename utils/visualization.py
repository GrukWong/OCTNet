"""Result visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_history(
    history: dict[str, list[float]], output_path: str | Path
) -> None:
    """Save side-by-side training/validation accuracy and loss curves."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_accuracy"], label="Training Accuracy")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")

    axes[1].plot(epochs, history["train_loss"], label="Training Loss")
    axes[1].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray, class_names: list[str], output_path: str | Path
) -> None:
    """Save an annotated confusion-matrix heatmap."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
