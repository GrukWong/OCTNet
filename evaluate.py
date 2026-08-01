"""Evaluate an OCTNet checkpoint on a labeled dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import OCTNet
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.data import SPLITS, build_dataloaders, build_datasets
from utils.logging import configure_logging
from utils.metrics import build_classification_metrics, collect_predictions
from utils.visualization import plot_confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(path_value: str) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line options for checkpoint evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a checkpoint produced by train.py.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "octnet.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--data-root", default=None, help="Dataset root directory.")
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="test",
        help="Labeled dataset split to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory for the report, confusion matrix, and evaluation log.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate a saved model and write standard classification artifacts."""
    args = parse_args()
    config = load_config(args.config)
    data_root = _project_path(args.data_root or config["data"]["root"])
    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "evaluate.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_by_split = build_datasets(data_root, config["data"])
    dataloaders = build_dataloaders(
        datasets_by_split, config["loader"], device=device
    )
    dataset_classes = datasets_by_split["train"].classes
    model = OCTNet(num_classes=len(dataset_classes)).to(device)
    checkpoint = load_checkpoint(_project_path(args.checkpoint), model, device)
    class_names = checkpoint.get("class_names", dataset_classes)
    if class_names != dataset_classes:
        raise ValueError(
            "Checkpoint classes do not match the dataset class mapping: "
            f"{class_names} != {dataset_classes}"
        )

    labels, predictions = collect_predictions(model, dataloaders[args.split], device)
    accuracy, report, matrix = build_classification_metrics(
        labels, predictions, class_names
    )
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    plot_confusion_matrix(matrix, class_names, output_dir / "confusion_matrix.png")
    logger.info("Device: %s", device)
    logger.info("Evaluated split: %s", args.split)
    logger.info("Global accuracy: %.4f", accuracy)
    logger.info("Classification report: %s", report_path)


if __name__ == "__main__":
    main()
