"""Train OCTNet on the retinal OCT dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from models import OCTNet
from utils.config import load_config, save_config
from utils.data import build_dataloaders, build_datasets
from utils.engine import train_model
from utils.logging import configure_logging
from utils.seed import set_seed
from utils.visualization import plot_training_history


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(path_value: str) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "octnet.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root containing train/, val/, and test/ directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for logs, checkpoint, config, and training curves.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full training and validation pipeline."""
    args = parse_args()
    config = load_config(args.config)
    training_config = config["training"]
    data_root = _project_path(args.data_root or config["data"]["root"])
    output_dir = _project_path(args.output_dir or config["output"]["root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "train.log")

    set_seed(
        seed=int(training_config["seed"]),
        deterministic=bool(training_config["deterministic"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    datasets_by_split = build_datasets(data_root, config["data"])
    dataloaders = build_dataloaders(
        datasets_by_split, config["loader"], device=device
    )
    class_names = datasets_by_split["train"].classes
    dataset_sizes = {name: len(dataset) for name, dataset in datasets_by_split.items()}
    logger.info("Classes: %s", class_names)
    logger.info("Dataset sizes: %s", dataset_sizes)

    model = OCTNet(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_config["label_smoothing"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config["scheduler_t_max"]),
        eta_min=float(training_config["scheduler_eta_min"]),
    )
    checkpoint_path = output_dir / "best.pt"
    save_config(config, output_dir / "config.yaml")

    history, best_accuracy, best_epoch = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=int(training_config["epochs"]),
        class_names=class_names,
        config=config,
        checkpoint_path=str(checkpoint_path),
        logger=logger,
    )
    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    plot_training_history(history, output_dir / "training_curves.png")
    logger.info(
        "Training complete. Best validation accuracy: %.4f at epoch %d.",
        best_accuracy,
        best_epoch,
    )
    logger.info("Best checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
