"""Dataset construction and image preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


SPLITS = ("train", "val", "test")


def build_transforms(data_config: dict[str, Any]) -> dict[str, transforms.Compose]:
    """Create the original experiment's train and evaluation transforms."""
    image_size = int(data_config["image_size"])
    mean = list(data_config["normalization"]["mean"])
    std = list(data_config["normalization"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    evaluation = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    training = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.4),
        ]
    )
    return {"train": training, "val": evaluation, "test": evaluation}


def build_datasets(
    data_root: str | Path, data_config: dict[str, Any]
) -> dict[str, datasets.ImageFolder]:
    """Load train, validation, and test folders and validate class mappings."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. See dataset/README.md for setup."
        )

    transform_by_split = build_transforms(data_config)
    datasets_by_split: dict[str, datasets.ImageFolder] = {}
    for split in SPLITS:
        split_path = root / split
        if not split_path.is_dir():
            raise FileNotFoundError(f"Missing dataset split directory: {split_path}")
        datasets_by_split[split] = datasets.ImageFolder(
            split_path, transform=transform_by_split[split]
        )

    train_classes = datasets_by_split["train"].classes
    for split, dataset in datasets_by_split.items():
        if dataset.classes != train_classes:
            raise ValueError(
                f"Class mapping for '{split}' differs from train: {dataset.classes}"
            )
    return datasets_by_split


def build_dataloaders(
    datasets_by_split: dict[str, datasets.ImageFolder],
    loader_config: dict[str, Any],
    device: torch.device,
) -> dict[str, DataLoader[Any]]:
    """Build platform-safe data loaders while preserving loader parameters."""
    num_workers = int(loader_config["num_workers"])
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    common_kwargs: dict[str, Any] = {
        "batch_size": int(loader_config["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": bool(loader_config["pin_memory"]) and device.type == "cuda",
    }
    if num_workers > 0:
        common_kwargs["persistent_workers"] = bool(
            loader_config["persistent_workers"]
        )
        common_kwargs["prefetch_factor"] = int(loader_config["prefetch_factor"])

    return {
        split: DataLoader(
            dataset,
            shuffle=split == "train",
            **common_kwargs,
        )
        for split, dataset in datasets_by_split.items()
    }
