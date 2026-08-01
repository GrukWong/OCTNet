"""Training and validation loop implementation."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import save_checkpoint


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    phase: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch and return loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    running_correct = 0
    sample_count = 0

    progress_bar = tqdm(dataloader, desc=f"{phase} {epoch}", leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type="cuda", enabled=True)
            if device.type == "cuda"
            else nullcontext()
        )
        grad_context = torch.enable_grad() if is_training else torch.inference_mode()
        with grad_context, autocast_context:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if is_training:
            if scaler is None:
                raise RuntimeError("A GradScaler is required for training")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError(f"The {phase} dataloader contains no samples")
    return running_loss / sample_count, running_correct / sample_count


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader[Any]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int,
    class_names: list[str],
    config: dict[str, Any],
    checkpoint_path: str,
    logger: logging.Logger,
) -> tuple[dict[str, list[float]], float, int]:
    """Train OCTNet and persist the highest-validation-accuracy checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be at least one")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_accuracy = float("-inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            epoch=epoch,
            phase="train",
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss, val_accuracy = _run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            epoch=epoch,
            phase="val",
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        logger.info(
            "Epoch %d/%d | train loss %.4f | train acc %.4f | "
            "val loss %.4f | val acc %.4f",
            epoch,
            epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_accuracy,
                class_names=class_names,
                config=config,
            )
            logger.info(
                "Saved best checkpoint at epoch %d (val accuracy %.4f).",
                epoch,
                best_accuracy,
            )

    return history, best_accuracy, best_epoch
