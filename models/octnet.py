"""ConvNeXt-inspired OCTNet architecture used in the original experiment."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = channels // reduction
        if hidden_channels < 1:
            raise ValueError("channels must be at least as large as reduction")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply learned channel weights to a feature map."""
        batch_size, channels, _, _ = inputs.shape
        weights = self.avg_pool(inputs).view(batch_size, channels)
        weights = self.fc(weights).view(batch_size, channels, 1, 1)
        return inputs * weights


class ConvNeXtBlock(nn.Module):
    """Residual depthwise-convolution block from the original OCTNet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.pwconv1 = nn.Conv2d(channels, 4 * channels, kernel_size=1)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * channels, channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return the residual block output."""
        residual = inputs
        outputs = self.dwconv(inputs)
        outputs = self.norm(outputs)
        outputs = self.pwconv1(outputs)
        outputs = self.activation(outputs)
        outputs = self.pwconv2(outputs)
        return residual + outputs


def _stage(in_channels: int, out_channels: int, use_se: bool) -> nn.Sequential:
    """Build one OCTNet feature stage without changing the original logic."""
    layers: list[nn.Module] = []
    if in_channels != out_channels:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
    layers.append(ConvNeXtBlock(out_channels))
    if use_se:
        layers.append(SEBlock(out_channels))
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class OCTNet(nn.Module):
    """Custom five-stage CNN for eight-class retinal OCT classification."""

    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _stage(3, 32, use_se=False),
            _stage(32, 64, use_se=False),
            _stage(64, 128, use_se=True),
            _stage(128, 256, use_se=True),
            _stage(256, 512, use_se=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Produce unnormalized class logits."""
        outputs = self.features(inputs)
        outputs = self.avgpool(outputs)
        return self.classifier(outputs)
