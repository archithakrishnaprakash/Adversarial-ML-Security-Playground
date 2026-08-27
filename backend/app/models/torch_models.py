from __future__ import annotations

import torch
import torch.nn as nn


class LogisticRegressionTorch(nn.Module):
    """Plain multinomial logistic regression, written in torch so it's
    differentiable and can be attacked with gradient-based methods."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(self.flatten(x))


class SmallNN(nn.Module):
    """A small fully-connected network. Works for both flattened images and
    tabular feature vectors."""

    def __init__(self, in_features: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(self.flatten(x))


class SmallCNN(nn.Module):
    """A tiny CNN for 8x8 single-channel images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (N, 1, 8, 8) or (N, 8, 8) — normalize shape
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.fc(self.conv(x))
