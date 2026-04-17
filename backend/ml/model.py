"""PyTorch ANN for binary Zoom-call prediction."""
from __future__ import annotations
import torch
import torch.nn as nn


class ZoomNet(nn.Module):
    """Feed-forward ANN: [B, input_dim] → [B, 1] logit."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (256, 128, 64, 32),
                 dropout: float = 0.25):
        super().__init__()
        dims = [input_dim, *hidden]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no BN/Dropout on the last hidden layer
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))  # raw logits
