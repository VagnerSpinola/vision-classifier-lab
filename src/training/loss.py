from __future__ import annotations

import torch
from torch import nn


def build_loss(
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    device: torch.device | None = None,
) -> nn.Module:
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)