from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch import nn

from app.models.classifier import checkpoint_payload


def save_checkpoint(
    model: nn.Module,
    architecture: str,
    class_names: list[str],
    image_size: int,
    checkpoint_path: str | Path,
    registry_path: str | Path | None,
    metrics: dict[str, float],
    epoch: int,
) -> Path:
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = checkpoint_payload(
        model=model,
        architecture=architecture,
        class_names=class_names,
        image_size=image_size,
        model_version=f"epoch-{epoch}",
    )
    payload["metrics"] = metrics
    payload["epoch"] = epoch
    torch.save(payload, target)

    if registry_path is not None:
        registry_target = Path(registry_path)
        registry_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, registry_target)

    return target