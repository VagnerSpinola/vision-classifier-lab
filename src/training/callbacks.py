from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch import nn

from src.training.checkpoint import save_checkpoint


@dataclass(slots=True)
class ModelCheckpointCallback:
    checkpoint_path: Path
    registry_path: Path | None
    monitor: str
    mode: str
    architecture: str
    class_names: list[str]
    image_size: int
    best_score: float | None = None

    def maybe_save(self, model: nn.Module, metrics: dict[str, float], epoch: int) -> Path | None:
        score = metrics[self.monitor]
        if self.best_score is None:
            is_best = True
        elif self.mode == "max":
            is_best = score >= self.best_score
        else:
            is_best = score <= self.best_score

        if not is_best:
            return None

        self.best_score = score
        return save_checkpoint(
            model=model,
            architecture=self.architecture,
            class_names=self.class_names,
            image_size=self.image_size,
            checkpoint_path=self.checkpoint_path,
            registry_path=self.registry_path,
            metrics=metrics,
            epoch=epoch,
        )


@dataclass(slots=True)
class EarlyStoppingCallback:
    monitor: str
    mode: str
    patience: int
    best_score: float | None = None
    wait_count: int = 0

    def should_stop(self, metrics: dict[str, float]) -> bool:
        score = metrics[self.monitor]
        if self.best_score is None:
            self.best_score = score
            self.wait_count = 0
            return False

        improved = score >= self.best_score if self.mode == "max" else score <= self.best_score
        if improved:
            self.best_score = score
            self.wait_count = 0
            return False

        self.wait_count += 1
        return self.wait_count >= self.patience