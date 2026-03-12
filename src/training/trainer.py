from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from app.core.logging import get_logger
from src.evaluation.metrics import compute_classification_metrics
from src.training.callbacks import EarlyStoppingCallback, ModelCheckpointCallback


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EpochResult:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class MetricTracker(Protocol):
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        ...


@dataclass(slots=True)
class TrainingHistory:
    epochs: list[dict[str, float | int]] = field(default_factory=list)
    best_checkpoint: str | None = None
    best_metric: float | None = None


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        class_names: list[str],
        checkpoint_path: str | Path,
        registry_path: str | Path | None,
        architecture: str,
        image_size: int,
        best_metric: str,
        early_stopping_patience: int,
        tracker: MetricTracker | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.class_names = class_names
        self.checkpoint_path = Path(checkpoint_path)
        self.registry_path = Path(registry_path) if registry_path is not None else None
        self.architecture = architecture
        self.image_size = image_size
        self.best_metric = best_metric
        self.tracker = tracker
        self._validate_monitor_name(best_metric)
        self.checkpoint_callback = ModelCheckpointCallback(
            checkpoint_path=self.checkpoint_path,
            registry_path=self.registry_path,
            monitor=best_metric,
            mode="max",
            architecture=architecture,
            class_names=class_names,
            image_size=image_size,
        )
        self.early_stopping = EarlyStoppingCallback(
            monitor=best_metric,
            mode="max",
            patience=early_stopping_patience,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> TrainingHistory:
        history = TrainingHistory()

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_result = self._run_epoch(train_loader, training=True)
            val_result = self._run_epoch(val_loader, training=False)

            epoch_summary = self._build_epoch_summary(epoch, train_result, val_result)
            history.epochs.append(epoch_summary)

            LOGGER.info(
                (
                    "epoch=%s train_loss=%.4f train_acc=%.4f train_f1=%.4f "
                    "val_loss=%.4f val_acc=%.4f val_f1=%.4f"
                ),
                epoch,
                train_result.loss,
                train_result.accuracy,
                train_result.f1_score,
                val_result.loss,
                val_result.accuracy,
                val_result.f1_score,
            )

            if self.tracker is not None:
                self.tracker.log_metrics(epoch_summary, step=epoch)

            validation_metrics = self._to_metric_dict(val_result)
            saved_checkpoint = self.checkpoint_callback.maybe_save(self.model, metrics=validation_metrics, epoch=epoch)
            if saved_checkpoint is not None:
                history.best_checkpoint = str(saved_checkpoint)
                history.best_metric = getattr(val_result, self.best_metric)
                LOGGER.info("saved new best checkpoint to %s", saved_checkpoint)

            if self.early_stopping.should_stop(validation_metrics):
                LOGGER.info("early stopping triggered at epoch=%s", epoch)
                break

        return history

    def evaluate(self, dataloader: DataLoader, split_name: str = "test") -> dict[str, float]:
        result = self._run_epoch(dataloader, training=False)
        metrics = {
            f"{split_name}_loss": result.loss,
            f"{split_name}_accuracy": result.accuracy,
            f"{split_name}_precision": result.precision,
            f"{split_name}_recall": result.recall,
            f"{split_name}_f1_score": result.f1_score,
        }
        if self.tracker is not None:
            self.tracker.log_metrics(metrics)
        return metrics

    @staticmethod
    def _validate_monitor_name(metric_name: str) -> None:
        supported = {"loss", "accuracy", "precision", "recall", "f1_score"}
        if metric_name not in supported:
            raise ValueError(f"Unsupported monitor metric '{metric_name}'. Expected one of {sorted(supported)}")

    @staticmethod
    def _to_metric_dict(result: EpochResult) -> dict[str, float]:
        return {
            "loss": result.loss,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
        }

    def _build_epoch_summary(
        self,
        epoch: int,
        train_result: EpochResult,
        val_result: EpochResult,
    ) -> dict[str, float | int]:
        return {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "train_precision": train_result.precision,
            "train_recall": train_result.recall,
            "train_f1_score": train_result.f1_score,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "val_precision": val_result.precision,
            "val_recall": val_result.recall,
            "val_f1_score": val_result.f1_score,
        }

    def _run_epoch(self, dataloader: DataLoader, training: bool) -> EpochResult:
        self.model.train(mode=training)

        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        all_predictions: list[int] = []
        all_targets: list[int] = []

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if training:
                    loss.backward()
                    self.optimizer.step()

            predictions = outputs.argmax(dim=1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (predictions == targets).sum().item()
            total_examples += inputs.size(0)
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

        metrics = compute_classification_metrics(all_targets, all_predictions)

        return EpochResult(
            loss=total_loss / max(total_examples, 1),
            accuracy=total_correct / max(total_examples, 1),
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
        )