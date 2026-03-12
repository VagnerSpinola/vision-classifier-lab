from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from app.core.config import ExperimentConfig


class MLflowTracker:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._active_run: Any = None

    def __enter__(self) -> "MLflowTracker":
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        self._active_run = mlflow.start_run(run_name=self.config.mlflow.run_name or self.config.experiment_name)
        mlflow.log_params(self.config.to_flat_dict())
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is not None:
            mlflow.set_tag("run_status", "failed")
            mlflow.log_param("exception_type", getattr(exc_type, "__name__", "unknown"))
        mlflow.end_run()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def log_artifacts(self, paths: list[str | Path]) -> None:
        for path in paths:
            self.log_artifact(path)