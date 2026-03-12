from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from app.inference.contracts import InferenceBackend


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass(slots=True)
class AppSettings:
    project_name: str = os.getenv("PROJECT_NAME", "vision-mlops-classifier")
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))
    reload: bool = _get_bool("APP_RELOAD", False)
    metrics_path: str = os.getenv("APP_METRICS_PATH", "/metrics")
    inference_backend: InferenceBackend = os.getenv("INFERENCE_BACKEND", "torch")  # type: ignore[assignment]
    model_checkpoint: str = os.getenv(
        "MODEL_CHECKPOINT", "models/checkpoints/resnet18_best.pt"
    )
    model_onnx_path: str = os.getenv("MODEL_ONNX_PATH", "models/exported/model.onnx")
    model_version: str = os.getenv("MODEL_VERSION", "local-dev")
    model_architecture: str = os.getenv("MODEL_ARCHITECTURE", "resnet18")
    model_num_classes: int = int(os.getenv("MODEL_NUM_CLASSES", "2"))
    class_names: list[str] = field(
        default_factory=lambda: _get_list("MODEL_CLASS_NAMES", "class_a,class_b")
    )
    image_size: int = int(os.getenv("MODEL_IMAGE_SIZE", "224"))
    prediction_top_k: int = int(os.getenv("PREDICTION_TOP_K", "3"))
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    prometheus_enabled: bool = _get_bool("PROMETHEUS_ENABLED", True)


settings = AppSettings()