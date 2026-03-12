from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.settings import PROJECT_ROOT


@dataclass(slots=True)
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_degrees: int = 15
    color_jitter: float = 0.2
    random_resized_crop_scale: tuple[float, float] = (0.8, 1.0)


@dataclass(slots=True)
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    external_dir: str = "data/external"
    class_names: list[str] = field(default_factory=list)
    imbalance_strategy: str = "none"
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass(slots=True)
class ModelConfig:
    architecture: str
    num_classes: int
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout: float = 0.2
    checkpoint_path: str = "models/checkpoints/best_model.pt"
    registry_path: str = "models/registry/latest.pt"


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    image_size: int
    num_workers: int
    label_smoothing: float = 0.0
    best_metric: str = "f1_score"
    early_stopping_patience: int = 5


@dataclass(slots=True)
class EvaluationConfig:
    output_dir: str = "data/processed/evaluation"
    top_k: int = 3
    max_error_samples: int = 25


@dataclass(slots=True)
class MlflowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "vision-mlops-classifier"
    run_name: str | None = None


@dataclass(slots=True)
class ExportConfig:
    onnx_path: str = "models/exported/model.onnx"
    opset_version: int = 17
    dynamic_axes: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    mlflow: MlflowConfig
    export: ExportConfig

    def to_flat_dict(self) -> dict[str, Any]:
        flattened: dict[str, Any] = {"experiment_name": self.experiment_name, "seed": self.seed}
        for section_name in ("data", "model", "training", "evaluation", "mlflow", "export"):
            section = asdict(getattr(self, section_name))
            for key, value in _flatten_mapping(section).items():
                flattened[f"{section_name}.{key}"] = value
        return flattened


def _flatten_mapping(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        composite_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_mapping(value, composite_key))
        else:
            flattened[composite_key] = value
    return flattened


def _read_yaml(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of config file: {path}")
    return data


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    raw_config = _read_yaml(config_path)

    augmentation = AugmentationConfig(**raw_config.get("data", {}).get("augmentation", {}))
    data_config = DataConfig(**{**raw_config["data"], "augmentation": augmentation})
    model_config = ModelConfig(**raw_config["model"])
    training_config = TrainingConfig(**raw_config["training"])
    evaluation_config = EvaluationConfig(**raw_config.get("evaluation", {}))
    mlflow_config = MlflowConfig(**raw_config.get("mlflow", {}))
    export_config = ExportConfig(**raw_config.get("export", {}))

    if not data_config.class_names:
        train_dir = PROJECT_ROOT / data_config.processed_dir / "train"
        data_config.class_names = discover_class_names(train_dir)

    return ExperimentConfig(
        experiment_name=raw_config["experiment_name"],
        seed=int(raw_config.get("seed", 42)),
        data=data_config,
        model=model_config,
        training=training_config,
        evaluation=evaluation_config,
        mlflow=mlflow_config,
        export=export_config,
    )


def discover_class_names(train_dir: Path) -> list[str]:
    if not train_dir.exists():
        return []
    return sorted(entry.name for entry in train_dir.iterdir() if entry.is_dir())