from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


InferenceBackend = Literal["torch", "onnx"]


@dataclass(slots=True, frozen=True)
class TopPrediction:
    class_name: str
    score: float


@dataclass(slots=True, frozen=True)
class PredictionResult:
    predicted_class: str
    confidence: float
    top_k: list[TopPrediction]


@dataclass(slots=True, frozen=True)
class PredictorMetadata:
    backend: InferenceBackend
    loaded: bool
    class_names: tuple[str, ...]
    model_version: str