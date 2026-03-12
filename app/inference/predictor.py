from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from app.core.settings import settings
from app.core.logging import get_logger
from app.inference.contracts import InferenceBackend, PredictionResult, PredictorMetadata
from app.inference.postprocess import extract_prediction, extract_top_k
from app.inference.preprocess import load_image_from_bytes, preprocess_image
from app.models.classifier import load_model_from_checkpoint
from src.utils.device import get_device


LOGGER = get_logger(__name__)


if TYPE_CHECKING:
    from app.inference.onnx_predictor import ONNXImagePredictor


class TorchImagePredictor:
    def __init__(self) -> None:
        self.device = get_device()
        self.model: torch.nn.Module | None = None
        self.class_names: list[str] = settings.class_names
        self.image_size: int = settings.image_size
        self.model_version: str = settings.model_version
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        model, checkpoint = load_model_from_checkpoint(
            checkpoint_path=settings.model_checkpoint,
            device=self.device,
        )
        self.model = model
        self.class_names = list(checkpoint.get("class_names", self.class_names))
        self.image_size = int(checkpoint.get("image_size", self.image_size))
        self.model_version = str(checkpoint.get("model_version", self.model_version))
        self._loaded = True
        LOGGER.info("loaded inference checkpoint from %s", settings.model_checkpoint)

    def predict(self, image_bytes: bytes) -> PredictionResult:
        if not self._loaded:
            self.load()
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        image = load_image_from_bytes(image_bytes)
        batch = preprocess_image(image, image_size=self.image_size).to(self.device)

        with torch.inference_mode():
            logits = self.model(batch)

        predicted_class, confidence = extract_prediction(logits, self.class_names)
        top_k = extract_top_k(logits, self.class_names, settings.prediction_top_k)
        return PredictionResult(predicted_class=predicted_class, confidence=confidence, top_k=top_k)

    def predict_batch(self, image_payloads: list[bytes]) -> list[PredictionResult]:
        results: list[PredictionResult] = []
        for image_payload in image_payloads:
            results.append(self.predict(image_payload))
        return results

    def metadata(self) -> PredictorMetadata:
        return PredictorMetadata(
            backend="torch",
            loaded=self._loaded,
            class_names=tuple(self.class_names),
            model_version=self.model_version,
        )


class PredictionService:
    def __init__(self) -> None:
        self._torch_predictor = TorchImagePredictor()
        self._onnx_predictor: ONNXImagePredictor | None = None

    def _backend(self) -> InferenceBackend:
        return settings.inference_backend

    def _get_onnx_predictor(self) -> ONNXImagePredictor:
        if self._onnx_predictor is None:
            from app.inference.onnx_predictor import ONNXImagePredictor

            self._onnx_predictor = ONNXImagePredictor()
        return self._onnx_predictor

    def load(self) -> None:
        if self._backend() == "onnx":
            self._get_onnx_predictor().load()
            return
        self._torch_predictor.load()

    def predict_bytes(self, image_bytes: bytes) -> PredictionResult:
        if self._backend() == "onnx":
            return self._get_onnx_predictor().predict(image_bytes)
        return self._torch_predictor.predict(image_bytes)

    def predict_batch(self, image_payloads: list[bytes]) -> list[PredictionResult]:
        return [self.predict_bytes(payload) for payload in image_payloads]

    def metadata(self) -> PredictorMetadata:
        if self._backend() == "onnx":
            if self._onnx_predictor is not None:
                return self._onnx_predictor.metadata()
            return PredictorMetadata(
                backend="onnx",
                loaded=False,
                class_names=tuple(settings.class_names),
                model_version=settings.model_version,
            )
        return self._torch_predictor.metadata()


prediction_service = PredictionService()