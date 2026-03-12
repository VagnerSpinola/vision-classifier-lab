from __future__ import annotations

import numpy as np
import torch

from app.core.logging import get_logger
from app.core.settings import settings
from app.inference.contracts import PredictionResult, PredictorMetadata
from app.inference.postprocess import extract_top_k
from app.inference.preprocess import load_image_from_bytes, preprocess_image


LOGGER = get_logger(__name__)


class ONNXImagePredictor:
    def __init__(self) -> None:
        self.session = None
        self.class_names = settings.class_names
        self.image_size = settings.image_size
        self.model_version = settings.model_version
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        import onnxruntime as ort

        self.session = ort.InferenceSession(settings.model_onnx_path, providers=["CPUExecutionProvider"])
        self._loaded = True
        LOGGER.info("loaded ONNX model from %s", settings.model_onnx_path)

    def predict(self, image_bytes: bytes) -> PredictionResult:
        if not self._loaded:
            self.load()
        if self.session is None:
            raise RuntimeError("ONNX runtime session not available")

        image = load_image_from_bytes(image_bytes)
        batch = preprocess_image(image, image_size=self.image_size).cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: batch})
        logits = torch.from_numpy(outputs[0])
        top_k = extract_top_k(logits, self.class_names, settings.prediction_top_k)
        predicted_class = top_k[0].class_name
        confidence = top_k[0].score
        return PredictionResult(predicted_class=predicted_class, confidence=confidence, top_k=top_k)

    def metadata(self) -> PredictorMetadata:
        return PredictorMetadata(
            backend="onnx",
            loaded=self._loaded,
            class_names=tuple(self.class_names),
            model_version=self.model_version,
        )